"""
Mini VL-JEPA: Non-autoregressive video->text-embedding prediction.

Architecture
------------
  [frozen] V-JEPA 2 video encoder (facebook/vjepa2-vitl-fpc64-256)
      |
      v   video tokens (B, N, D_v)
  [trainable] Query-conditioned JEPA Predictor (small transformer)
      |
      v   predicted sentence embedding  z_hat  (B, D_t)
  [frozen] Multilingual target text encoder (LaBSE / multilingual-e5)
      |
      v   target sentence embedding     z      (B, D_t)

Loss: cosine alignment + InfoNCE (in-batch negatives) + variance regularizer
      (anti-collapse, VICReg-style std term).

A tiny decoder (mT5-small w/ LoRA, trained separately in decoder.py) maps
z_hat -> text only when selective decoding fires.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class PredictorConfig:
    video_dim: int = 1024          # V-JEPA 2 ViT-L hidden size
    text_dim: int = 768            # LaBSE embedding size
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    num_latent_queries: int = 8    # learned queries that pool video tokens
    dropout: float = 0.1
    max_video_tokens: int = 1024


# ---------------------------------------------------------------------------
# Frozen encoders (wrappers)
# ---------------------------------------------------------------------------
class FrozenVideoEncoder(nn.Module):
    """V-JEPA 2 via HuggingFace transformers. Frozen; eval mode; no grads."""

    def __init__(self, model_name: str = "facebook/vjepa2-vitl-fpc64-256"):
        super().__init__()
        from transformers import AutoModel, AutoVideoProcessor

        self.processor = AutoVideoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """pixel_values_videos: (B, T, C, H, W) -> (B, N, D_v) token features."""
        out = self.model(pixel_values_videos=pixel_values_videos)
        return out.last_hidden_state


class FrozenTextTarget(nn.Module):
    """Multilingual sentence-embedding target space (the JEPA 'y encoder')."""

    def __init__(self, model_name: str = "sentence-transformers/LaBSE"):
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, texts: list[str], device) -> torch.Tensor:
        emb = self.model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True
        )
        return emb.to(device).clone()   # .clone() escapes inference mode

# ---------------------------------------------------------------------------
# Trainable JEPA predictor
# ---------------------------------------------------------------------------
class JEPAPredictor(nn.Module):
    """
    Small transformer that cross-attends learned latent queries (optionally
    conditioned on a text question, for VQA mode) over frozen video tokens,
    then projects the pooled latent to the target text-embedding space.
    """

    def __init__(self, cfg: PredictorConfig):
        super().__init__()
        self.cfg = cfg

        self.video_proj = nn.Linear(cfg.video_dim, cfg.hidden_dim)
        self.latent_queries = nn.Parameter(
            torch.randn(1, cfg.num_latent_queries, cfg.hidden_dim) * 0.02
        )
        # Optional question conditioning (VQA mode): question embedding is
        # projected and prepended to the latent queries.
        self.question_proj = nn.Linear(cfg.text_dim, cfg.hidden_dim)

        layer = nn.TransformerDecoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=cfg.num_layers)

        self.out_proj = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.text_dim),
        )

    def forward(
        self,
        video_tokens: torch.Tensor,               # (B, N, D_v)
        question_emb: torch.Tensor | None = None, # (B, D_t) or None
    ) -> torch.Tensor:
        B = video_tokens.size(0)
        mem = self.video_proj(video_tokens)               # (B, N, H)

        q = self.latent_queries.expand(B, -1, -1)         # (B, Q, H)
        if question_emb is not None:
            qcond = self.question_proj(question_emb).unsqueeze(1)  # (B,1,H)
            q = torch.cat([qcond, q], dim=1)

        h = self.decoder(tgt=q, memory=mem)               # (B, Q(+1), H)
        pooled = h.mean(dim=1)                            # (B, H)
        z_hat = self.out_proj(pooled)                     # (B, D_t)
        return F.normalize(z_hat, dim=-1)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def jepa_loss(
    z_hat: torch.Tensor,   # (B, D) predicted, L2-normalized
    z: torch.Tensor,       # (B, D) target,   L2-normalized
    temperature: float = 0.05,
    w_cos: float = 1.0,
    w_nce: float = 1.0,
    w_std: float = 0.1,
) -> dict:
    """Cosine alignment + symmetric InfoNCE + variance (anti-collapse)."""
    # 1) cosine alignment
    loss_cos = (1.0 - (z_hat * z).sum(dim=-1)).mean()

    # 2) in-batch InfoNCE (both directions)
    logits = z_hat @ z.t() / temperature            # (B, B)
    labels = torch.arange(z.size(0), device=z.device)
    loss_nce = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )

    # 3) variance term on predictions (guards against collapse)
    std = z_hat.std(dim=0)
    loss_std = F.relu(1.0 / (z_hat.size(-1) ** 0.5) - std).mean()

    total = w_cos * loss_cos + w_nce * loss_nce + w_std * loss_std
    return {
        "loss": total,
        "loss_cos": loss_cos.detach(),
        "loss_nce": loss_nce.detach(),
        "loss_std": loss_std.detach(),
    }


# ---------------------------------------------------------------------------
# Selective decoding (streaming)
# ---------------------------------------------------------------------------
class SelectiveDecoder:
    """
    Decode-only-on-change policy for streaming video:
    fire the text decoder when the predicted embedding drifts more than
    `threshold` (cosine distance) from the last *decoded* embedding.
    """

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold
        self.last_decoded: torch.Tensor | None = None

    def should_decode(self, z_hat: torch.Tensor) -> tuple[bool, float]:
        if self.last_decoded is None:
            self.last_decoded = z_hat.detach()
            return True, 1.0
        drift = 1.0 - F.cosine_similarity(
            z_hat, self.last_decoded, dim=-1
        ).item()
        if drift > self.threshold:
            self.last_decoded = z_hat.detach()
            return True, drift
        return False, drift
