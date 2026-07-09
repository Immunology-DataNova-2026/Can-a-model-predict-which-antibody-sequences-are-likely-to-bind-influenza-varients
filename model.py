"""Dual-encoder antibody-antigen binding model with cross-attention fusion.

Architecture (see plan doc for full rationale):
  1. Antibody encoder and antigen encoder: two separately-weighted copies of a
     pretrained protein language model (default: ESM-2, facebook/esm2_t6_8M_UR50D).
  2. Cross-attention interaction module: stacked bidirectional cross-attention
     layers where antibody tokens attend to antigen tokens and vice versa, so the
     model learns residue-level complementarity instead of relying on pooled
     embeddings alone.
  3. Attention pooling + MLP head: pools each stream into a single vector,
     concatenates them, and predicts a binding logit (and optionally an
     affinity regression value via a second head).
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class CrossAttentionLayer(nn.Module):
    """One bidirectional cross-attention block: antibody<->antigen, each with a residual FFN."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.ab_to_ag_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ag_to_ab_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)

        self.ab_norm1 = nn.LayerNorm(hidden_size)
        self.ag_norm1 = nn.LayerNorm(hidden_size)

        self.ab_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ag_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ab_norm2 = nn.LayerNorm(hidden_size)
        self.ag_norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ab_h, ag_h, ab_key_padding_mask, ag_key_padding_mask):
        # antibody tokens attend to antigen tokens, and vice versa
        ab_attn_out, _ = self.ab_to_ag_attn(ab_h, ag_h, ag_h, key_padding_mask=ag_key_padding_mask)
        ag_attn_out, _ = self.ag_to_ab_attn(ag_h, ab_h, ab_h, key_padding_mask=ab_key_padding_mask)

        ab_h = self.ab_norm1(ab_h + self.dropout(ab_attn_out))
        ag_h = self.ag_norm1(ag_h + self.dropout(ag_attn_out))

        ab_h = self.ab_norm2(ab_h + self.dropout(self.ab_ffn(ab_h)))
        ag_h = self.ag_norm2(ag_h + self.dropout(self.ag_ffn(ag_h)))

        return ab_h, ag_h


class AttentionPooling(nn.Module):
    """Learned attention pooling over a sequence of token embeddings, mask-aware."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # h: (batch, seq, hidden), attention_mask: (batch, seq) with 1=real token
        scores = self.score(h).squeeze(-1)  # (batch, seq)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq, 1)
        return (h * weights).sum(dim=1)  # (batch, hidden)


class AntibodyAntigenBindingModel(nn.Module):
    def __init__(
        self,
        backbone: str = "facebook/esm2_t6_8M_UR50D",
        num_cross_attn_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_encoders: bool = False,
        predict_affinity: bool = False,
    ):
        super().__init__()
        self.antibody_encoder = AutoModel.from_pretrained(backbone)
        self.antigen_encoder = AutoModel.from_pretrained(backbone)
        hidden_size = self.antibody_encoder.config.hidden_size

        if freeze_encoders:
            for p in self.antibody_encoder.parameters():
                p.requires_grad = False
            for p in self.antigen_encoder.parameters():
                p.requires_grad = False

        self.cross_attn_layers = nn.ModuleList(
            [CrossAttentionLayer(hidden_size, num_heads, dropout) for _ in range(num_cross_attn_layers)]
        )

        self.ab_pool = AttentionPooling(hidden_size)
        self.ag_pool = AttentionPooling(hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.predict_affinity = predict_affinity
        if predict_affinity:
            self.affinity_head = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )

    def forward(
        self,
        antibody_input_ids,
        antibody_attention_mask,
        antigen_input_ids,
        antigen_attention_mask,
    ) -> dict:
        ab_h = self.antibody_encoder(
            input_ids=antibody_input_ids, attention_mask=antibody_attention_mask
        ).last_hidden_state
        ag_h = self.antigen_encoder(
            input_ids=antigen_input_ids, attention_mask=antigen_attention_mask
        ).last_hidden_state

        # nn.MultiheadAttention key_padding_mask expects True = ignore
        ab_key_padding_mask = antibody_attention_mask == 0
        ag_key_padding_mask = antigen_attention_mask == 0

        for layer in self.cross_attn_layers:
            ab_h, ag_h = layer(ab_h, ag_h, ab_key_padding_mask, ag_key_padding_mask)

        ab_pooled = self.ab_pool(ab_h, antibody_attention_mask)
        ag_pooled = self.ag_pool(ag_h, antigen_attention_mask)

        fused = torch.cat([ab_pooled, ag_pooled], dim=-1)
        binding_logit = self.classifier(fused).squeeze(-1)

        out = {"binding_logit": binding_logit}
        if self.predict_affinity:
            out["affinity_pred"] = self.affinity_head(fused).squeeze(-1)
        return out
