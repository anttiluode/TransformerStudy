from __future__ import annotations

import math

import torch
from torch import nn

from .config import ExperimentConfig
from .episodes import episode_length


def gelu_derivative(x: torch.Tensor) -> torch.Tensor:
    """Derivative of PyTorch's default exact GELU."""
    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    return (
        0.5 * (1.0 + torch.erf(x * inv_sqrt_2))
        + x * torch.exp(-0.5 * x.square()) * inv_sqrt_2pi
    )


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model,
            cfg.heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, cfg.d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal: torch.Tensor,
        *,
        return_diagnostics: bool = False,
        mlp_mask: torch.Tensor | None = None,
    ):
        n = self.ln1(x)
        a, attention_weights = self.attn(
            n,
            n,
            n,
            attn_mask=causal,
            need_weights=return_diagnostics,
            average_attn_weights=False,
        )
        h = x + a
        mlp_input = self.ln2(h)
        preactivation = self.mlp[0](mlp_input)
        hidden = self.mlp[1](preactivation)
        if mlp_mask is not None:
            mask = torch.as_tensor(mlp_mask, device=hidden.device, dtype=hidden.dtype)
            if mask.ndim != 1 or mask.shape[0] != hidden.shape[-1]:
                raise ValueError("MLP mask must be a vector with mlp_hidden entries")
            hidden = hidden * mask.view(1, 1, -1)
        out = h + self.mlp[2](hidden)
        if not return_diagnostics:
            return out
        diagnostics = {
            "mlp_preactivation": preactivation,
            "gelu_derivative": gelu_derivative(preactivation),
            "attention_weights": attention_weights,
        }
        return out, diagnostics


class TinyTransformer(nn.Module):
    def __init__(self, config: ExperimentConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.max_seq_len = episode_length(config)
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(self.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.layers)])
        self.final_ln = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, vocab_size, bias=False)

    @staticmethod
    def _apply_residual_patch(
        x: torch.Tensor,
        residual_patch: dict | None,
        layer: int,
    ) -> torch.Tensor:
        if residual_patch is None or int(residual_patch.get("layer", -1)) != layer:
            return x
        if "position" not in residual_patch or "value" not in residual_patch:
            raise ValueError("residual patch requires layer, position, and value")
        position = int(residual_patch["position"])
        if position < 0:
            position += x.shape[1]
        if not 0 <= position < x.shape[1]:
            raise ValueError("residual patch position is out of range")
        value = torch.as_tensor(residual_patch["value"], device=x.device, dtype=x.dtype)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[1] != x.shape[2]:
            raise ValueError("residual patch value must have shape [batch, d_model] or [d_model]")
        if value.shape[0] == 1 and x.shape[0] != 1:
            value = value.expand(x.shape[0], -1)
        if value.shape[0] != x.shape[0]:
            raise ValueError("residual patch batch dimension does not match tokens")
        patched = x.clone()
        patched[:, position, :] = value
        return patched

    def forward(
        self,
        tokens: torch.Tensor,
        return_residuals: bool = False,
        return_diagnostics: bool = False,
        residual_patch: dict | None = None,
        mlp_masks: dict[int, torch.Tensor] | None = None,
    ):
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        batch, seq = tokens.shape
        if seq > self.max_seq_len:
            raise ValueError(f"sequence length {seq} exceeds max {self.max_seq_len}")
        if residual_patch is not None:
            patch_layer = int(residual_patch.get("layer", -1))
            if not 0 <= patch_layer <= self.config.layers:
                raise ValueError("residual patch layer is out of range")
        positions = torch.arange(seq, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        x = self._apply_residual_patch(x, residual_patch, 0)
        residuals = [x]
        diagnostics: list[dict[str, torch.Tensor]] = []
        causal = torch.triu(
            torch.ones((seq, seq), dtype=torch.bool, device=tokens.device), diagonal=1
        )
        for block_index, block in enumerate(self.blocks):
            mask = None if mlp_masks is None else mlp_masks.get(block_index)
            if return_diagnostics:
                x, diag = block(
                    x,
                    causal,
                    return_diagnostics=True,
                    mlp_mask=mask,
                )
                diagnostics.append(diag)
            else:
                x = block(x, causal, mlp_mask=mask)
            x = self._apply_residual_patch(x, residual_patch, block_index + 1)
            residuals.append(x)
        logits = self.output(self.final_ln(x))
        if return_residuals and return_diagnostics:
            return logits, residuals, diagnostics
        if return_residuals:
            return logits, residuals
        if return_diagnostics:
            return logits, diagnostics
        return logits
