import torch
from torch import nn

from .config import ExperimentConfig
from .episodes import episode_length


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

    def forward(self, x: torch.Tensor, causal: torch.Tensor) -> torch.Tensor:
        n = self.ln1(x)
        a = self.attn(n, n, n, attn_mask=causal, need_weights=False)[0]
        h = x + a
        return h + self.mlp(self.ln2(h))


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

    def forward(self, tokens: torch.Tensor, return_residuals: bool = False):
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        batch, seq = tokens.shape
        if seq > self.max_seq_len:
            raise ValueError(f"sequence length {seq} exceeds max {self.max_seq_len}")
        positions = torch.arange(seq, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        residuals = [x]
        causal = torch.triu(
            torch.ones((seq, seq), dtype=torch.bool, device=tokens.device), diagonal=1
        )
        for block in self.blocks:
            x = block(x, causal)
            residuals.append(x)
        logits = self.output(self.final_ln(x))
        if return_residuals:
            return logits, residuals
        return logits
