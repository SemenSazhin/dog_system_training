"""Sequence classifiers for human cleanup-after-dog detection from pose features.

Architectures compared:
- LSTM / BiLSTM / GRU (recurrent baselines)
- MambaLite (selective SSM-style block in pure PyTorch)
- TransformerEncoder / TemporalTransformer (attention-based)
"""

from __future__ import annotations

import math
from typing import Dict, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, bidirectional: bool = False, num_classes: int = 2):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        out, _ = self.lstm(x)
        feat = out[:, -1, :]
        return self.head(feat)


class GRUClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, bidirectional: bool = False, num_classes: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class _MambaBlock(nn.Module):
    """Lightweight selective-state style block (pure PyTorch, no CUDA mamba-ssm)."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv = nn.Conv1d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + d_inner)
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.d_state = d_state
        self.d_inner = d_inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = self.conv(x_branch.transpose(1, 2)).transpose(1, 2)
        x_branch = F.silu(x_branch)

        x_dbl = self.x_proj(x_branch)
        delta, B, C = torch.split(
            x_dbl,
            [self.d_inner, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta))  # [B, T, d_inner]
        A = -torch.exp(self.A_log)  # [d_state]

        # Discretize and run a simplified selective scan over time
        y = []
        h = torch.zeros(x.size(0), self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        for t in range(x.size(1)):
            dt = delta[:, t].unsqueeze(-1)  # [B, d_inner, 1]
            # A_bar ~ exp(dt * A)
            a_bar = torch.exp(dt * A)  # broadcast [B, d_inner, d_state]
            b_bar = dt * B[:, t].unsqueeze(1)  # [B, 1, d_state] -> [B, d_inner, d_state]
            h = a_bar * h + b_bar * x_branch[:, t].unsqueeze(-1)
            yt = torch.einsum("bdn,bn->bd", h, C[:, t]) + self.D * x_branch[:, t]
            y.append(yt)
        y = torch.stack(y, dim=1)
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaLiteClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, num_classes: int = 2, d_state: int = 16):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.blocks = nn.ModuleList(
            [_MambaBlock(hidden_size, d_state=d_state) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = h + blk(self.norm(h))
        feat = h.mean(dim=1)
        return self.head(feat)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoderClassifier(nn.Module):
    """Standard Transformer encoder over pose feature frames + CLS token."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2,
        max_len: int = 512,
    ):
        super().__init__()
        if hidden_size % nhead != 0:
            hidden_size = nhead * max(1, hidden_size // nhead)
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        nn.init.normal_(self.cls_token, std=0.02)
        self.pos = PositionalEncoding(hidden_size, max_len=max_len + 1, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        h = self.input_proj(x)
        cls = self.cls_token.expand(bsz, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos(h)
        h = self.encoder(h)
        h = self.norm(h[:, 0])
        return self.head(h)


class TemporalTransformerClassifier(nn.Module):
    """Temporal Transformer with mean+max pooling (PoseFormer-lite style)."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        nhead: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2,
        max_len: int = 512,
    ):
        super().__init__()
        if hidden_size % nhead != 0:
            hidden_size = nhead * max(1, hidden_size // nhead)
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos = PositionalEncoding(hidden_size, max_len=max_len, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        pooled_dim = hidden_size * 2
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pooled_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        h = self.norm(h)
        mean_pool = h.mean(dim=1)
        max_pool, _ = h.max(dim=1)
        feat = torch.cat([mean_pool, max_pool], dim=-1)
        return self.head(feat)


ARCH_REGISTRY: Dict[str, Type[nn.Module]] = {
    "lstm": LSTMClassifier,
    "bilstm": lambda **kw: LSTMClassifier(bidirectional=True, **{k: v for k, v in kw.items() if k != "bidirectional"}),
    "gru": GRUClassifier,
    "bigru": lambda **kw: GRUClassifier(bidirectional=True, **{k: v for k, v in kw.items() if k != "bidirectional"}),
    "mamba_lite": MambaLiteClassifier,
    "transformer": TransformerEncoderClassifier,
    "temporal_transformer": TemporalTransformerClassifier,
}


def build_model(name: str, input_size: int, **kwargs) -> nn.Module:
    key = name.lower().strip()
    if key not in ARCH_REGISTRY:
        raise KeyError(f"Unknown architecture '{name}'. Available: {list(ARCH_REGISTRY)}")
    factory = ARCH_REGISTRY[key]
    # Filter kwargs not accepted by some constructors
    import inspect

    if callable(factory) and not isinstance(factory, type):
        # lambdas for bi* wrappers
        return factory(input_size=input_size, **kwargs)

    sig = inspect.signature(factory.__init__)
    accepted = {k: v for k, v in {"input_size": input_size, **kwargs}.items() if k in sig.parameters}
    return factory(**accepted)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
