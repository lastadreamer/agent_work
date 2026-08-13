"""AlphaZero dual-head residual network. All sizes come from the JSON config."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from xiangqi_engine.config import Cfg, load_config, n_input_planes_from_config, resolve_device
from xiangqi_engine.encode import Encoder


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    """fθ(s) → (policy logits over 8100 from-to actions, value in [-1, 1])."""

    def __init__(self, cfg: Cfg | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else load_config()
        in_ch = n_input_planes_from_config(self.cfg)
        net = self.cfg["network"]
        channels = int(net["channels"])
        blocks = int(net["blocks"])
        policy_ch = int(net["policy_head_channels"])
        value_ch = int(net["value_head_channels"])
        value_hidden = int(net["value_hidden"])
        action_size = int(self.cfg["action"]["size"])
        spatial = int(self.cfg["board"]["ranks"]) * int(self.cfg["board"]["files"])

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, policy_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(policy_ch),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(policy_ch * spatial, action_size)

        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, value_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(value_ch),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(value_ch * spatial, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
        )

        self.in_channels = in_ch
        self.action_size = action_size

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, 10, 9) → policy logits (B, A), value (B,)."""
        h = self.tower(self.stem(x))
        p = self.policy_fc(self.policy_conv(h).flatten(1))
        v = torch.tanh(self.value_fc(self.value_conv(h).flatten(1))).squeeze(-1)
        return p, v

    def masked_policy(self, logits: torch.Tensor, legal_indices: Iterable[int]) -> torch.Tensor:
        """Softmax over legal from-to indices only. Returns a dense (A,) tensor."""
        idx = torch.as_tensor(list(legal_indices), device=logits.device, dtype=torch.long)
        if idx.numel() == 0:
            return torch.zeros(self.action_size, device=logits.device)
        legal_logits = logits.index_select(0, idx)
        probs = torch.zeros(self.action_size, device=logits.device, dtype=logits.dtype)
        probs[idx] = torch.softmax(legal_logits, dim=0)
        return probs


def build_network(cfg: Cfg | None = None, device: str | None = None) -> PolicyValueNet:
    cfg = cfg if cfg is not None else load_config()
    net = PolicyValueNet(cfg)
    net.to(resolve_device(cfg, device))
    return net


@torch.no_grad()
def infer(net: PolicyValueNet, encoder: Encoder, board, device: str | None = None):
    """Convenience: encode one board, return (legal_indices, legal_probs, value)."""
    dev = device or next(net.parameters()).device
    x = torch.from_numpy(encoder.tensor(board)).unsqueeze(0).to(dev)
    net.eval()
    logits, value = net(x)
    legal = encoder.legal_action_indices(board)
    probs = net.masked_policy(logits[0], legal)
    legal_probs = [float(probs[i]) for i in legal]
    return legal, legal_probs, float(value[0])
