"""One training step: L = w_p * CE(π, p) + w_v * (z - v)^2 (+ weight decay in the optimizer)."""

from __future__ import annotations

from typing import Any

import numpy as np

from xiangqi_engine.config import Cfg, load_config, resolve_device
from xiangqi_engine.replay import ReplayBuffer


def build_optimizer(net, cfg: Cfg | None = None):
    import torch

    cfg = cfg if cfg is not None else load_config()
    t = cfg["train"]
    lr = float(t["lr"])
    wd = float(t["weight_decay"])
    name = str(t.get("optimizer", "adam")).lower()
    if name == "sgd":
        return torch.optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=float(t.get("momentum", 0.9)),
            weight_decay=wd,
        )
    return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)


def batch_loss(net, states, policies, values, cfg: Cfg) -> tuple[Any, dict[str, float]]:
    import torch.nn.functional as F

    t = cfg["train"]
    logits, v = net(states)
    log_p = F.log_softmax(logits, dim=-1)
    policy_loss = -(policies * log_p).sum(dim=-1).mean()
    value_loss = F.mse_loss(v, values)
    loss = float(t["policy_loss_weight"]) * policy_loss + float(t["value_loss_weight"]) * value_loss
    return loss, {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
    }


def train_batches(
    net,
    buffer: ReplayBuffer,
    cfg: Cfg | None = None,
    n_batches: int | None = None,
    device: str | None = None,
    optimizer=None,
) -> dict[str, float]:
    import torch

    cfg = cfg if cfg is not None else load_config()
    device = resolve_device(cfg, device)
    net.to(device)
    net.train()
    opt = optimizer or build_optimizer(net, cfg)
    t = cfg["train"]
    n_batches = int(t["batches_per_iter"] if n_batches is None else n_batches)
    batch_size = int(t["batch_size"])
    clip = float(t.get("grad_clip", 0.0))
    rng = np.random.default_rng(int(cfg["seed"]))
    acc = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
    for _ in range(n_batches):
        states, policies, values = buffer.sample(batch_size, rng)
        st = torch.from_numpy(states).to(device)
        pi = torch.from_numpy(policies).to(device)
        z = torch.from_numpy(values).to(device)
        opt.zero_grad(set_to_none=True)
        loss, parts = batch_loss(net, st, pi, z, cfg)
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        opt.step()
        for k, val in parts.items():
            acc[k] += val
    n = max(n_batches, 1)
    return {k: v / n for k, v in acc.items()}
