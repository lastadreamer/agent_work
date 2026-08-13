"""Replay buffer of (s, π, z) samples. π is stored sparse (legal slots only)."""

from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from xiangqi_engine.config import Cfg, load_config


@dataclass
class Sample:
    state: np.ndarray  # float32 (C, 10, 9)
    policy_index: np.ndarray  # int32
    policy_prob: np.ndarray  # float32, same length, sums to 1
    value: float  # z in [-1, 1] from the player to move at `state`


def sample_from_dense(state: np.ndarray, policy: list[float] | np.ndarray, value: float) -> Sample:
    pi = np.asarray(policy, dtype=np.float32)
    idx = np.flatnonzero(pi > 0).astype(np.int32)
    if idx.size == 0:
        raise ValueError("policy has no mass")
    return Sample(state.astype(np.float32, copy=False), idx, pi[idx], float(value))


class ReplayBuffer:
    def __init__(self, cfg: Cfg | None = None, capacity: int | None = None):
        self.cfg = cfg if cfg is not None else load_config()
        cap = int(self.cfg["replay"]["capacity"] if capacity is None else capacity)
        self.capacity = cap
        self.action_size = int(self.cfg["action"]["size"])
        self._items: deque[Sample] = deque(maxlen=cap)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, sample: Sample) -> None:
        self._items.append(sample)

    def extend(self, samples: list[Sample]) -> None:
        self._items.extend(samples)

    def ready(self) -> bool:
        return len(self._items) >= int(self.cfg["replay"]["min_size"])

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._items:
            raise RuntimeError("replay buffer is empty")
        rng = rng or np.random.default_rng()
        n = min(batch_size, len(self._items))
        picks = rng.choice(len(self._items), size=n, replace=False)
        states = np.stack([self._items[i].state for i in picks], axis=0)
        policies = np.zeros((n, self.action_size), dtype=np.float32)
        values = np.empty(n, dtype=np.float32)
        for row, i in enumerate(picks):
            s = self._items[i]
            policies[row, s.policy_index] = s.policy_prob
            values[row] = s.value
        return states, policies, values

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(list(self._items), fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        with Path(path).open("rb") as fh:
            items = pickle.load(fh)
        self._items = deque(items, maxlen=self.capacity)
