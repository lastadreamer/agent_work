"""Board <-> tensor and Move <-> policy index, driven by config/default.json."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

from xiangqi_engine._xiangqi import (
    ACTION_FROM_TO,
    Board,
    EncodeSpec,
    Move,
    encode_state,
    index_to_move,
    legal_indices,
    move_to_index,
    n_input_planes,
    should_flip,
)
from xiangqi_engine.config import Cfg, load_config, spec_from_config


class Encoder:
    """Keeps a sliding window of boards so history planes stay aligned with the net."""

    def __init__(self, cfg: Cfg | None = None):
        self.cfg = cfg if cfg is not None else load_config()
        self.spec: EncodeSpec = spec_from_config(self.cfg)
        self._history: deque[Board] = deque(maxlen=max(self.spec.history_length - 1, 1))
        self._have_current = False
        self._current: Board | None = None

    @property
    def n_planes(self) -> int:
        return int(n_input_planes(self.spec))

    @property
    def action_size(self) -> int:
        return int(self.cfg["action"]["size"])

    def reset(self, board: Board | None = None) -> None:
        self._history.clear()
        self._have_current = False
        self._current = None
        if board is not None:
            self.observe(board)

    def observe(self, board: Board) -> None:
        if self._have_current and self._current is not None:
            self._history.append(self._current.copy())
        self._current = board.copy()
        self._have_current = True

    def past_for(self, board: Board) -> list[Board]:
        """History boards to pass to encode_state for ``board`` (oldest first)."""
        past = list(self._history)
        if self._have_current and self._current is not None and self._current.hash() != board.hash():
            past.append(self._current)
        return past

    def tensor(self, board: Board | None = None, *, observe: bool = False):
        """Return float32 array shaped (C, 10, 9).

        Prior ``observe`` calls are the past. If ``board`` is a new position
        (different hash from the last observe), that last observe is appended
        to the history window automatically.
        """
        if observe:
            if board is None:
                raise ValueError("observe=True requires a board")
            self.observe(board)
            board = None
        if board is None:
            if self._current is None:
                raise RuntimeError("Encoder.tensor() needs a board; call observe() or pass board=")
            return encode_state(self._current, self.spec, list(self._history))
        return encode_state(board, self.spec, self.past_for(board))

    def flip(self, board: Board) -> bool:
        return bool(should_flip(board, self.spec))

    def move_index(self, board: Board, move: Move) -> int:
        return int(move_to_index(move, self.flip(board)))

    def move_from_index(self, board: Board, index: int) -> Move:
        return index_to_move(int(index), self.flip(board))

    def legal_action_indices(self, board: Board) -> list[int]:
        return [int(i) for i in legal_indices(board, self.spec)]

    def policy_target(self, board: Board, visit_counts: Sequence[int] | None = None):
        """Dense length-8100 vector. Uniform over legal moves if counts are omitted."""
        indices = self.legal_action_indices(board)
        target = [0.0] * ACTION_FROM_TO
        if not indices:
            return target
        if visit_counts is None:
            p = 1.0 / len(indices)
            for idx in indices:
                target[idx] = p
            return target
        if len(visit_counts) != len(indices):
            raise ValueError("visit_counts must align with legal_action_indices")
        total = float(sum(visit_counts))
        if total <= 0:
            p = 1.0 / len(indices)
            for idx in indices:
                target[idx] = p
            return target
        for idx, n in zip(indices, visit_counts):
            target[idx] = float(n) / total
        return target


def encode_board(board: Board, cfg: Cfg | None = None, history: Iterable[Board] = ()):
    spec = spec_from_config(cfg if cfg is not None else load_config())
    return encode_state(board, spec, list(history))
