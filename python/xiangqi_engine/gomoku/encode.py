"""Gomoku CHW encoding: our/opp stone planes, current-player perspective."""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence

import numpy as np

from xiangqi_engine.config import Cfg, load_config
from xiangqi_engine.gomoku.board import EMPTY, GomokuBoard, GomokuMove, stone_of


def n_gomoku_planes(cfg: Cfg) -> int:
    encode = cfg["encode"]
    planes = encode["planes"]
    frame = 0
    if planes.get("our_pieces", True):
        frame += 1
    if planes.get("opp_pieces", True):
        frame += 1
    extra = 0
    if planes.get("side_to_move"):
        extra += 1
    if planes.get("ones"):
        extra += 1
    t = max(int(encode.get("history_length", 1)), 1)
    return t * frame + extra


def encode_gomoku(board: GomokuBoard, cfg: Cfg, history: Sequence[GomokuBoard] = ()) -> np.ndarray:
    size = board.size
    encode = cfg["encode"]
    planes = encode["planes"]
    t = max(int(encode.get("history_length", 1)), 1)
    our = bool(planes.get("our_pieces", True))
    opp = bool(planes.get("opp_pieces", True))
    frame_c = int(our) + int(opp)
    extra = []
    if planes.get("side_to_move"):
        extra.append(1.0 if board.side_to_move() == 0 else 0.0)
    if planes.get("ones"):
        extra.append(1.0)
    total_c = t * frame_c + len(extra)
    out = np.zeros((total_c, size, size), dtype=np.float32)
    us = board.side_to_move()
    them_stone = stone_of(us ^ 1)
    our_stone = stone_of(us)

    past = list(history)
    use = past[-(t - 1) :] if t > 1 else []
    frames: list[GomokuBoard] = [None] * (t - 1 - len(use)) + use + [board]  # type: ignore[list-item]
    slot = 0
    for frame in frames:
        if frame is None or frame_c == 0:
            slot += frame_c
            continue
        for sq, p in enumerate(frame.squares):
            if p == EMPTY:
                continue
            r, c = divmod(sq, size)
            ch = 0
            if our and p == our_stone:
                out[slot + ch, r, c] = 1.0
            if opp:
                ch = int(our)
                if p == them_stone:
                    out[slot + ch, r, c] = 1.0
        slot += frame_c
    for i, value in enumerate(extra):
        out[t * frame_c + i].fill(value)
    return out


class GomokuEncoder:
    def __init__(self, cfg: Cfg | None = None):
        self.cfg = cfg if cfg is not None else load_config()
        self.size = int(self.cfg["board"]["files"])
        self._history: deque[GomokuBoard] = deque(maxlen=max(int(self.cfg["encode"]["history_length"]) - 1, 1))
        self._have_current = False
        self._current: GomokuBoard | None = None

    @property
    def n_planes(self) -> int:
        return n_gomoku_planes(self.cfg)

    @property
    def action_size(self) -> int:
        return int(self.cfg["action"]["size"])

    def reset(self, board: GomokuBoard | None = None) -> None:
        self._history.clear()
        self._have_current = False
        self._current = None
        if board is not None:
            self.observe(board)

    def observe(self, board: GomokuBoard) -> None:
        if self._have_current and self._current is not None:
            self._history.append(self._current.copy())
        self._current = board.copy()
        self._have_current = True

    def past_for(self, board: GomokuBoard) -> list[GomokuBoard]:
        past = list(self._history)
        if self._have_current and self._current is not None and self._current.hash() != board.hash():
            past.append(self._current)
        return past

    def encode(self, board: GomokuBoard, history: Iterable[GomokuBoard] | None = None) -> np.ndarray:
        hist = list(self.past_for(board) if history is None else history)
        return encode_gomoku(board, self.cfg, hist)

    def tensor(self, board: GomokuBoard | None = None, *, observe: bool = False) -> np.ndarray:
        if observe:
            if board is None:
                raise ValueError("observe=True requires a board")
            self.observe(board)
            board = None
        if board is None:
            if self._current is None:
                raise RuntimeError("GomokuEncoder.tensor() needs a board")
            return encode_gomoku(self._current, self.cfg, list(self._history))
        return self.encode(board)

    def legal_action_indices(self, board: GomokuBoard) -> list[int]:
        return [mv.sq for mv in board.legal_moves()]

    def move_from_index(self, board: GomokuBoard, index: int) -> GomokuMove:
        del board
        return GomokuMove(int(index), self.size)

    def move_index(self, board: GomokuBoard, move: GomokuMove) -> int:
        del board
        return int(move.sq)

    def play(self, board: GomokuBoard, index: int) -> None:
        board.make_move(self.move_from_index(board, index))
