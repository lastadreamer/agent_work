"""Pick the board / encoder for cfg['game'] (xiangqi default, or gomoku)."""

from __future__ import annotations

from typing import Any, Mapping


def game_name(cfg: Mapping[str, Any] | None = None) -> str:
    if cfg is None:
        return "xiangqi"
    return str(cfg.get("game") or "xiangqi").strip().lower()


def make_board(cfg: Mapping[str, Any]):
    if game_name(cfg) == "gomoku":
        from xiangqi_engine.gomoku.board import GomokuBoard

        return GomokuBoard(int(cfg["board"]["files"]))
    from xiangqi_engine._xiangqi import Board

    return Board()


def make_encoder(cfg):
    if game_name(cfg) == "gomoku":
        from xiangqi_engine.gomoku.encode import GomokuEncoder

        return GomokuEncoder(cfg)
    from xiangqi_engine.encode import Encoder

    return Encoder(cfg)
