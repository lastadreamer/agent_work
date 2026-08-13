"""Interactive game session used by the play UI. Does not change training or MCTS."""

from __future__ import annotations

from xiangqi_engine._xiangqi import RED, Board, Move, Outcome, TerminalReason, square_to_iccs
from xiangqi_engine.config import Cfg, deepcopy_config, load_config
from xiangqi_engine.encode import Encoder
from xiangqi_engine.mcts import MCTS, UniformEvaluator, terminal_value

GLYPHS = {
    0: "",
    1: "帅",
    2: "仕",
    3: "相",
    4: "马",
    5: "车",
    6: "炮",
    7: "兵",
    8: "将",
    9: "士",
    10: "象",
    11: "馬",
    12: "車",
    13: "砲",
    14: "卒",
}

OUTCOME_TEXT = {
    Outcome.ONGOING: "",
    Outcome.RED_WIN: "红胜",
    Outcome.BLACK_WIN: "黑胜",
    Outcome.DRAW: "和棋",
}

REASON_TEXT = {
    TerminalReason.NONE: "",
    TerminalReason.CHECKMATE: "将死",
    TerminalReason.STALEMATE: "困毙",
    TerminalReason.REPETITION: "允许不变",
    TerminalReason.NO_PROGRESS: "六十回合",
    TerminalReason.MAX_PLY: "超限",
    TerminalReason.PERPETUAL_CHECK: "长将",
    TerminalReason.PERPETUAL_CHASE: "长捉",
}


def _role(name: str) -> str:
    name = (name or "human").strip().lower()
    if name in ("ai", "engine", "machine", "bot"):
        return "ai"
    return "human"


class PlaySession:
    def __init__(self, cfg: Cfg | None = None):
        self.cfg = deepcopy_config(cfg if cfg is not None else load_config())
        self.red = "human"
        self.black = "human"
        self.simulations = int(self.cfg.get("play", {}).get("simulations") or self.cfg["mcts"]["simulations"])
        self.checkpoint = str(self.cfg.get("play", {}).get("checkpoint") or "")
        self.history: list[str] = []
        self.board = Board()
        self._net = None
        self._net_path = None
        self._error = ""
        self._encoder = Encoder(self.cfg)
        self._search: MCTS | None = None

    def new_game(
        self,
        red: str = "human",
        black: str = "ai",
        simulations: int | None = None,
        checkpoint: str | None = None,
    ) -> dict:
        self.red = _role(red)
        self.black = _role(black)
        if simulations is not None:
            self.simulations = int(simulations)
        if checkpoint is not None:
            self.checkpoint = str(checkpoint)
        self.history = []
        self.board = Board()
        self._error = ""
        self._load_net()
        self._bind_search()
        return self.state()

    def _load_net(self) -> None:
        path = self.checkpoint.strip()
        if not path:
            self._net = None
            self._net_path = None
            return
        if self._net is not None and self._net_path == path:
            return
        try:
            from xiangqi_engine.loop import load_checkpoint
            from xiangqi_engine.network import PolicyValueNet

            net = PolicyValueNet(self.cfg)
            load_checkpoint(path, net)
            net.eval()
            self._net = net
            self._net_path = path
        except Exception as exc:
            self._net = None
            self._net_path = None
            self._error = f"加载权重失败：{exc}"

    def _evaluator(self, encoder: Encoder):
        if self._net is None:
            return UniformEvaluator(encoder)
        from xiangqi_engine.mcts import NetworkEvaluator

        return NetworkEvaluator(self._net, encoder, device="cpu")

    def _bind_search(self) -> None:
        """One MCTS tree for the whole game: descend on a move, retreat on undo."""
        self._encoder = Encoder(self.cfg)
        self._encoder.reset(self.board)
        ev = self._evaluator(self._encoder)
        self._search = MCTS(self.cfg, ev, encoder=self._encoder, seed=1)
        self._search.ensure_expanded(self.board)
        if self.red == "ai" or self.black == "ai":
            self._search.run(
                self.board,
                simulations=self.simulations,
                add_noise=False,
                temperature=0.0,
                reuse=True,
            )

    def _advance_tree(self, move: Move) -> None:
        if self._search is None:
            return
        if not self._search.root.expanded:
            self._search.ensure_expanded(self.board)
        idx = self._encoder.move_index(self.board, move)
        if not self._search.advance(idx):
            self._search.reset()

    def _retreat_tree(self, plies: int) -> None:
        if self._search is None or plies <= 0:
            return
        self._search.retreat(plies)

    def _side_role(self, side: int | None = None) -> str:
        if side is None:
            side = self.board.side_to_move()
        return self.red if side == RED else self.black

    def _rebuild(self) -> None:
        self.board = Board()
        self._encoder.reset(self.board)
        for iccs in self.history:
            self.board.push_iccs(iccs)
            self._encoder.observe(self.board)

    def move(self, iccs: str) -> dict:
        self._error = ""
        if self._terminal():
            self._error = "对局已经结束"
            return self.state()
        try:
            mv = Move.from_iccs(iccs)
        except Exception:
            self._error = f"着法格式不对：{iccs}"
            return self.state()
        if not self.board.is_legal(mv):
            self._error = f"非法着法：{iccs}"
            return self.state()
        self._advance_tree(mv)
        self.board.push(mv)
        self.history.append(mv.iccs())
        self._encoder.observe(self.board)
        return self.state()

    def undo(self, plies: int = 1) -> dict:
        self._error = ""
        n = max(0, int(plies))
        if n == 0 or not self.history:
            return self.state()
        n = min(n, len(self.history))
        del self.history[-n:]
        self._rebuild()
        self._retreat_tree(n)
        return self.state()

    def undo_human_turn(self) -> dict:
        """Take back until a human is to move (or the start)."""
        self._error = ""
        if not self.history:
            return self.state()
        popped = 1
        self.history.pop()
        self._rebuild()
        if self.history and self._side_role() == "ai" and not self._terminal():
            self.history.pop()
            popped += 1
            self._rebuild()
        self._retreat_tree(popped)
        return self.state()

    def ai_move(self) -> dict:
        self._error = ""
        if self._terminal():
            self._error = "对局已经结束"
            return self.state()
        if self._side_role() != "ai":
            self._error = "当前不是机器走棋"
            return self.state()
        if self._search is None:
            self._bind_search()
        result = self._search.run(
            self.board,
            simulations=self.simulations,
            add_noise=False,
            temperature=0.0,
            reuse=True,
        )
        if result.move is None:
            self._error = "机器没有合法着"
            return self.state()
        self._advance_tree(result.move)
        self.board.push(result.move)
        self.history.append(result.move.iccs())
        self._encoder.observe(self.board)
        return self.state()

    def _terminal(self) -> bool:
        return terminal_value(self.board) is not None

    def state(self) -> dict:
        squares = []
        for rank in range(9, -1, -1):
            row = []
            for file in range(9):
                sq = rank * 9 + file
                piece = int(self.board.piece_at(sq))
                row.append(
                    {
                        "sq": sq,
                        "file": file,
                        "rank": rank,
                        "iccs": square_to_iccs(sq),
                        "piece": piece,
                        "glyph": GLYPHS.get(piece, ""),
                        "color": None if piece == 0 else ("red" if piece < 8 else "black"),
                    }
                )
            squares.append(row)

        legal_from: dict[str, list[str]] = {}
        if not self._terminal():
            for mv in self.board.legal_moves():
                legal_from.setdefault(square_to_iccs(mv.from_sq), []).append(square_to_iccs(mv.to_sq))

        term = self.board.terminal()
        last = None
        if self.history:
            iccs = self.history[-1]
            last = {"from": iccs[:2], "to": iccs[2:4], "iccs": iccs}
        play = self.cfg.get("play", {})
        return {
            "squares": squares,
            "fen": self.board.fen(),
            "side": "red" if self.board.side_to_move() == RED else "black",
            "in_check": bool(self.board.in_check()),
            "outcome": OUTCOME_TEXT.get(term.outcome, ""),
            "reason": REASON_TEXT.get(term.reason, "") if term.outcome != Outcome.ONGOING else "",
            "over": term.outcome != Outcome.ONGOING,
            "history": list(self.history),
            "last_move": last,
            "legal_from": legal_from,
            "red": self.red,
            "black": self.black,
            "to_move_role": self._side_role(),
            "simulations": self.simulations,
            "checkpoint": self.checkpoint,
            "error": self._error,
            "can_undo": bool(self.history),
            "ply": int(self.board.ply()),
            "halfmove": int(self.board.halfmove_clock()),
            "play_defaults": {
                "checkpoint": play.get("checkpoint", ""),
                "simulations": int(play.get("simulations") or self.simulations),
            },
        }
