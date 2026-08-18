"""Gomoku play session. Same MCTS/checkpoint path as Xiangqi, different board."""

from __future__ import annotations

from xiangqi_engine._xiangqi import BLACK, RED, Outcome
from xiangqi_engine.config import Cfg, deepcopy_config, load_config
from xiangqi_engine.game import make_board, make_encoder
from xiangqi_engine.gomoku.board import GomokuMove, sq_to_iccs
from xiangqi_engine.mcts import MCTS, NetworkEvaluator, UniformEvaluator, terminal_value
from xiangqi_engine.play.session import resolved_checkpoint


OUTCOME_TEXT = {
    Outcome.ONGOING: "",
    Outcome.RED_WIN: "白胜",
    Outcome.BLACK_WIN: "黑胜",
    Outcome.DRAW: "和棋",
}

REASON_TEXT = {"FIVE": "五连", "FULL": "满盘", "": ""}


def _role(name: str) -> str:
    name = (name or "human").strip().lower()
    if name in ("ai", "engine", "machine", "bot"):
        return "ai"
    return "human"


class GomokuPlaySession:
    def __init__(self, cfg: Cfg | None = None):
        self.cfg = deepcopy_config(cfg if cfg is not None else load_config())
        self.red = "ai"
        self.black = "human"
        self.simulations = int(self.cfg.get("play", {}).get("simulations") or self.cfg["mcts"]["simulations"])
        self.checkpoint = str(self.cfg.get("play", {}).get("checkpoint") or "")
        self.history: list[str] = []
        self.board = make_board(self.cfg)
        self._net = None
        self._net_path = None
        self._error = ""
        self._encoder = make_encoder(self.cfg)
        self._search: MCTS | None = None

    def new_game(
        self,
        red: str = "ai",
        black: str = "human",
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
        self.board = make_board(self.cfg)
        self._error = ""
        self._load_net()
        self._bind_search()
        return self.state()

    def _load_net(self) -> None:
        path = resolved_checkpoint(self.checkpoint)
        self.checkpoint = path
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

    def _evaluator(self, encoder):
        if self._net is None:
            return UniformEvaluator(encoder)
        return NetworkEvaluator(self._net, encoder, device="cpu")

    def _bind_search(self) -> None:
        self._encoder = make_encoder(self.cfg)
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

    def _advance_tree(self, move: GomokuMove) -> None:
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
        self.board = make_board(self.cfg)
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
            mv = GomokuMove.from_iccs(iccs, self.board.size)
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
        size = self.board.size
        squares = []
        for rank in range(size - 1, -1, -1):
            row = []
            for file in range(size):
                sq = rank * size + file
                stone = int(self.board.piece_at(sq))
                row.append(
                    {
                        "sq": sq,
                        "file": file,
                        "rank": rank,
                        "iccs": sq_to_iccs(sq, size),
                        "stone": stone,
                        "color": None if stone == 0 else ("black" if stone == 1 else "white"),
                    }
                )
            squares.append(row)
        legal = []
        if not self._terminal():
            legal = [mv.iccs() for mv in self.board.legal_moves()]
        term = self.board.terminal()
        winning = [sq_to_iccs(sq, size) for sq in self.board.winning_squares()]
        last = None
        if self.history:
            last = {"iccs": self.history[-1]}
        return {
            "game": "gomoku",
            "size": size,
            "squares": squares,
            "fen": self.board.fen(),
            "side": "black" if self.board.side_to_move() == BLACK else "white",
            "outcome": OUTCOME_TEXT.get(term.outcome, ""),
            "reason": REASON_TEXT.get(term.reason, "") if term.outcome != Outcome.ONGOING else "",
            "over": term.outcome != Outcome.ONGOING,
            "history": list(self.history),
            "last_move": last,
            "legal": legal,
            "winning": winning,
            "red": self.red,
            "black": self.black,
            "to_move_role": self._side_role(),
            "simulations": self.simulations,
            "checkpoint": self.checkpoint,
            "error": self._error,
            "can_undo": bool(self.history),
            "ply": int(self.board.ply()),
            "play_defaults": {
                "checkpoint": self.checkpoint,
                "simulations": self.simulations,
            },
        }
