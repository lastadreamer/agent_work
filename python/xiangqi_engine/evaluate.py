"""Play two evaluators against each other. No Dirichlet; temperature 0."""

from __future__ import annotations

from dataclasses import dataclass

from xiangqi_engine._xiangqi import RED, Outcome
from xiangqi_engine.config import Cfg, load_config
from xiangqi_engine.game import make_board, make_encoder
from xiangqi_engine.mcts import MCTS, terminal_value
from xiangqi_engine.progress import Progress
from xiangqi_engine.selfplay import outcome_to_red_z


@dataclass
class MatchResult:
    wins: int
    losses: int
    draws: int
    n_games: int

    @property
    def win_rate(self) -> float:
        # Score from the challenger's view: win=1, draw=0.5. Used vs threshold.
        if self.n_games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.n_games


def play_eval_game(cfg: Cfg, ev_first, ev_second, simulations: int, seed: int) -> float:
    """Return z from the first player's view (red in Xiangqi, black in Gomoku)."""
    board = make_board(cfg)
    enc = make_encoder(cfg)
    enc.reset(board)
    first_side = board.side_to_move()
    first = MCTS(cfg, ev_first, encoder=enc, seed=seed)
    second = MCTS(cfg, ev_second, encoder=enc, seed=seed + 1)
    reuse_tree = bool(cfg["mcts"].get("reuse_tree", True))
    max_plies = int(cfg["selfplay"]["max_plies"])
    for _ in range(max_plies):
        if terminal_value(board) is not None:
            break
        mcts = first if board.side_to_move() == first_side else second
        result = mcts.run(
            board,
            simulations=simulations,
            add_noise=False,
            temperature=0.0,
            reuse=reuse_tree,
        )
        if result.move is None:
            break
        board.push(result.move)
        enc.observe(board)
        if reuse_tree:
            if not first.advance(result.action_index):
                first.reset()
            if not second.advance(result.action_index):
                second.reset()
        else:
            first.reset()
            second.reset()
    term = board.terminal()
    outcome = term.outcome if term.outcome != Outcome.ONGOING else Outcome.DRAW
    z_red = outcome_to_red_z(outcome)
    return z_red if first_side == RED else -z_red


def play_match(
    cfg: Cfg | None,
    challenger,
    incumbent,
    n_games: int | None = None,
    simulations: int | None = None,
    seed: int = 0,
) -> MatchResult:
    """`challenger` and `incumbent` are Evaluator objects. Challenger moves first on even games."""
    cfg = cfg if cfg is not None else load_config()
    n_games = int(cfg["eval"]["n_games"] if n_games is None else n_games)
    simulations = int(cfg["eval"]["mcts_simulations"] if simulations is None else simulations)
    wins = losses = draws = 0
    progress = Progress("eval", n_games)
    for i in range(n_games):
        if i % 2 == 0:
            z_red = play_eval_game(cfg, challenger, incumbent, simulations, seed + i)
            z_ch = z_red
        else:
            z_red = play_eval_game(cfg, incumbent, challenger, simulations, seed + i)
            z_ch = -z_red
        if z_ch > 0:
            wins += 1
        elif z_ch < 0:
            losses += 1
        else:
            draws += 1
        progress.update(i + 1, extra=f"W{wins} L{losses} D{draws}")
    return MatchResult(wins, losses, draws, n_games)
