"""Self-play: one game produces (s, π, z) from the current player's view."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from xiangqi_engine._xiangqi import BLACK, RED, Board, Outcome
from xiangqi_engine.config import Cfg, load_config
from xiangqi_engine.encode import Encoder
from xiangqi_engine.mcts import MCTS, UniformEvaluator, terminal_value
from xiangqi_engine.replay import Sample, sample_from_dense


@dataclass
class GameRecord:
    samples: list[Sample]
    outcome: Outcome
    plies: int
    red_z: float


def outcome_to_red_z(outcome: Outcome) -> float:
    if outcome == Outcome.RED_WIN:
        return 1.0
    if outcome == Outcome.BLACK_WIN:
        return -1.0
    return 0.0


def play_game(
    cfg: Cfg | None = None,
    evaluator=None,
    encoder: Encoder | None = None,
    seed: int = 0,
    simulations: int | None = None,
) -> GameRecord:
    cfg = cfg if cfg is not None else load_config()
    enc = encoder if encoder is not None else Encoder(cfg)
    if evaluator is None:
        evaluator = UniformEvaluator(enc)
    board = Board()
    enc.reset(board)
    mcts = MCTS(cfg, evaluator, encoder=enc, seed=seed)
    max_plies = int(cfg["selfplay"]["max_plies"])
    temp_moves = int(cfg["mcts"]["temperature_moves"])
    tau_open = float(cfg["mcts"]["temperature"])
    reuse_tree = bool(cfg["mcts"].get("reuse_tree", True))
    pending: list[tuple[np.ndarray, list[float], int]] = []

    for ply in range(max_plies):
        z_now = terminal_value(board)
        if z_now is not None:
            break
        tau = tau_open if ply < temp_moves else 0.0
        state = enc.tensor(board)
        side = int(board.side_to_move())
        result = mcts.run(
            board,
            simulations=simulations,
            add_noise=True,
            temperature=tau,
            reuse=reuse_tree,
        )
        if result.move is None:
            break
        pending.append((state, result.policy, side))
        board.push(result.move)
        enc.observe(board)
        if reuse_tree:
            mcts.advance(result.action_index)
        else:
            mcts.reset()

    term = board.terminal()
    outcome = term.outcome if term.outcome != Outcome.ONGOING else Outcome.DRAW
    red_z = outcome_to_red_z(outcome)
    samples = []
    for state, policy, side in pending:
        z = red_z if side == RED else -red_z
        samples.append(sample_from_dense(state, policy, z))
    return GameRecord(samples, outcome, len(pending), red_z)


def play_games(
    cfg: Cfg | None = None,
    evaluator=None,
    n_games: int | None = None,
    n_workers: int | None = None,
    seed: int = 0,
    simulations: int | None = None,
    state_dict=None,
) -> list[GameRecord]:
    """Run games. n_workers>1 uses processes and needs a CPU state_dict (or uniform)."""
    cfg = cfg if cfg is not None else load_config()
    n_games = int(cfg["selfplay"]["n_games_per_iter"] if n_games is None else n_games)
    n_workers = int(cfg["selfplay"]["n_workers"] if n_workers is None else n_workers)
    seeds = [seed + i for i in range(n_games)]
    if n_workers <= 1:
        enc = Encoder(cfg)
        ev = evaluator if evaluator is not None else UniformEvaluator(enc)
        return [play_game(cfg, ev, Encoder(cfg), s, simulations) for s in seeds]

    cfg_dict = dict(cfg)
    # spawn: parent may already have imported torch; fork+threads is unsafe.
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(cfg_dict, state_dict, simulations, counter),
    ) as pool:
        return list(pool.map(_play_worker, seeds))


_WORKER: dict = {}


def selfplay_worker_device(cfg: Cfg | dict[str, Any], worker_id: int = 0) -> str:
    """Map a worker onto cpu / cuda:N. Falls back to cpu if CUDA is missing."""
    name = str((cfg.get("selfplay") or {}).get("device") or "cpu").strip().lower()
    if name in ("", "cpu"):
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    visible = max(int(torch.cuda.device_count()), 1)
    wanted = int((cfg.get("selfplay") or {}).get("gpus") or visible)
    n = max(1, min(wanted, visible))
    if name in ("auto", "cuda"):
        return f"cuda:{int(worker_id) % n}"
    return name


def _init_worker(cfg_dict: dict, state_dict, simulations, counter) -> None:
    from xiangqi_engine.config import Cfg
    from xiangqi_engine.encode import Encoder

    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass

    cfg = Cfg(cfg_dict)
    with counter.get_lock():
        worker_id = int(counter.value)
        counter.value += 1
    _WORKER["cfg"] = cfg
    _WORKER["simulations"] = simulations
    _WORKER["state_dict"] = state_dict
    _WORKER["encoder_factory"] = lambda: Encoder(cfg)
    _WORKER["device"] = selfplay_worker_device(cfg, worker_id)


def _play_worker(seed: int) -> GameRecord:
    cfg = _WORKER["cfg"]
    enc = _WORKER["encoder_factory"]()
    state_dict = _WORKER["state_dict"]
    if state_dict is None:
        ev = UniformEvaluator(enc)
    else:
        from xiangqi_engine.mcts import NetworkEvaluator
        from xiangqi_engine.network import PolicyValueNet

        net = PolicyValueNet(cfg)
        net.load_state_dict(state_dict)
        net.to(_WORKER["device"])
        net.eval()
        ev = NetworkEvaluator(net, enc, device=_WORKER["device"])
    return play_game(cfg, ev, enc, seed, _WORKER["simulations"])
