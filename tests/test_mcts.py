import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from xiangqi_engine import ACTION_FROM_TO, Board, Encoder, Move, load_config
from xiangqi_engine.config import deepcopy_config
from xiangqi_engine.mcts import MCTS, SearchResult, UniformEvaluator, terminal_value
from xiangqi_engine._xiangqi import Outcome


MATE_IN_ONE = "4k4/R8/4H4/9/9/9/9/9/9/4K3R w - - 0 1"
MATE_MOVE = "i0i9"


def _cfg():
    return deepcopy_config(load_config())


def _mcts(cfg=None, encoder=None, seed=0) -> MCTS:
    cfg = cfg if cfg is not None else _cfg()
    enc = encoder if encoder is not None else Encoder(cfg)
    return MCTS(cfg, UniformEvaluator(enc), encoder=enc, seed=seed)


def test_mate_in_one_position_is_really_mate():
    b = Board(MATE_IN_ONE)
    assert b.terminal().outcome == Outcome.ONGOING
    b.push_iccs(MATE_MOVE)
    term = b.terminal()
    assert b.in_check()
    assert b.legal_moves() == []
    assert term.outcome == Outcome.RED_WIN
    assert terminal_value(b) == -1.0


def test_search_restores_board():
    b = Board()
    fen = b.fen()
    ply = b.ply()
    result = _mcts().run(b, simulations=32, add_noise=False, temperature=0)
    assert b.fen() == fen
    assert b.ply() == ply
    assert result.move is not None
    assert b.is_legal(result.move)


def test_visit_counts_sum_to_simulations():
    b = Board()
    result = _mcts().run(b, simulations=40, add_noise=False, temperature=1.0)
    assert sum(result.visit_counts.values()) == 40
    assert abs(sum(result.policy) - 1.0) < 1e-6
    assert all(p == 0.0 or i in result.visit_counts for i, p in enumerate(result.policy))
    assert len(result.policy) == ACTION_FROM_TO


def test_visits_only_on_legal_actions():
    b = Board()
    enc = Encoder()
    legal = set(enc.legal_action_indices(b))
    result = _mcts(encoder=enc).run(b, simulations=20, add_noise=False, temperature=1.0)
    assert set(result.visit_counts) <= legal
    assert result.action_index in legal


def test_mate_in_one_collects_visits():
    b = Board(MATE_IN_ONE)
    enc = Encoder()
    result = _mcts(encoder=enc).run(b, simulations=64, add_noise=False, temperature=0)
    # Several rank-8 rook slides are immediate 困毙; i0i9 is checkmate. Any
    # immediate win backs up as Q=+1, so PUCT stays on the first one it proves.
    after = b.copy()
    after.push(result.move)
    assert after.legal_moves() == []
    assert after.terminal().outcome == Outcome.RED_WIN
    assert result.root_value == 1.0
    assert result.visit_counts[result.action_index] == 64


def test_dirichlet_changes_root_priors():
    cfg = _cfg()
    cfg["mcts"]["dirichlet_epsilon"] = 1.0
    cfg["mcts"]["dirichlet_alpha"] = 0.3
    b = Board()
    enc = Encoder(cfg)
    ev = UniformEvaluator(enc)
    noisy = MCTS(cfg, ev, encoder=enc, seed=1)
    quiet = MCTS(cfg, UniformEvaluator(Encoder(cfg)), encoder=Encoder(cfg), seed=1)
    noisy.run(b, simulations=1, add_noise=True, temperature=0)
    quiet.run(b, simulations=1, add_noise=False, temperature=0)
    assert noisy.root.prior is not None and quiet.root.prior is not None
    assert not np.allclose(noisy.root.prior, quiet.root.prior)
    assert abs(float(noisy.root.prior.sum()) - 1.0) < 1e-6


def test_temperature_zero_is_greedy():
    b = Board()
    result = _mcts(seed=3).run(b, simulations=24, add_noise=False, temperature=0)
    top = max(result.visit_counts, key=result.visit_counts.get)
    assert result.action_index == top
    assert result.policy[top] == max(result.policy)


def test_history_length_two_search():
    cfg = _cfg()
    cfg["encode"]["history_length"] = 2
    enc = Encoder(cfg)
    start = Board()
    enc.observe(start)
    moved = start.copy()
    moved.push(Move.from_iccs("b2e2"))
    result = _mcts(cfg, encoder=enc).run(moved, simulations=12, add_noise=False, temperature=0)
    assert result.n_simulations == 12
    assert moved.side_to_move() == 1


def test_terminal_root_returns_empty():
    b = Board("R3k4/4P4/4R4/9/9/9/9/9/9/4K4 b - - 0 1")
    result = _mcts().run(b, simulations=8, add_noise=False)
    assert result.move is None
    assert result.action_index == -1
    assert isinstance(result, SearchResult)


def test_advance_reuses_subtree_visits():
    b = Board()
    m = _mcts(seed=0)
    first = m.run(b, simulations=40, add_noise=False, temperature=0, reuse=True)
    assert first.action_index in first.visit_counts
    kept = m.advance(first.action_index)
    assert kept
    b.push(first.move)
    warmed = int(m.root.n.sum()) if m.root.expanded and m.root.n is not None else 0
    second = m.run(b, simulations=20, add_noise=False, temperature=0, reuse=True)
    assert second.move is not None
    assert sum(second.visit_counts.values()) == warmed + 20


def test_advance_unknown_action_starts_fresh():
    b = Board()
    m = _mcts(seed=1)
    m.run(b, simulations=8, add_noise=False, temperature=0, reuse=True)
    assert m.advance(ACTION_FROM_TO - 1) is False
    assert m.root.expanded is False


def test_thread_pool_independent_trees():
    def job(seed: int) -> str:
        b = Board()
        return _mcts(seed=seed).run(b, simulations=10, add_noise=False, temperature=0).move.iccs()

    with ThreadPoolExecutor(max_workers=4) as pool:
        iccs = list(pool.map(job, range(8)))
    assert len(iccs) == 8
    assert all(len(s) == 4 for s in iccs)


@pytest.mark.parametrize("simulations", [1, 8])
def test_network_evaluator_smoke(simulations):
    torch = pytest.importorskip("torch")
    from xiangqi_engine.mcts import NetworkEvaluator
    from xiangqi_engine.network import PolicyValueNet

    cfg = _cfg()
    cfg["network"]["blocks"] = 1
    cfg["network"]["channels"] = 8
    cfg["network"]["policy_head_channels"] = 4
    cfg["network"]["value_head_channels"] = 4
    cfg["network"]["value_hidden"] = 16
    enc = Encoder(cfg)
    net = PolicyValueNet(cfg).eval()
    lock = threading.Lock()
    ev = NetworkEvaluator(net, enc, device="cpu", lock=lock)
    result = MCTS(cfg, ev, encoder=enc, seed=0).run(
        Board(), simulations=simulations, add_noise=False, temperature=0
    )
    assert result.move is not None
    assert abs(sum(result.policy) - 1.0) < 1e-5
    del torch
