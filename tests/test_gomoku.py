from pathlib import Path

import pytest

from xiangqi_engine._xiangqi import BLACK, Outcome
from xiangqi_engine.config import deepcopy_config, load_config
from xiangqi_engine.game import game_name, make_board, make_encoder
from xiangqi_engine.gomoku.board import GomokuBoard
from xiangqi_engine.mcts import MCTS, UniformEvaluator, terminal_value
from xiangqi_engine.selfplay import play_game


def _smoke_cfg(tmp_path: Path | None = None):
    cfg = deepcopy_config(load_config("config/gomoku_smoke.json"))
    if tmp_path is not None:
        cfg["paths"]["checkpoint_dir"] = str(tmp_path / "ckpt")
        cfg["paths"]["log_dir"] = str(tmp_path / "logs")
        cfg["paths"]["best_checkpoint"] = str(tmp_path / "ckpt" / "best.pt")
        cfg["paths"]["latest_checkpoint"] = str(tmp_path / "ckpt" / "latest.pt")
        cfg["paths"]["replay_dir"] = str(tmp_path / "replay")
    return cfg


def test_gomoku_config_and_factories():
    cfg = _smoke_cfg()
    assert game_name(cfg) == "gomoku"
    b = make_board(cfg)
    enc = make_encoder(cfg)
    assert isinstance(b, GomokuBoard)
    assert b.size == 9
    assert b.side_to_move() == BLACK
    assert enc.action_size == 81
    assert enc.tensor(b).shape == (enc.n_planes, 9, 9)


def test_five_in_a_row_wins():
    b = GomokuBoard(9)
    for iccs in ("a0", "a8", "b0", "b8", "c0", "c8", "d0", "d8", "e0"):
        b.push_iccs(iccs)
    term = b.terminal()
    assert term.outcome == Outcome.BLACK_WIN
    assert term.reason == "FIVE"
    assert terminal_value(b) == -1.0  # white to move, black already won


def test_make_unmake_restores():
    b = GomokuBoard(9)
    fen0 = b.fen()
    b.push_iccs("e4")
    assert b.fen() != fen0
    b.unmake_move()
    assert b.fen() == fen0
    assert b.ply() == 0


def test_mcts_restores_gomoku_board():
    cfg = _smoke_cfg()
    b = make_board(cfg)
    enc = make_encoder(cfg)
    fen = b.fen()
    ply = b.ply()
    result = MCTS(cfg, UniformEvaluator(enc), encoder=enc, seed=0).run(
        b, simulations=16, add_noise=False, temperature=0
    )
    assert b.fen() == fen
    assert b.ply() == ply
    assert result.move is not None
    assert b.is_legal(result.move)
    assert len(result.policy) == 81


def test_batched_mcts_restores_gomoku_board():
    cfg = _smoke_cfg()
    cfg["mcts"]["batch_size"] = 8
    cfg["mcts"]["virtual_loss"] = 1
    b = make_board(cfg)
    enc = make_encoder(cfg)
    fen = b.fen()
    ply = b.ply()
    result = MCTS(cfg, UniformEvaluator(enc), encoder=enc, seed=0).run(
        b, simulations=16, add_noise=False, temperature=0
    )
    assert b.fen() == fen
    assert b.ply() == ply
    assert result.move is not None
    assert b.is_legal(result.move)
    assert sum(result.visit_counts.values()) == 16


def test_selfplay_game_emits_samples():
    cfg = _smoke_cfg()
    rec = play_game(cfg, seed=0, simulations=4)
    assert rec.plies == len(rec.samples)
    assert rec.plies > 0
    for s in rec.samples:
        assert s.state.shape[1:] == (9, 9)
        assert abs(float(s.policy_prob.sum()) - 1.0) < 1e-5


def test_xiangqi_default_unchanged():
    cfg = load_config("config/default.json")
    assert game_name(cfg) == "xiangqi"
    assert cfg["action"]["size"] == 8100


@pytest.mark.skipif(pytest.importorskip("torch") is None, reason="torch")
def test_gomoku_one_training_iteration(tmp_path):
    from xiangqi_engine.loop import run_loop

    cfg = _smoke_cfg(tmp_path)
    history = run_loop(cfg)
    assert len(history) == 1
    assert history[0]["games"] == 2
    assert history[0]["train"] is not None
    assert (tmp_path / "ckpt" / "iter_0001.pt").is_file()
