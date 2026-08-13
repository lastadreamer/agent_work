import numpy as np

from xiangqi_engine import ACTION_FROM_TO, Encoder, load_config
from xiangqi_engine.config import deepcopy_config
from xiangqi_engine.mcts import UniformEvaluator
from xiangqi_engine.replay import ReplayBuffer, sample_from_dense
from xiangqi_engine.selfplay import GameRecord, play_game, play_games


def _fast_cfg():
    cfg = deepcopy_config(load_config())
    cfg["mcts"]["simulations"] = 4
    cfg["mcts"]["temperature_moves"] = 2
    cfg["selfplay"]["max_plies"] = 12
    cfg["selfplay"]["n_games_per_iter"] = 2
    cfg["selfplay"]["n_workers"] = 1
    cfg["replay"]["min_size"] = 1
    return cfg


def test_play_game_emits_samples_with_consistent_z():
    cfg = _fast_cfg()
    enc = Encoder(cfg)
    rec = play_game(cfg, UniformEvaluator(enc), enc, seed=0, simulations=4)
    assert isinstance(rec, GameRecord)
    assert rec.plies == len(rec.samples)
    assert rec.plies > 0
    assert rec.red_z in (-1.0, 0.0, 1.0)
    for s in rec.samples:
        assert s.state.shape[0] == enc.n_planes
        assert s.state.shape[1:] == (10, 9)
        assert abs(float(s.policy_prob.sum()) - 1.0) < 1e-5
        assert s.value in (-1.0, 0.0, 1.0)
        if rec.red_z == 0.0:
            assert s.value == 0.0


def test_replay_buffer_roundtrip_batch():
    cfg = _fast_cfg()
    enc = Encoder(cfg)
    rec = play_game(cfg, UniformEvaluator(enc), Encoder(cfg), seed=1, simulations=3)
    buf = ReplayBuffer(cfg, capacity=64)
    buf.extend(rec.samples)
    assert buf.ready()
    states, policies, values = buf.sample(4)
    assert states.ndim == 4
    assert policies.shape[1] == ACTION_FROM_TO
    assert values.shape[0] == states.shape[0]
    assert np.allclose(policies.sum(axis=1), 1.0, atol=1e-5)


def test_sample_from_dense_keeps_mass():
    state = np.zeros((2, 10, 9), dtype=np.float32)
    pi = [0.0] * ACTION_FROM_TO
    pi[10] = 0.25
    pi[11] = 0.75
    s = sample_from_dense(state, pi, 1.0)
    assert list(s.policy_index) == [10, 11]
    assert abs(float(s.policy_prob.sum()) - 1.0) < 1e-6


def test_process_pool_uniform_games():
    cfg = _fast_cfg()
    cfg["selfplay"]["n_workers"] = 2
    games = play_games(cfg, n_games=2, n_workers=2, seed=3, simulations=2, state_dict=None)
    assert len(games) == 2
    assert all(g.plies == len(g.samples) for g in games)
