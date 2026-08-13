import pytest

torch = pytest.importorskip("torch")

from xiangqi_engine import Encoder, load_config
from xiangqi_engine.config import deepcopy_config
from xiangqi_engine.mcts import UniformEvaluator
from xiangqi_engine.network import PolicyValueNet
from xiangqi_engine.replay import ReplayBuffer
from xiangqi_engine.selfplay import play_game
from xiangqi_engine.train import build_optimizer, train_batches


def _tiny():
    cfg = deepcopy_config(load_config())
    cfg["network"]["blocks"] = 1
    cfg["network"]["channels"] = 8
    cfg["network"]["policy_head_channels"] = 4
    cfg["network"]["value_head_channels"] = 4
    cfg["network"]["value_hidden"] = 16
    cfg["mcts"]["simulations"] = 4
    cfg["selfplay"]["max_plies"] = 10
    cfg["train"]["batch_size"] = 8
    cfg["train"]["batches_per_iter"] = 3
    cfg["train"]["lr"] = 0.05
    cfg["replay"]["min_size"] = 1
    cfg["device"] = "cpu"
    return cfg


def test_train_batches_finite_and_can_overfit():
    cfg = _tiny()
    enc = Encoder(cfg)
    rec = play_game(cfg, UniformEvaluator(enc), Encoder(cfg), seed=0, simulations=4)
    buf = ReplayBuffer(cfg, capacity=256)
    # Repeat the same game so the batch is easy to fit.
    for _ in range(8):
        buf.extend(rec.samples)
    net = PolicyValueNet(cfg)
    opt = build_optimizer(net, cfg)
    first = train_batches(net, buf, cfg, n_batches=1, device="cpu", optimizer=opt)
    later = train_batches(net, buf, cfg, n_batches=12, device="cpu", optimizer=opt)
    assert first["loss"] == first["loss"]  # not NaN
    assert later["loss"] <= first["loss"] + 1e-5
