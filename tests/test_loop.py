from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from xiangqi_engine.config import deepcopy_config, load_config
from xiangqi_engine.loop import (
    checkpoint_replay_path,
    load_checkpoint,
    run_loop,
    save_checkpoint,
)
from xiangqi_engine.network import PolicyValueNet


def _loop_cfg(tmp_path: Path):
    cfg = deepcopy_config(load_config())
    cfg["device"] = "cpu"
    cfg["network"]["blocks"] = 1
    cfg["network"]["channels"] = 8
    cfg["network"]["policy_head_channels"] = 4
    cfg["network"]["value_head_channels"] = 4
    cfg["network"]["value_hidden"] = 16
    cfg["mcts"]["simulations"] = 4
    cfg["mcts"]["temperature_moves"] = 2
    cfg["selfplay"]["n_games_per_iter"] = 2
    cfg["selfplay"]["n_workers"] = 1
    cfg["selfplay"]["max_plies"] = 8
    cfg["replay"]["capacity"] = 512
    cfg["replay"]["min_size"] = 4
    cfg["train"]["batch_size"] = 4
    cfg["train"]["batches_per_iter"] = 2
    cfg["eval"]["n_games"] = 2
    cfg["eval"]["mcts_simulations"] = 2
    cfg["eval"]["win_rate_threshold"] = 1.1  # never promote; just run eval
    cfg["loop"]["n_iterations"] = 1
    cfg["loop"]["save_every"] = 1
    cfg["loop"]["eval_every"] = 1
    cfg["paths"]["checkpoint_dir"] = str(tmp_path / "ckpt")
    cfg["paths"]["log_dir"] = str(tmp_path / "logs")
    cfg["paths"]["best_checkpoint"] = str(tmp_path / "ckpt" / "best.pt")
    cfg["paths"]["latest_checkpoint"] = str(tmp_path / "ckpt" / "latest.pt")
    cfg["paths"]["replay_dir"] = str(tmp_path / "replay")
    return cfg


def test_one_training_iteration(tmp_path):
    cfg = _loop_cfg(tmp_path)
    history = run_loop(cfg)
    assert len(history) == 1
    m = history[0]
    assert m["games"] == 2
    assert m["samples"] > 0
    assert m["train"] is not None
    assert "loss" in m["train"]
    assert "eval" in m
    assert (tmp_path / "ckpt" / "iter_0001.pt").is_file()
    assert (tmp_path / "logs" / "train.jsonl").is_file()
    net = PolicyValueNet(cfg)
    it = load_checkpoint(tmp_path / "ckpt" / "iter_0001.pt", net)
    assert it == 1


def test_checkpoint_roundtrip(tmp_path):
    cfg = _loop_cfg(tmp_path)
    net = PolicyValueNet(cfg)
    path = tmp_path / "one.pt"
    save_checkpoint(path, net, cfg, 7)
    other = PolicyValueNet(cfg)
    assert load_checkpoint(path, other) == 7
    for a, b in zip(net.state_dict().values(), other.state_dict().values()):
        assert torch.equal(a, b)


def test_resume_restores_weights_replay_and_iteration(tmp_path):
    cfg = _loop_cfg(tmp_path)
    first = run_loop(cfg)
    assert first[0]["iteration"] == 1
    ckpt = tmp_path / "ckpt" / "iter_0001.pt"
    replay = checkpoint_replay_path(ckpt)
    latest = tmp_path / "ckpt" / "latest.pt"
    assert ckpt.is_file()
    assert replay.is_file()
    assert latest.is_file()

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "optimizer" in payload
    assert "best" in payload
    assert payload["iteration"] == 1

    cfg2 = _loop_cfg(tmp_path)
    second = run_loop(cfg2, resume=str(ckpt))
    assert second[0]["iteration"] == 2
    assert second[0]["buffer"] >= first[0]["buffer"]

    net = PolicyValueNet(cfg)
    best = PolicyValueNet(cfg)
    assert load_checkpoint(ckpt, net, best=best) == 1


def test_old_weight_only_checkpoint_still_resumes(tmp_path):
    cfg = _loop_cfg(tmp_path)
    net = PolicyValueNet(cfg)
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, net, cfg, 3)
    cfg["loop"]["n_iterations"] = 1
    history = run_loop(cfg, resume=str(path))
    assert history[0]["iteration"] == 4
