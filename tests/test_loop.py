from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from xiangqi_engine.config import deepcopy_config, load_config, resolve_device
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


def _load_pair(cfg, tmp_path: Path):
    learner = PolicyValueNet(cfg)
    best = PolicyValueNet(cfg)
    load_checkpoint(tmp_path / "ckpt" / "latest.pt", learner)
    load_checkpoint(tmp_path / "ckpt" / "best.pt", best)
    return learner, best


def _weights_equal(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_selfplay_uses_frozen_best_when_not_promoted(tmp_path, monkeypatch):
    from xiangqi_engine import loop as loop_mod

    snapshots = []
    orig = loop_mod.play_games

    def wrapped(cfg, evaluator=None, **kwargs):
        if evaluator is not None:
            snapshots.append(
                {k: v.detach().cpu().clone() for k, v in evaluator.net.state_dict().items()}
            )
        return orig(cfg, evaluator=evaluator, **kwargs)

    monkeypatch.setattr(loop_mod, "play_games", wrapped)
    cfg = _loop_cfg(tmp_path)
    history = run_loop(cfg)
    assert history[0]["promoted"] is False
    assert snapshots
    learner, best = _load_pair(cfg, tmp_path)
    for a, b in zip(snapshots[0].values(), best.state_dict().values()):
        assert torch.equal(a, b)
    assert not _weights_equal(learner, best)


def test_later_selfplay_keeps_unpromoted_best(tmp_path, monkeypatch):
    from xiangqi_engine import loop as loop_mod

    snapshots = []
    orig = loop_mod.play_games

    def wrapped(cfg, evaluator=None, **kwargs):
        if evaluator is not None:
            snapshots.append(
                {k: v.detach().cpu().clone() for k, v in evaluator.net.state_dict().items()}
            )
        return orig(cfg, evaluator=evaluator, **kwargs)

    monkeypatch.setattr(loop_mod, "play_games", wrapped)
    cfg = _loop_cfg(tmp_path)
    cfg["loop"]["n_iterations"] = 2
    history = run_loop(cfg)
    assert [m["promoted"] for m in history] == [False, False]
    assert len(snapshots) == 2
    learner, best = _load_pair(cfg, tmp_path)
    for snap in snapshots:
        for a, b in zip(snap.values(), best.state_dict().values()):
            assert torch.equal(a, b)
    assert not _weights_equal(learner, best)


def test_promotion_copies_learner_into_best(tmp_path):
    cfg = _loop_cfg(tmp_path)
    cfg["eval"]["win_rate_threshold"] = -1.0
    history = run_loop(cfg)
    assert history[0]["promoted"] is True
    learner, best = _load_pair(cfg, tmp_path)
    assert _weights_equal(learner, best)


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


def test_resume_skips_truncated_replay(tmp_path):
    cfg = _loop_cfg(tmp_path)
    first = run_loop(cfg)
    ckpt = tmp_path / "ckpt" / "iter_0001.pt"
    sidecar = checkpoint_replay_path(ckpt)
    sidecar.write_bytes(b"")
    cfg2 = _loop_cfg(tmp_path)
    history = run_loop(cfg2, resume=str(ckpt))
    assert history[0]["iteration"] == 2
    assert history[0]["buffer"] >= first[0]["games"]


def test_smoke_config_is_cpu_only():
    cfg = load_config("config/smoke.json")
    assert cfg["device"] == "cpu"
    assert cfg["selfplay"]["device"] == "cpu"
    assert cfg["selfplay"]["n_workers"] == 1
    assert resolve_device(cfg) == "cpu"
