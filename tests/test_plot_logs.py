import json
from pathlib import Path

import pytest

from xiangqi_engine.plot_logs import (
    load_records,
    log_path_from_config,
    plot_records,
    promotion_iters,
    series_from_records,
    summarize,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_load_skips_bad_lines_and_keeps_last_iteration(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"iteration": 1, "promoted": False, "train": {"loss": 1.0}}),
                "not json",
                json.dumps({"iteration": 2, "promoted": True, "train": {"loss": 0.5}}),
                json.dumps({"iteration": 1, "promoted": True, "train": {"loss": 0.9}}),
                "",
                json.dumps({"no_iteration": 3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    recs = load_records(path)
    assert [r["iteration"] for r in recs] == [1, 2]
    assert recs[0]["train"]["loss"] == 0.9
    assert recs[0]["promoted"] is True
    assert promotion_iters(recs) == [1, 2]


def test_series_handles_missing_train_and_eval():
    recs = [
        {
            "iteration": 1,
            "games": 10,
            "samples": 200,
            "buffer": 200,
            "selfplay": {"red": 4, "black": 5, "draw": 1},
            "train": None,
            "promoted": False,
            "selfplay_sec": 12.0,
            "sec": 12.0,
        },
        {
            "iteration": 2,
            "games": 10,
            "samples": 180,
            "buffer": 380,
            "selfplay": {"red": 6, "black": 4, "draw": 0},
            "train": {"loss": 5.2, "policy_loss": 5.1, "value_loss": 0.04},
            "eval": {"wins": 30, "losses": 0, "draws": 0, "win_rate": 1.0},
            "promoted": True,
            "selfplay_sec": 11.0,
            "train_sec": 2.0,
            "eval_sec": 8.0,
            "sec": 21.0,
        },
    ]
    s = series_from_records(recs)
    assert s["loss"] == [None, 5.2]
    assert s["win_rate"] == [None, 1.0]
    assert s["plies"][0] == 20.0
    assert s["sp_red_frac"][0] == pytest.approx(0.4)
    assert s["promoted"] == [False, True]
    text = summarize(recs)
    assert "promoted 1 time(s): 2" in text
    assert "policy=5.1000" in text


def test_log_path_from_config(tmp_path):
    cfg = tmp_path / "g.json"
    cfg.write_text(
        json.dumps({"paths": {"log_dir": "logs/gomoku"}, "game": "gomoku"}),
        encoding="utf-8",
    )
    assert log_path_from_config(cfg) == Path("logs/gomoku") / "train.jsonl"


def test_plot_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    recs = [
        {
            "iteration": i,
            "games": 8,
            "samples": 200 - i,
            "buffer": 200 * i,
            "selfplay": {"red": 3, "black": 4, "draw": 1},
            "train": {
                "loss": 5.0 - 0.1 * i,
                "policy_loss": 4.8 - 0.1 * i,
                "value_loss": 0.2 / i,
            },
            "eval": {
                "wins": 30 if i == 4 else 8,
                "losses": 0 if i == 4 else 8,
                "draws": 0,
                "win_rate": 1.0 if i == 4 else 0.5,
            }
            if i % 2 == 0
            else None,
            "promoted": i == 4,
            "selfplay_sec": 10.0,
            "train_sec": 1.0,
            "eval_sec": 2.0 if i % 2 == 0 else None,
            "sec": 13.0,
        }
        for i in range(1, 7)
    ]
    out = tmp_path / "curves.png"
    plot_records(recs, out, title="test")
    assert out.is_file()
    assert out.stat().st_size > 1000
