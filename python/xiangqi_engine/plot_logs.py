"""Plot training curves from logs/**/train.jsonl. Does not import torch or the C++ engine.

    python -m xiangqi_engine.plot_logs logs/gomoku/train.jsonl
    xiangqi-plot --config config/gomoku.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Parse jsonl. Skip blank / broken lines. Keep the last row per iteration."""
    path = Path(path)
    latest: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"warning: skip {path}:{lineno}: {exc}", file=sys.stderr)
                continue
            if not isinstance(rec, dict) or "iteration" not in rec:
                print(f"warning: skip {path}:{lineno}: no iteration", file=sys.stderr)
                continue
            it = int(rec["iteration"])
            if it not in latest:
                order.append(it)
            latest[it] = rec
    return [latest[i] for i in sorted(order)]


def log_path_from_config(config_path: str | Path) -> Path:
    """Read paths.log_dir from a JSON config without going through load_config()."""
    with Path(config_path).open(encoding="utf-8") as fh:
        raw = json.load(fh)
    log_dir = Path(raw.get("paths", {}).get("log_dir") or "logs")
    return log_dir / "train.jsonl"


def _num(rec: dict[str, Any], *keys: str) -> float | None:
    cur: Any = rec
    for key in keys:
        if not isinstance(cur, dict) or key not in cur or cur[key] is None:
            return None
        cur = cur[key]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def _int(rec: dict[str, Any], *keys: str) -> int | None:
    value = _num(rec, *keys)
    return None if value is None else int(value)


def series_from_records(records: list[dict[str, Any]]) -> dict[str, list]:
    """Turn jsonl rows into aligned lists. Missing eval/train are None."""
    iters = [int(r["iteration"]) for r in records]
    out: dict[str, list] = {
        "iteration": iters,
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "win_rate": [],
        "eval_wins": [],
        "eval_losses": [],
        "eval_draws": [],
        "eval_games": [],
        "promoted": [],
        "samples": [],
        "games": [],
        "buffer": [],
        "plies": [],
        "sp_red": [],
        "sp_black": [],
        "sp_draw": [],
        "sp_red_frac": [],
        "sp_black_frac": [],
        "sp_draw_frac": [],
        "selfplay_sec": [],
        "train_sec": [],
        "eval_sec": [],
        "sec": [],
    }
    for rec in records:
        train = rec.get("train") if isinstance(rec.get("train"), dict) else None
        ev = rec.get("eval") if isinstance(rec.get("eval"), dict) else None
        sp = rec.get("selfplay") if isinstance(rec.get("selfplay"), dict) else {}
        out["loss"].append(_num(train, "loss") if train else None)
        out["policy_loss"].append(_num(train, "policy_loss") if train else None)
        out["value_loss"].append(_num(train, "value_loss") if train else None)
        out["win_rate"].append(_num(ev, "win_rate") if ev else None)
        w = _int(ev, "wins") if ev else None
        l = _int(ev, "losses") if ev else None
        d = _int(ev, "draws") if ev else None
        out["eval_wins"].append(w)
        out["eval_losses"].append(l)
        out["eval_draws"].append(d)
        if w is None or l is None or d is None:
            out["eval_games"].append(None)
        else:
            out["eval_games"].append(w + l + d)
        out["promoted"].append(bool(rec.get("promoted")))
        samples = _num(rec, "samples")
        games = _num(rec, "games")
        out["samples"].append(samples)
        out["games"].append(games)
        out["buffer"].append(_num(rec, "buffer"))
        if samples is not None and games and games > 0:
            out["plies"].append(samples / games)
        else:
            out["plies"].append(None)
        red = float(sp.get("red") or 0)
        black = float(sp.get("black") or 0)
        draw = float(sp.get("draw") or 0)
        total = red + black + draw
        out["sp_red"].append(red)
        out["sp_black"].append(black)
        out["sp_draw"].append(draw)
        out["sp_red_frac"].append(red / total if total else None)
        out["sp_black_frac"].append(black / total if total else None)
        out["sp_draw_frac"].append(draw / total if total else None)
        out["selfplay_sec"].append(_num(rec, "selfplay_sec"))
        out["train_sec"].append(_num(rec, "train_sec"))
        out["eval_sec"].append(_num(rec, "eval_sec"))
        out["sec"].append(_num(rec, "sec"))
    return out


def promotion_iters(records: list[dict[str, Any]]) -> list[int]:
    return [int(r["iteration"]) for r in records if r.get("promoted")]


def summarize(records: list[dict[str, Any]]) -> str:
    if not records:
        return "no iterations"
    s = series_from_records(records)
    promos = promotion_iters(records)
    first, last = s["iteration"][0], s["iteration"][-1]
    lines = [
        f"iters {first}–{last}  n={len(records)}",
        f"promoted {len(promos)} time(s)"
        + (": " + ", ".join(str(i) for i in promos) if promos else ""),
    ]
    last_rec = records[-1]
    train = last_rec.get("train") or {}
    if train:
        parts = []
        for key, label in (
            ("loss", "loss"),
            ("policy_loss", "policy"),
            ("value_loss", "value"),
        ):
            if key in train and train[key] is not None:
                parts.append(f"{label}={float(train[key]):.4f}")
        if parts:
            lines.append("last train  " + " ".join(parts))
    ev = last_rec.get("eval")
    if isinstance(ev, dict) and ev.get("win_rate") is not None:
        lines.append(
            f"last eval   wr={float(ev['win_rate']):.3f}  "
            f"W{ev.get('wins', '?')} L{ev.get('losses', '?')} D{ev.get('draws', '?')}"
        )
    sp = last_rec.get("selfplay") or {}
    if sp:
        line = (
            f"last selfplay  R{sp.get('red', 0)} B{sp.get('black', 0)} D{sp.get('draw', 0)}"
            f"  {last_rec.get('games', '?')} games"
        )
        if s["plies"][-1] is not None:
            line += f"  avg {s['plies'][-1]:.1f} plies"
        lines.append(line)
    return "\n".join(lines)


def _xy(xs: list, ys: list) -> tuple[list, list]:
    px, py = [], []
    for x, y in zip(xs, ys):
        if y is None:
            continue
        px.append(x)
        py.append(y)
    return px, py


def _mark_promos(ax, iters: list[int]) -> None:
    for i in iters:
        ax.axvline(i, color="#c0392b", alpha=0.35, linewidth=1, linestyle="--", zorder=0)


def _time_scale(values: list) -> tuple[list, str]:
    nums = [v for v in values if v is not None]
    if not nums:
        return values, "s"
    peak = max(nums)
    if peak >= 7200:
        return [None if v is None else v / 3600.0 for v in values], "h"
    if peak >= 180:
        return [None if v is None else v / 60.0 for v in values], "min"
    return values, "s"


def plot_records(
    records: list[dict[str, Any]],
    out_path: str | Path | None = None,
    *,
    title: str | None = None,
    show: bool = False,
    dpi: int = 140,
):
    """Draw a 3×2 figure. Policy/total loss share an axis; value loss uses the right axis."""
    if not records:
        raise ValueError("no records to plot")
    try:
        if not show:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to plot. Install with:\n"
            "  uv pip install matplotlib\n"
            "  # or: uv sync --extra train"
        ) from exc

    s = series_from_records(records)
    x = s["iteration"]
    promos = promotion_iters(records)
    promo_x = [i for i, flag in zip(x, s["promoted"]) if flag]

    fig, axes = plt.subplots(3, 2, figsize=(14.5, 10.5), sharex=True)
    fig.suptitle(title or "train.jsonl", fontsize=13)

    ax = axes[0, 0]
    _mark_promos(ax, promos)
    ax.plot(*_xy(x, s["loss"]), label="loss", color="#1f77b4")
    ax.plot(*_xy(x, s["policy_loss"]), label="policy", color="#2ca02c")
    ax.set_ylabel("policy / total")
    ax.set_title("loss")
    ax.grid(True, alpha=0.3)
    ax_v = ax.twinx()
    ax_v.plot(*_xy(x, s["value_loss"]), label="value", color="#ff7f0e")
    ax_v.set_ylabel("value", color="#ff7f0e")
    if promo_x:
        pts = [(i, _lookup(x, s["loss"], i)) for i in promo_x]
        pts = [(a, b) for a, b in pts if b is not None]
        if pts:
            ax.scatter(
                [a for a, _ in pts],
                [b for _, b in pts],
                marker="*",
                s=80,
                color="#c0392b",
                zorder=5,
                label="promoted",
            )
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_v.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8)

    ax = axes[0, 1]
    _mark_promos(ax, promos)
    px, py = _xy(x, s["win_rate"])
    ax.plot(px, py, marker="o", markersize=3, label="eval wr")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("win rate")
    ax.set_title("eval vs best  (★ = promoted)")
    ax.grid(True, alpha=0.3)
    if promo_x:
        pts = [(i, _lookup(x, s["win_rate"], i)) for i in promo_x]
        pts = [(a, b) for a, b in pts if b is not None]
        if pts:
            ax.scatter(
                [a for a, _ in pts],
                [b for _, b in pts],
                marker="*",
                s=110,
                color="#c0392b",
                zorder=5,
                label="promoted",
            )
    ax.legend(loc="best", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(*_xy(x, s["eval_wins"]), color="#2ca02c", alpha=0.4, label="W")
    ax2.plot(*_xy(x, s["eval_losses"]), color="#d62728", alpha=0.4, label="L")
    ax2.plot(*_xy(x, s["eval_draws"]), color="#7f7f7f", alpha=0.4, label="D")
    ax2.set_ylabel("eval W / L / D")

    ax = axes[1, 0]
    _mark_promos(ax, promos)
    ax.plot(*_xy(x, s["sp_red_frac"]), label="red/white win", color="#d62728")
    ax.plot(*_xy(x, s["sp_black_frac"]), label="black win", color="#1f1f1f")
    ax.plot(*_xy(x, s["sp_draw_frac"]), label="draw", color="#7f7f7f")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("fraction")
    ax.set_title("self-play outcomes")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    _mark_promos(ax, promos)
    ax.plot(*_xy(x, s["plies"]), label="avg plies/game", color="#1f77b4")
    ax.set_ylabel("plies")
    ax.set_title("game length  (samples / games)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[2, 0]
    _mark_promos(ax, promos)
    ax.plot(*_xy(x, s["buffer"]), label="buffer", color="#1f77b4")
    ax.plot(*_xy(x, s["samples"]), label="samples this iter", color="#ff7f0e")
    ax.set_ylabel("positions")
    ax.set_xlabel("iteration")
    ax.set_title("replay")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[2, 1]
    _mark_promos(ax, promos)
    times = {
        "self-play": s["selfplay_sec"],
        "train": s["train_sec"],
        "eval": s["eval_sec"],
        "iter total": s["sec"],
    }
    scaled = {k: _time_scale(v) for k, v in times.items()}
    unit = scaled["iter total"][1]
    for name, (vals, u) in scaled.items():
        if u != unit:
            factor = {"s": 1, "min": 60, "h": 3600}[u] / {"s": 1, "min": 60, "h": 3600}[unit]
            vals = [None if v is None else v * factor for v in vals]
        ax.plot(*_xy(x, vals), label=name)
    ax.set_ylabel(unit)
    ax.set_xlabel("iteration")
    ax.set_title("wall time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _lookup(xs: list, ys: list, x: int):
    for a, b in zip(xs, ys):
        if a == x:
            return b
    return None


def default_jsonl_candidates() -> list[Path]:
    return [
        Path("logs/train.jsonl"),
        Path("logs/gomoku/train.jsonl"),
        Path("logs/smoke/train.jsonl"),
        Path("logs/gomoku_smoke/train.jsonl"),
    ]


def resolve_jsonl(path: str | None, config: str | None) -> Path:
    if path:
        p = Path(path)
        if p.is_dir():
            p = p / "train.jsonl"
        return p
    if config:
        return log_path_from_config(config)
    found = [p for p in default_jsonl_candidates() if p.is_file()]
    if len(found) == 1:
        return found[0]
    if not found:
        raise SystemExit(
            "no train.jsonl found; pass a path or --config\n"
            "looked for: " + ", ".join(str(p) for p in default_jsonl_candidates())
        )
    raise SystemExit(
        "several train.jsonl files; pass one explicitly:\n"
        + "\n".join(f"  {p}" for p in found)
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot loss / eval / self-play / promotions from train.jsonl"
    )
    parser.add_argument(
        "jsonl",
        nargs="?",
        default=None,
        help="path to train.jsonl, or a log directory",
    )
    parser.add_argument("--config", default=None, help="JSON config; uses paths.log_dir")
    parser.add_argument("-o", "--out", default=None, help="output PNG (default: next to the jsonl)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args(argv)

    jsonl = resolve_jsonl(args.jsonl, args.config)
    if not jsonl.is_file():
        raise SystemExit(f"not found: {jsonl}")
    records = load_records(jsonl)
    if not records:
        raise SystemExit(f"no usable rows in {jsonl}")
    print(summarize(records), flush=True)
    out = Path(args.out) if args.out else jsonl.with_suffix(".png")
    plot_records(
        records,
        out,
        title=args.title or str(jsonl),
        show=args.show,
        dpi=args.dpi,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
