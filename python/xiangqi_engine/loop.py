"""AlphaZero iteration: self-play → train → eval → maybe promote.

    python -m xiangqi_engine.loop --config config/default.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from xiangqi_engine.config import Cfg, deepcopy_config, load_config, resolve_device
from xiangqi_engine.encode import Encoder
from xiangqi_engine.evaluate import play_match
from xiangqi_engine.mcts import NetworkEvaluator, UniformEvaluator
from xiangqi_engine.replay import ReplayBuffer
from xiangqi_engine.selfplay import play_games
from xiangqi_engine.train import train_batches


def save_checkpoint(path: str | Path, net, cfg: Cfg, iteration: int, extra: dict | None = None) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
        "iteration": iteration,
        "config": dict(cfg),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, net) -> int:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["model"])
    return int(payload.get("iteration", 0))


def _cpu_state_dict(net) -> dict:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def run_iteration(
    cfg: Cfg,
    net,
    best,
    buffer: ReplayBuffer,
    iteration: int,
    device: str,
) -> dict:
    t0 = time.time()
    n_workers = int(cfg["selfplay"]["n_workers"])
    if n_workers > 1:
        games = play_games(
            cfg,
            n_workers=n_workers,
            seed=int(cfg["seed"]) + iteration * 10007,
            state_dict=_cpu_state_dict(net),
        )
    else:
        enc = Encoder(cfg)
        ev = NetworkEvaluator(net, enc, device=device)
        games = play_games(
            cfg,
            evaluator=ev,
            n_workers=1,
            seed=int(cfg["seed"]) + iteration * 10007,
        )
    n_samples = 0
    outcomes = {"red": 0, "black": 0, "draw": 0}
    for g in games:
        buffer.extend(g.samples)
        n_samples += len(g.samples)
        if g.red_z > 0:
            outcomes["red"] += 1
        elif g.red_z < 0:
            outcomes["black"] += 1
        else:
            outcomes["draw"] += 1

    metrics: dict = {
        "iteration": iteration,
        "games": len(games),
        "samples": n_samples,
        "buffer": len(buffer),
        "selfplay": outcomes,
        "selfplay_sec": time.time() - t0,
    }

    if buffer.ready():
        t1 = time.time()
        metrics["train"] = train_batches(net, buffer, cfg, device=device)
        metrics["train_sec"] = time.time() - t1
    else:
        metrics["train"] = None

    eval_every = int(cfg["loop"].get("eval_every", 1))
    promoted = False
    if metrics["train"] is not None and eval_every > 0 and iteration % eval_every == 0:
        t2 = time.time()
        enc = Encoder(cfg)
        chal = NetworkEvaluator(net, enc, device=device)
        hold = NetworkEvaluator(best, enc, device=device)
        match = play_match(cfg, chal, hold, seed=int(cfg["seed"]) + iteration)
        metrics["eval"] = {
            "wins": match.wins,
            "losses": match.losses,
            "draws": match.draws,
            "win_rate": match.win_rate,
        }
        metrics["eval_sec"] = time.time() - t2
        if match.win_rate >= float(cfg["eval"]["win_rate_threshold"]):
            best.load_state_dict(copy.deepcopy(net.state_dict()))
            promoted = True
    metrics["promoted"] = promoted
    metrics["sec"] = time.time() - t0
    return metrics


def run_loop(cfg: Cfg | None = None, resume: str | None = None) -> list[dict]:
    from xiangqi_engine.network import PolicyValueNet

    cfg = cfg if cfg is not None else load_config()
    device = resolve_device(cfg)
    net = PolicyValueNet(cfg).to(device)
    start_iter = 0
    if resume:
        start_iter = load_checkpoint(resume, net)
    best = PolicyValueNet(cfg).to(device)
    best.load_state_dict(copy.deepcopy(net.state_dict()))
    buffer = ReplayBuffer(cfg)
    paths = cfg["paths"]
    ckpt_dir = Path(paths["checkpoint_dir"])
    log_dir = Path(paths["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.jsonl"
    history = []
    n_iters = int(cfg["loop"]["n_iterations"])
    save_every = int(cfg["loop"].get("save_every", 1))

    for i in range(start_iter + 1, start_iter + n_iters + 1):
        metrics = run_iteration(cfg, net, best, buffer, i, device)
        history.append(metrics)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics) + "\n")
        print(metrics, flush=True)
        if save_every > 0 and i % save_every == 0:
            save_checkpoint(ckpt_dir / f"iter_{i:04d}.pt", net, cfg, i, {"metrics": metrics})
            save_checkpoint(paths["best_checkpoint"], best, cfg, i)
    return history


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AlphaZero Xiangqi training loop")
    parser.add_argument("--config", default=None, help="JSON config path")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--checkpoint", default=None, help="resume weights")
    args = parser.parse_args(argv)
    cfg = deepcopy_config(load_config(args.config))
    if args.iterations is not None:
        cfg["loop"]["n_iterations"] = args.iterations
    run_loop(cfg, resume=args.checkpoint)


if __name__ == "__main__":
    main()
