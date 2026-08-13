"""AlphaZero iteration: self-play → train → eval → maybe promote.

    python -m xiangqi_engine.loop --config config/default.json
    python -m xiangqi_engine.loop --resume
    python -m xiangqi_engine.loop --checkpoint checkpoints/iter_0005.pt
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
from xiangqi_engine.mcts import NetworkEvaluator
from xiangqi_engine.replay import ReplayBuffer
from xiangqi_engine.selfplay import play_games
from xiangqi_engine.train import build_optimizer, train_batches


def checkpoint_replay_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".replay.pkl")


def latest_checkpoint_path(cfg: Cfg) -> Path:
    paths = cfg["paths"]
    named = paths.get("latest_checkpoint")
    if named:
        return Path(named)
    return Path(paths["checkpoint_dir"]) / "latest.pt"


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


def load_checkpoint(path: str | Path, net, best=None, optimizer=None) -> int:
    """Load learner weights. Optionally restore best net and optimizer.

    Returns the saved iteration. Older files that only have `model` still work
    (play UI and early checkpoints). MCTS trees are not stored: each move
    searches from scratch, which is the usual AlphaZero setup.
    """
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    net.load_state_dict(payload["model"])
    if best is not None:
        if payload.get("best") is not None:
            best.load_state_dict(payload["best"])
        else:
            best.load_state_dict(copy.deepcopy(net.state_dict()))
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("iteration", 0))


def optimizer_to_device(optimizer, device: str) -> None:
    import torch

    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_training_checkpoint(
    path: str | Path,
    net,
    cfg: Cfg,
    iteration: int,
    best=None,
    optimizer=None,
    buffer: ReplayBuffer | None = None,
    extra: dict | None = None,
) -> None:
    extra = dict(extra or {})
    if best is not None:
        extra["best"] = {k: v.detach().cpu() for k, v in best.state_dict().items()}
    if optimizer is not None:
        extra["optimizer"] = optimizer.state_dict()
    save_checkpoint(path, net, cfg, iteration, extra)
    if buffer is not None:
        buffer.save(checkpoint_replay_path(path))


def _cpu_state_dict(net) -> dict:
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def run_iteration(
    cfg: Cfg,
    net,
    best,
    buffer: ReplayBuffer,
    iteration: int,
    device: str,
    optimizer=None,
) -> dict:
    t0 = time.time()
    n_workers = int(cfg["selfplay"]["n_workers"])
    print(
        f"iter {iteration}: self-play "
        f"{cfg['selfplay']['n_games_per_iter']} games, {n_workers} workers",
        flush=True,
    )
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
        print(f"iter {iteration}: train buffer={len(buffer)}", flush=True)
        metrics["train"] = train_batches(
            net,
            buffer,
            cfg,
            device=device,
            optimizer=optimizer,
            seed=int(cfg["seed"]) + iteration,
        )
        metrics["train_sec"] = time.time() - t1
    else:
        metrics["train"] = None

    eval_every = int(cfg["loop"].get("eval_every", 1))
    promoted = False
    if metrics["train"] is not None and eval_every > 0 and iteration % eval_every == 0:
        t2 = time.time()
        print(f"iter {iteration}: eval {cfg['eval']['n_games']} games", flush=True)
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


def _try_load_replay(buffer: ReplayBuffer, resume: str | Path, cfg: Cfg) -> Path | None:
    sidecar = checkpoint_replay_path(resume)
    if sidecar.is_file():
        buffer.load(sidecar)
        return sidecar
    fallback = Path(cfg["paths"].get("replay_dir", "data/replay")) / "buffer.pkl"
    if fallback.is_file():
        buffer.load(fallback)
        return fallback
    return None


def run_loop(cfg: Cfg | None = None, resume: str | None = None) -> list[dict]:
    cfg = cfg if cfg is not None else load_config()
    device = resolve_device(cfg)
    n_iters = int(cfg["loop"]["n_iterations"])
    print(
        f"loop start device={device} "
        f"net={cfg['network']['blocks']}x{cfg['network']['channels']} "
        f"selfplay={cfg['selfplay']['n_games_per_iter']} games "
        f"x {cfg['selfplay']['n_workers']} workers "
        f"sims={cfg['mcts']['simulations']} "
        f"iters={n_iters}",
        flush=True,
    )
    from xiangqi_engine.network import PolicyValueNet

    net = PolicyValueNet(cfg).to(device)
    best = PolicyValueNet(cfg).to(device)
    optimizer = build_optimizer(net, cfg)
    buffer = ReplayBuffer(cfg)
    start_iter = 0
    if resume:
        start_iter = load_checkpoint(resume, net, best=best, optimizer=optimizer)
        optimizer_to_device(optimizer, device)
        replay_src = _try_load_replay(buffer, resume, cfg)
        print(
            f"resume {resume}: iteration={start_iter} buffer={len(buffer)}"
            f" replay={replay_src or '(empty)'}",
            flush=True,
        )
    else:
        best.load_state_dict(copy.deepcopy(net.state_dict()))
    paths = cfg["paths"]
    ckpt_dir = Path(paths["checkpoint_dir"])
    log_dir = Path(paths["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.jsonl"
    latest = latest_checkpoint_path(cfg)
    history = []
    save_every = int(cfg["loop"].get("save_every", 1))

    for i in range(start_iter + 1, start_iter + n_iters + 1):
        metrics = run_iteration(cfg, net, best, buffer, i, device, optimizer=optimizer)
        history.append(metrics)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics) + "\n")
        print(metrics, flush=True)
        if save_every > 0 and i % save_every == 0:
            extra = {"metrics": metrics}
            save_training_checkpoint(
                ckpt_dir / f"iter_{i:04d}.pt",
                net,
                cfg,
                i,
                best=best,
                optimizer=optimizer,
                buffer=buffer,
                extra=extra,
            )
            save_training_checkpoint(
                latest,
                net,
                cfg,
                i,
                best=best,
                optimizer=optimizer,
                buffer=buffer,
                extra=extra,
            )
            save_checkpoint(paths["best_checkpoint"], best, cfg, i)
    return history


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AlphaZero Xiangqi training loop")
    parser.add_argument("--config", default=None, help="JSON config path")
    parser.add_argument("--iterations", type=int, default=None, help="how many more iterations to run")
    parser.add_argument("--checkpoint", default=None, help="resume from this .pt snapshot")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from paths.latest_checkpoint (weights, best, optimizer, replay)",
    )
    args = parser.parse_args(argv)
    cfg = deepcopy_config(load_config(args.config))
    if args.iterations is not None:
        cfg["loop"]["n_iterations"] = args.iterations
    resume = args.checkpoint
    if resume is None and args.resume:
        resume = str(latest_checkpoint_path(cfg))
        if not Path(resume).is_file():
            raise SystemExit(f"no latest checkpoint at {resume}")
    run_loop(cfg, resume=resume)


if __name__ == "__main__":
    main()
