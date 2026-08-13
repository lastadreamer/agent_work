"""Load the single JSON control file used by encode / network / later training."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

from xiangqi_engine._xiangqi import ACTION_FROM_TO, EncodeSpec

_DOC_PREFIX = "_"


class Cfg(dict):
    """Dict that also allows cfg.network.blocks attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return _wrap(value)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self[name] = value


def _wrap(value: Any) -> Any:
    if isinstance(value, Cfg):
        return value
    if isinstance(value, dict):
        return Cfg(value)
    return value


def _strip_doc(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_doc(v) for k, v in obj.items() if not str(k).startswith(_DOC_PREFIX)}
    if isinstance(obj, list):
        return [_strip_doc(v) for v in obj]
    return obj


def default_config_path() -> Path:
    env = os.environ.get("XIANGQI_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "config" / "default.json",
        here.parents[2] / "config" / "default.json",
        here.parent / "data" / "default.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "config/default.json not found; set XIANGQI_CONFIG or pass a path to load_config()"
    )


def load_config(path: str | os.PathLike[str] | None = None) -> Cfg:
    cfg_path = Path(path).expanduser().resolve() if path else default_config_path()
    with cfg_path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    cfg = Cfg(_strip_doc(raw))
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    encode = cfg["encode"]
    action = cfg["action"]
    if encode.get("perspective") not in ("current_player", "absolute"):
        raise ValueError("encode.perspective must be 'current_player' or 'absolute'")
    if int(encode["history_length"]) < 1:
        raise ValueError("encode.history_length must be >= 1")
    if action.get("encoding") != "from_to":
        raise ValueError("only action.encoding='from_to' is implemented")
    if int(action["size"]) != ACTION_FROM_TO:
        raise ValueError(f"action.size must be {ACTION_FROM_TO} for from_to encoding")
    net = cfg["network"]
    for key in ("blocks", "channels", "policy_head_channels", "value_head_channels", "value_hidden"):
        if int(net[key]) < 1:
            raise ValueError(f"network.{key} must be >= 1")


def spec_from_config(cfg: Mapping[str, Any]) -> EncodeSpec:
    encode = cfg["encode"]
    planes = encode["planes"]
    spec = EncodeSpec()
    spec.perspective_current_player = encode.get("perspective", "current_player") == "current_player"
    spec.our_pieces = bool(planes["our_pieces"])
    spec.opp_pieces = bool(planes["opp_pieces"])
    spec.side_to_move = bool(planes["side_to_move"])
    spec.halfmove = bool(planes["halfmove"])
    spec.fullmove = bool(planes["fullmove"])
    spec.ones = bool(planes["ones"])
    spec.halfmove_scale = float(encode.get("halfmove_scale", 120))
    spec.fullmove_scale = float(encode.get("fullmove_scale", 400))
    spec.history_length = int(encode["history_length"])
    return spec


def n_input_planes_from_config(cfg: Mapping[str, Any]) -> int:
    from xiangqi_engine._xiangqi import n_input_planes

    return int(n_input_planes(spec_from_config(cfg)))


def resolve_device(cfg: Mapping[str, Any], override: str | None = None) -> str:
    name = override or cfg.get("device", "auto")
    if name != "auto":
        return str(name)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def deepcopy_config(cfg: Mapping[str, Any]) -> Cfg:
    return Cfg(copy.deepcopy(dict(cfg)))
