from pathlib import Path

import pytest

from xiangqi_engine import START_FEN, load_config
from xiangqi_engine.config import deepcopy_config
from xiangqi_engine.play.server import static_relpath
from xiangqi_engine.play.gomoku_session import GomokuPlaySession
from xiangqi_engine.play.session import PlaySession, resolved_checkpoint


def _session():
    cfg = deepcopy_config(load_config())
    cfg["mcts"]["simulations"] = 2
    s = PlaySession(cfg)
    s.new_game(red="human", black="human", simulations=2, checkpoint="")
    return s


def test_new_game_start_position():
    s = _session()
    st = s.state()
    assert st["fen"] == START_FEN
    assert st["side"] == "red"
    assert st["red"] == "human"
    assert st["black"] == "human"
    assert st["over"] is False
    assert st["history"] == []
    assert st["last_move"] is None
    assert "b2" in st["legal_from"]
    assert "e2" in st["legal_from"]["b2"]
    assert st["squares"][0][0]["rank"] == 9
    assert st["squares"][0][0]["file"] == 0
    assert st["squares"][-1][4]["glyph"] == "帅"


def test_state_shows_actual_checkpoint_not_config_default():
    cfg = deepcopy_config(load_config())
    cfg["mcts"]["simulations"] = 2
    cfg["play"]["checkpoint"] = "config-default.pt"
    cfg["play"]["simulations"] = 80
    s = PlaySession(cfg)
    s.new_game(red="human", black="human", simulations=4, checkpoint="used.pt")
    st = s.state()
    assert st["checkpoint"] == "used.pt"
    assert st["play_defaults"]["checkpoint"] == "used.pt"
    assert st["play_defaults"]["simulations"] == 4
    s.new_game(red="human", black="human", simulations=4, checkpoint="")
    assert s.state()["checkpoint"] == ""
    assert s.state()["play_defaults"]["checkpoint"] == ""


def test_idle_repetition_shows_chinese_reason():
    s = _session()
    for iccs in ["h2e2", "h7e7", "e2h2", "e7h7"] * 2:
        s.move(iccs)
    st = s.state()
    assert st["over"] is True
    assert st["outcome"] == "和棋"
    assert st["reason"] == "允许不变"


def test_legal_and_illegal_move():
    s = _session()
    st = s.move("b2e2")
    assert st["error"] == ""
    assert st["history"] == ["b2e2"]
    assert st["last_move"] == {"from": "b2", "to": "e2", "iccs": "b2e2"}
    assert st["side"] == "black"
    assert st["ply"] == 1
    bad = s.move("b2e2")
    assert bad["error"]
    assert bad["history"] == ["b2e2"]


def test_undo_and_human_turn():
    s = _session()
    s.new_game(red="human", black="ai", simulations=2, checkpoint="")
    s.move("b2e2")
    s.ai_move()
    assert len(s.history) == 2
    s.undo(1)
    assert len(s.history) == 1
    s.undo_human_turn()
    assert s.history == []
    assert s.state()["fen"] == START_FEN

    s.move("b2e2")
    s.ai_move()
    s.undo_human_turn()
    assert s.history == []


def test_resolved_checkpoint_absolutizes_existing_file(tmp_path):
    f = tmp_path / "best.pt"
    f.write_bytes(b"x")
    assert resolved_checkpoint(str(f)) == str(f.resolve())
    assert resolved_checkpoint("missing.pt") == "missing.pt"
    assert resolved_checkpoint("  ") == ""


def test_ai_move_uniform_and_refuse_human_side():
    s = _session()
    s.new_game(red="ai", black="human", simulations=2, checkpoint="")
    st = s.ai_move()
    assert st["error"] == ""
    assert len(st["history"]) == 1
    assert st["side"] == "black"
    refused = s.ai_move()
    assert refused["error"]
    assert len(refused["history"]) == 1


def test_one_tree_descends_on_move_and_retreats_on_undo():
    s = _session()
    s.new_game(red="human", black="ai", simulations=8, checkpoint="")
    assert s._search is not None
    start = s._search.root
    assert start.expanded
    visits_at_start = int(start.n.sum())
    assert visits_at_start == 8
    s.move("b2e2")
    assert s._search.root is not start
    assert len(s._search._ancestors) == 1
    s.ai_move()
    assert len(s._search._ancestors) == 2
    s.undo(2)
    assert s.history == []
    assert s._search.root is start
    assert int(s._search.root.n.sum()) == visits_at_start


def test_static_paths_stay_inside_bundle():
    assert static_relpath("/") == "index.html"
    assert static_relpath("/", "gomoku") == "gomoku.html"
    assert static_relpath("/static/style.css") == "style.css"
    assert static_relpath("/static/app.js") == "app.js"
    assert static_relpath("/static/result.css") == "result.css"
    assert static_relpath("/static/../session.py") is None
    assert static_relpath("/app.js") == "app.js"


def test_gomoku_session_place_and_undo():
    cfg = deepcopy_config(load_config("config/gomoku_smoke.json"))
    cfg["mcts"]["simulations"] = 2
    s = GomokuPlaySession(cfg)
    s.new_game(red="human", black="human", simulations=2, checkpoint="")
    st = s.state()
    assert st["game"] == "gomoku"
    assert st["size"] == 9
    assert st["side"] == "black"
    assert "e4" in st["legal"]
    s.move("e4")
    assert s.history == ["e4"]
    assert s.state()["side"] == "white"
    s.undo(1)
    assert s.history == []
    assert s.state()["fen"] == s.board.fen()
    assert s.state()["winning"] == []


def test_gomoku_state_includes_winning_line():
    cfg = deepcopy_config(load_config("config/gomoku_smoke.json"))
    cfg["mcts"]["simulations"] = 2
    s = GomokuPlaySession(cfg)
    s.new_game(red="human", black="human", simulations=2, checkpoint="")
    for iccs in ("a0", "a8", "b0", "b8", "c0", "c8", "d0", "d8", "e0"):
        s.move(iccs)
    st = s.state()
    assert st["over"] is True
    assert st["outcome"] == "黑胜"
    assert st["reason"] == "五连"
    assert st["winning"] == ["a0", "b0", "c0", "d0", "e0"]


def test_gomoku_state_shows_actual_checkpoint():
    cfg = deepcopy_config(load_config("config/gomoku_smoke.json"))
    cfg["mcts"]["simulations"] = 2
    cfg["play"]["checkpoint"] = "config-default.pt"
    cfg["play"]["simulations"] = 80
    s = GomokuPlaySession(cfg)
    s.new_game(red="human", black="human", simulations=4, checkpoint="used.pt")
    st = s.state()
    assert st["checkpoint"] == "used.pt"
    assert st["play_defaults"]["checkpoint"] == "used.pt"
    assert st["play_defaults"]["simulations"] == 4


def test_published_gomoku_best_loads():
    torch = pytest.importorskip("torch")
    from xiangqi_engine.loop import load_checkpoint
    from xiangqi_engine.network import PolicyValueNet

    path = Path("checkpoints/gomoku/best.pt")
    assert path.is_file()
    cfg = load_config("config/gomoku.json")
    assert cfg["play"]["checkpoint"] == "checkpoints/gomoku/best.pt"
    net = PolicyValueNet(cfg)
    iteration = load_checkpoint(path, net)
    assert iteration >= 1
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert (payload.get("config") or {}).get("game") == "gomoku"
