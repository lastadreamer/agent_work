from xiangqi_engine import START_FEN, load_config
from xiangqi_engine.config import deepcopy_config
from xiangqi_engine.play.server import static_relpath
from xiangqi_engine.play.session import PlaySession


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
    assert static_relpath("/static/style.css") == "style.css"
    assert static_relpath("/static/app.js") == "app.js"
    assert static_relpath("/static/../session.py") is None
    assert static_relpath("/app.js") == "app.js"
