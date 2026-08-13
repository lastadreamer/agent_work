from xiangqi_engine import (
    ACTION_FROM_TO,
    BLACK,
    RED,
    Board,
    Encoder,
    Move,
    encode_board,
    flip_square,
    index_to_move,
    load_config,
    move_to_index,
    spec_from_config,
)
from xiangqi_engine.config import n_input_planes_from_config


def test_config_planes_match_tensor():
    cfg = load_config()
    b = Board()
    x = encode_board(b, cfg)
    assert x.shape == (n_input_planes_from_config(cfg), 10, 9)
    assert x.dtype.name == "float32"


def test_start_position_red_at_bottom():
    cfg = load_config()
    x = encode_board(Board(), cfg)
    # Plane 0 = our king. Red to move, no flip: king on e0.
    assert x[0, 0, 4] == 1.0
    assert x[0].sum() == 1.0
    # Plane 7 = opponent king on e9.
    assert x[7, 9, 4] == 1.0
    # Five red pawns on rank 3.
    assert x[6, 3].sum() == 5.0
    # Halfmove extra plane is 0 at start.
    assert x[14].max() == 0.0


def test_black_to_move_is_flipped():
    fen = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR b - - 0 1"
    b = Board(fen)
    cfg = load_config()
    spec = spec_from_config(cfg)
    assert spec.perspective_current_player
    enc = Encoder(cfg)
    assert enc.flip(b) is True
    x = encode_board(b, cfg)
    # After 180° rotation, Black's king (e9) sits on e0 as "our" king.
    assert x[0, 0, 4] == 1.0
    # Red's king appears as opponent on e9.
    assert x[7, 9, 4] == 1.0


def test_absolute_perspective_does_not_flip():
    cfg = load_config()
    cfg["encode"]["perspective"] = "absolute"
    fen = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR b - - 0 1"
    x = encode_board(Board(fen), cfg)
    # "Our" is still Black, but squares stay red-at-bottom.
    assert x[0, 9, 4] == 1.0
    assert x[7, 0, 4] == 1.0


def test_from_to_index_roundtrip():
    for from_sq in range(90):
        for to_sq in (0, 4, 44, 89):
            mv = Move(from_sq, to_sq)
            for flip in (False, True):
                idx = move_to_index(mv, flip)
                assert 0 <= idx < ACTION_FROM_TO
                back = index_to_move(idx, flip)
                assert back.from_sq == mv.from_sq
                assert back.to_sq == mv.to_sq


def test_flip_square_involution():
    assert flip_square(0) == 89
    assert flip_square(4) == 85
    assert flip_square(flip_square(19)) == 19


def test_legal_indices_are_unique_and_in_range():
    b = Board()
    enc = Encoder()
    idxs = enc.legal_action_indices(b)
    assert len(idxs) == 44
    assert len(set(idxs)) == 44
    assert all(0 <= i < ACTION_FROM_TO for i in idxs)
    # Cannon capture b2b9 is a known start-position capture.
    cap = Move.from_iccs("b2b9")
    assert enc.move_index(b, cap) in idxs
    assert enc.move_from_index(b, enc.move_index(b, cap)) == cap


def test_black_move_index_uses_flipped_coordinates():
    fen = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR b - - 0 1"
    b = Board(fen)
    enc = Encoder()
    # 180° rotation maps red's b-file to the i-file, so the Black mirror of
    # b2b9 is h7h0, not a same-file vertical flip.
    mv = Move.from_iccs("h7h0")
    idx = enc.move_index(b, mv)
    red_mirror = Move.from_iccs("b2b9")
    assert idx == move_to_index(red_mirror, False)
    assert enc.move_from_index(b, idx) == mv


def test_history_window_keeps_current_player_as_us():
    cfg = load_config()
    cfg["encode"]["history_length"] = 2
    enc = Encoder(cfg)
    start = Board()
    enc.observe(start)
    moved = start.copy()
    moved.push(Move.from_iccs("b2e2"))
    x = enc.tensor(moved)
    assert x.shape[0] == n_input_planes_from_config(cfg)
    # Current player is Black; newest frame is the second 14-plane block.
    # Our king (Black) after flip is on e0 in the newest frame.
    newest = x[14:28]
    assert newest[0, 0, 4] == 1.0
    # Oldest frame is the start position, still from Black's point of view.
    oldest = x[0:14]
    assert oldest[0, 0, 4] == 1.0


def test_policy_target_uniform_over_legal():
    b = Board()
    enc = Encoder()
    target = enc.policy_target(b)
    assert len(target) == ACTION_FROM_TO
    legal = set(enc.legal_action_indices(b))
    assert abs(sum(target) - 1.0) < 1e-6
    for i, p in enumerate(target):
        if i in legal:
            assert abs(p - 1.0 / 44) < 1e-6
        else:
            assert p == 0.0


def test_doc_keys_stripped():
    cfg = load_config()
    assert "_doc" not in cfg
    assert "_doc" not in cfg["encode"]
