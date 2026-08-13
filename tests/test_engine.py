# Xiangqi engine — Phase 1 tests.
# Perft numbers: https://chessprogramming.org/Chinese_Chess_Perft_Results

import copy
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import pytest

from xiangqi_engine import (
    BLACK,
    BOARD_NBYTES,
    RED,
    START_FEN,
    Board,
    Move,
    Outcome,
    TerminalReason,
    play_random_games,
    square_to_iccs,
)

CPW_PERFT = [
    (START_FEN, {1: 44, 2: 1920, 3: 79666, 4: 3290240}),
    (
        "r1ea1a3/4kh3/2h1e4/pHp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2EAKAE2 w - - 0 1",
        {1: 38, 2: 1128, 3: 43929},
    ),
    (
        "1ceak4/9/h2a5/2p1p3p/5cp2/2h2H3/6PCP/3AE4/2C6/3A1K1H1 w - - 0 1",
        {1: 7, 2: 281, 3: 8620},
    ),
    (
        "5a3/3k5/3aR4/9/5r3/5h3/9/3A1A3/5K3/2EC2E2 w - - 0 1",
        {1: 25, 2: 424, 3: 9850},
    ),
    (
        "CRH1k1e2/3ca4/4ea3/9/2hr5/9/9/4E4/4A4/4KA3 w - - 0 1",
        {1: 28, 2: 516, 3: 14808},
    ),
    (
        "R1H1k1e2/9/3aea3/9/2hr5/2E6/9/4E4/4A4/4KA3 w - - 0 1",
        {1: 21, 2: 364, 3: 7626},
    ),
    (
        "C1hHk4/9/9/9/9/9/h1pp5/E3C4/9/3A1K3 w - - 0 1",
        {1: 28, 2: 222, 3: 6241},
    ),
    (
        "4ka3/4a4/9/9/4H4/p8/9/4C3c/7h1/2EK5 w - - 0 1",
        {1: 23, 2: 345, 3: 8124},
    ),
    (
        "2e1ka3/9/e3H4/4h4/9/9/9/4C4/2p6/2EK5 w - - 0 1",
        {1: 21, 2: 195, 3: 3883},
    ),
    (
        "1C2ka3/9/C1Hae1h2/p3p3p/6p2/9/P3P3P/3AE4/3p2c2/c1EAK4 w - - 0 1",
        {1: 30, 2: 830, 3: 22787},
    ),
    (
        "ChH1k1e2/c3a4/4ea3/9/2hr5/9/9/4C4/4A4/4KA3 w - - 0 1",
        {1: 19, 2: 583, 3: 11714},
    ),
]


def test_start_fen_roundtrip():
    b = Board()
    assert b.fen().startswith("rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w")
    assert b.side_to_move() == RED
    assert len(b.legal_moves()) == 44


def test_ucci_letters_parse():
    # Fairy-Stockfish / UCCI uses n=horse, b=elephant.
    ucci = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    b = Board(ucci)
    assert len(b.legal_moves()) == 44


@pytest.mark.parametrize("fen,depths", CPW_PERFT)
def test_cpw_perft(fen, depths):
    b = Board(fen)
    for depth, nodes in depths.items():
        got = b.perft(depth)
        assert got == nodes, f"perft({depth})={got} expected {nodes} fen={fen}"


def test_make_unmake_restores_hash_and_fen():
    b = Board()
    root = b.fen()
    root_hash = b.hash()
    for mv in b.legal_moves():
        b.push(mv)
        assert b.hash() == b.compute_hash()
        b.pop()
        assert b.fen() == root
        assert b.hash() == root_hash
        assert b.ply() == 0


def test_horse_hobble():
    # Red horse at b0; the right-and-up jump is blocked by the elephant on c0.
    b = Board()
    horse = Move.from_iccs("b0d1")  # would need empty c0
    assert horse not in b.legal_moves()
    assert Move.from_iccs("b0a2") in b.legal_moves()
    assert Move.from_iccs("b0c2") in b.legal_moves()


def test_elephant_cannot_cross_river():
    # Place a red elephant on the river bank eyeing a square across the river.
    b = Board("3k5/9/9/9/9/2E6/9/9/9/5K3 w - - 0 1")
    dests = {m.iccs()[2:] for m in b.legal_moves() if m.iccs().startswith("c4")}
    assert "a6" not in dests
    assert "e6" not in dests
    assert "a2" in dests
    assert "e2" in dests


def test_cannon_needs_screen_to_capture():
    b = Board()
    # Red cannon b2 captures black horse b9, jumping the black cannon on b7.
    assert Move.from_iccs("b2b9") in b.legal_moves()
    assert b.is_capture(Move.from_iccs("b2b9"))


def test_pawn_sideways_only_after_river():
    uncrossed = Board("4k4/9/9/9/9/9/4P4/9/9/4K4 w - - 0 1")
    dests = {m.iccs()[2:] for m in uncrossed.legal_moves() if m.iccs().startswith("e3")}
    assert dests == {"e4"}

    crossed = Board("3k5/9/9/9/4P4/9/9/9/9/5K3 w - - 0 1")
    dests = {m.iccs()[2:] for m in crossed.legal_moves() if m.iccs().startswith("e5")}
    assert dests == {"e6", "d5", "f5"}


def test_flying_general_is_illegal():
    # Kings on the same file with an empty file between: stepping off the file is
    # required; staying on file e remains illegal.
    b = Board("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1")
    assert b.in_check()
    dests = {m.iccs()[2:] for m in b.legal_moves() if m.from_sq == b.king_square(RED)}
    assert "e1" not in dests
    assert "d0" in dests
    assert "f0" in dests


def test_must_escape_check():
    # Rook on a9 checks along the back rank. A red pawn on e5 blocks flying
    # general so the king may step to e8.
    b = Board("R3k4/9/9/9/4P4/9/9/9/9/4K4 b - - 0 1")
    assert b.in_check()
    moves = b.legal_moves()
    assert Move.from_iccs("e9e8") in moves
    for mv in moves:
        b.push(mv)
        assert not b.is_attacked(b.king_square(BLACK), RED)
        b.pop()


def test_checkmate_is_a_loss():
    # Double attack: rook on a9 + pawn on e8 protected by rook on e7.
    b = Board("R3k4/4P4/4R4/9/9/9/9/9/9/4K4 b - - 0 1")
    assert b.in_check()
    assert b.legal_moves() == []
    term = b.terminal()
    assert term.outcome == Outcome.RED_WIN
    assert term.reason == TerminalReason.CHECKMATE


def test_stalemate_is_a_loss():
    # 困毙: black king has no square, is not in check. See README.
    b = Board("4k4/R8/4H4/9/9/9/9/9/9/4K4 b - - 0 1")
    assert not b.in_check()
    assert b.legal_moves() == []
    term = b.terminal()
    assert term.outcome == Outcome.RED_WIN
    assert term.reason == TerminalReason.STALEMATE


def test_threefold_draw():
    b = Board()
    cycle = ["h2e2", "h7e7", "e2h2", "e7h7"]
    # Start position counts as 1. After two full cycles the start hash appears 3 times.
    # Idle cannon shuttle: 允许不变, still a draw.
    for _ in range(2):
        for iccs in cycle:
            assert b.terminal().outcome == Outcome.ONGOING
            b.push_iccs(iccs)
    term = b.terminal()
    assert term.reason == TerminalReason.REPETITION
    assert term.outcome == Outcome.DRAW


def test_rook_chases_unprotected_horse():
    # Red rook on b8 attacks black horse on b9; horse has no protector.
    b = Board("1n6k/1R7/9/9/9/9/9/9/9/4K4 b - - 0 1")
    assert b.is_chasing(RED)
    assert not b.is_chasing(BLACK)


def test_protected_piece_is_not_chase():
    # Same rook-vs-horse, but a black chariot on a9 protects the horse.
    b = Board("rn6k/1R7/9/9/9/9/9/9/9/4K4 b - - 0 1")
    assert not b.is_chasing(RED)


def test_pawn_only_attack_is_not_chase():
    # Crossed red pawn on e5 attacks an unprotected black rook on e6.
    # 兵允许长捉: pawn-only attacks do not count as 捉.
    b = Board("4k4/9/9/4r4/4P4/9/9/9/9/4K4 w - - 0 1")
    assert not b.is_chasing(RED)


def test_perpetual_check_loses():
    # Rook on the d-file checks the black king. Red king sits on f0 so
    # d9-e9 is not a flying-general suicide. Cycle is forced 长将 by red.
    b = Board("3k5/9/9/9/9/9/9/9/3R5/5K3 b - - 0 1")
    assert b.in_check()
    cycle = ["d9e9", "d1e1", "e9d9", "e1d1"]
    for _ in range(2):
        for iccs in cycle:
            assert b.terminal().outcome == Outcome.ONGOING
            b.push_iccs(iccs)
    fen_after = b.fen()
    term = b.terminal()
    assert term.reason == TerminalReason.PERPETUAL_CHECK
    assert term.outcome == Outcome.BLACK_WIN
    assert b.fen() == fen_after
    b.pop()
    assert b.terminal().outcome == Outcome.ONGOING


def test_perpetual_chase_loses():
    # Red rook chases an unprotected horse: b9-d8 / b8-d7 and back.
    b = Board("1n6k/1R7/9/9/9/9/9/9/9/4K4 b - - 0 1")
    cycle = ["b9d8", "b8d7", "d8b9", "d7b8"]
    for _ in range(2):
        for iccs in cycle:
            assert b.terminal().outcome == Outcome.ONGOING
            b.push_iccs(iccs)
    fen_after = b.fen()
    term = b.terminal()
    assert term.reason == TerminalReason.PERPETUAL_CHASE
    assert term.outcome == Outcome.BLACK_WIN
    assert b.fen() == fen_after
    b.pop()
    assert b.terminal().outcome == Outcome.ONGOING


def test_copy_is_independent():
    a = Board()
    b = a.copy()
    b.push(b.legal_moves()[0])
    assert a.fen() != b.fen()
    assert a.ply() == 0
    c = copy.deepcopy(a)
    assert c.fen() == a.fen()


def test_pickle_roundtrip():
    b = Board()
    b.push(b.legal_moves()[0])
    b2 = pickle.loads(pickle.dumps(b))
    assert b2.fen() == b.fen()
    assert b2.hash() == b.hash()
    assert b2.legal_moves() == b.legal_moves()
    assert BOARD_NBYTES > 0


def _play_one(seed: int) -> int:
    r = play_random_games(2, 200, seed)
    return r.red_wins + r.black_wins + r.draws


def test_thread_pool_independent_boards():
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(play_random_games, [8] * 4, [120] * 4, range(1, 5)))
    total_games = sum(r.red_wins + r.black_wins + r.draws for r in results)
    assert total_games == 32


def test_process_pool_pickle_and_play():
    with ProcessPoolExecutor(max_workers=4) as pool:
        done = list(pool.map(_play_one, range(8)))
    assert all(x == 2 for x in done)


def test_iccs_helpers():
    assert square_to_iccs(0) == "a0"
    assert Move.from_iccs("a0a1").iccs() == "a0a1"


def test_random_play_terminates():
    r = play_random_games(20, 400, seed=42)
    assert r.red_wins + r.black_wins + r.draws == 20
    assert r.plies > 0
