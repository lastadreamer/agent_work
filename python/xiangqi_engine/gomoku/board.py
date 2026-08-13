"""Free-style Gomoku: five in a row wins, no Renju forbidden moves."""

from __future__ import annotations

from dataclasses import dataclass

from xiangqi_engine._xiangqi import BLACK, RED, Outcome

EMPTY = 0
STONE_BLACK = 1
STONE_WHITE = 2


def stone_of(side: int) -> int:
    return STONE_BLACK if side == BLACK else STONE_WHITE


def sq_to_iccs(sq: int, size: int) -> str:
    return f"{chr(ord('a') + (sq % size))}{sq // size}"


def iccs_to_sq(text: str, size: int) -> int:
    if len(text) < 2:
        raise ValueError(f"bad square: {text}")
    file = ord(text[0]) - ord("a")
    rank = int(text[1:])
    if file < 0 or file >= size or rank < 0 or rank >= size:
        raise ValueError(f"bad square: {text}")
    return rank * size + file


@dataclass(frozen=True)
class GomokuMove:
    sq: int
    size: int

    def iccs(self) -> str:
        return sq_to_iccs(self.sq, self.size)

    @staticmethod
    def from_iccs(text: str, size: int) -> GomokuMove:
        return GomokuMove(iccs_to_sq(text, size), size)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GomokuMove) and self.sq == other.sq and self.size == other.size


@dataclass
class GomokuTerminal:
    outcome: Outcome = Outcome.ONGOING
    reason: str = ""


class GomokuBoard:
    """Square board. Black (first) uses STONE_BLACK; white uses STONE_WHITE / RED."""

    def __init__(self, size: int = 15):
        if size < 5:
            raise ValueError("gomoku size must be >= 5")
        self.size = int(size)
        self.n_squares = self.size * self.size
        self.squares = [EMPTY] * self.n_squares
        self.side = BLACK
        self._ply = 0
        self.undos: list[int] = []

    def ply(self) -> int:
        return self._ply

    def copy(self) -> GomokuBoard:
        b = GomokuBoard(self.size)
        b.squares = self.squares[:]
        b.side = self.side
        b._ply = self._ply
        b.undos = self.undos[:]
        return b

    def hash(self) -> int:
        return hash((self.side, tuple(self.squares)))

    def fen(self) -> str:
        rows = []
        for rank in range(self.size - 1, -1, -1):
            chunk = []
            empty = 0
            for file in range(self.size):
                p = self.squares[rank * self.size + file]
                if p == EMPTY:
                    empty += 1
                    continue
                if empty:
                    chunk.append(str(empty))
                    empty = 0
                chunk.append("x" if p == STONE_BLACK else "o")
            if empty:
                chunk.append(str(empty))
            rows.append("".join(chunk))
        stm = "b" if self.side == BLACK else "w"
        return "/".join(rows) + f" {stm} {self._ply}"

    def side_to_move(self) -> int:
        return self.side

    def piece_at(self, sq: int) -> int:
        return self.squares[sq]

    def last_move(self) -> GomokuMove:
        if not self.undos:
            raise RuntimeError("no move to unmake")
        return GomokuMove(self.undos[-1], self.size)

    def is_legal(self, move: GomokuMove) -> bool:
        if move.size != self.size:
            return False
        if move.sq < 0 or move.sq >= self.n_squares:
            return False
        if self.squares[move.sq] != EMPTY:
            return False
        return self.terminal().outcome == Outcome.ONGOING

    def legal_moves(self) -> list[GomokuMove]:
        if self.terminal().outcome != Outcome.ONGOING:
            return []
        return [GomokuMove(i, self.size) for i, p in enumerate(self.squares) if p == EMPTY]

    def make_move(self, move: GomokuMove) -> None:
        if not self.is_legal(move):
            raise ValueError(f"illegal gomoku move {move.iccs()}")
        self.squares[move.sq] = stone_of(self.side)
        self.undos.append(move.sq)
        self.side ^= 1
        self._ply += 1

    def unmake_move(self) -> None:
        if not self.undos:
            raise RuntimeError("no move to unmake")
        sq = self.undos.pop()
        self.squares[sq] = EMPTY
        self.side ^= 1
        self._ply -= 1

    def push(self, move: GomokuMove) -> None:
        self.make_move(move)

    def push_iccs(self, text: str) -> None:
        self.make_move(GomokuMove.from_iccs(text, self.size))

    def in_check(self) -> bool:
        return False

    def halfmove_clock(self) -> int:
        return 0

    def _line_len(self, sq: int, dr: int, dc: int, stone: int) -> int:
        size = self.size
        rank, file = divmod(sq, size)
        n = 0
        r, c = rank + dr, file + dc
        while 0 <= r < size and 0 <= c < size and self.squares[r * size + c] == stone:
            n += 1
            r += dr
            c += dc
        return n

    def _is_five(self, sq: int) -> bool:
        stone = self.squares[sq]
        if stone == EMPTY:
            return False
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if 1 + self._line_len(sq, dr, dc, stone) + self._line_len(sq, -dr, -dc, stone) >= 5:
                return True
        return False

    def terminal(self) -> GomokuTerminal:
        if self.undos:
            sq = self.undos[-1]
            if self._is_five(sq):
                mover = self.side ^ 1
                outcome = Outcome.BLACK_WIN if mover == BLACK else Outcome.RED_WIN
                return GomokuTerminal(outcome, "FIVE")
        if self._ply >= self.n_squares or all(p != EMPTY for p in self.squares):
            return GomokuTerminal(Outcome.DRAW, "FULL")
        return GomokuTerminal(Outcome.ONGOING, "")
