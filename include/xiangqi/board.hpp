#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

namespace xiangqi {

// 10 ranks x 9 files = 90 squares. Index = rank * 9 + file.
// Rank 0 is red's back rank, rank 9 is black's. File 0 is the left file from red.
using Square = uint8_t;
using Piece = uint8_t;
using Color = uint8_t;

constexpr int N_SQUARES = 90;
constexpr int N_FILES = 9;
constexpr int N_RANKS = 10;
constexpr int MAX_PIECES = 16;
constexpr int MAX_MOVES = 256;
constexpr int MAX_PLY = 512;

constexpr Color RED = 0;
constexpr Color BLACK = 1;

constexpr Piece EMPTY = 0;
constexpr Piece R_KING = 1, R_ADVISOR = 2, R_ELEPHANT = 3, R_HORSE = 4,
                R_CHARIOT = 5, R_CANNON = 6, R_PAWN = 7;
constexpr Piece B_KING = 8, B_ADVISOR = 9, B_ELEPHANT = 10, B_HORSE = 11,
                B_CHARIOT = 12, B_CANNON = 13, B_PAWN = 14;

enum class PieceType : uint8_t {
    KING = 0,
    ADVISOR,
    ELEPHANT,
    HORSE,
    CHARIOT,
    CANNON,
    PAWN
};

enum class Outcome : uint8_t {
    ONGOING = 0,
    RED_WIN,     // side-to-move has no legal move and it is black to move, or vice versa
    BLACK_WIN,
    DRAW
};

enum class TerminalReason : uint8_t {
    NONE = 0,
    CHECKMATE,   // no legal move while in check
    STALEMATE,   // 困毙: no legal move, not in check — a loss in Xiangqi
    REPETITION,        // 允许不变: 三次重复且双方都不是长将/长捉
    NO_PROGRESS,       // 60 full moves without capture or pawn move
    MAX_PLY,
    PERPETUAL_CHECK,   // 长将: 循环中一方每步都是将军
    PERPETUAL_CHASE    // 长捉 / 一将一捉: 循环中一方每步都是将或捉
};

constexpr Square NO_SQUARE = 255;

// ICCS: files a-i, ranks 0-9. Start FEN uses CPW letters (H horse, E elephant).
constexpr const char* START_FEN =
    "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR w - - 0 1";

inline int file_of(Square sq) { return sq % N_FILES; }
inline int rank_of(Square sq) { return sq / N_FILES; }
inline Square make_square(int rank, int file) {
    return static_cast<Square>(rank * N_FILES + file);
}
inline bool in_board(int rank, int file) {
    return rank >= 0 && rank < N_RANKS && file >= 0 && file < N_FILES;
}

inline Color color_of(Piece p) { return p >= B_KING ? BLACK : RED; }
inline PieceType type_of(Piece p) {
    return static_cast<PieceType>(p >= B_KING ? (p - B_KING) : (p - R_KING));
}
inline Piece make_piece(Color c, PieceType t) {
    return static_cast<Piece>((c == RED ? R_KING : B_KING) + static_cast<uint8_t>(t));
}

inline bool in_palace(Square sq, Color c) {
    const int f = file_of(sq);
    const int r = rank_of(sq);
    if (f < 3 || f > 5) return false;
    return c == RED ? (r <= 2) : (r >= 7);
}

inline bool elephant_on_own_side(Square sq, Color c) {
    const int r = rank_of(sq);
    return c == RED ? (r <= 4) : (r >= 5);
}

inline bool pawn_has_crossed_river(Square sq, Color c) {
    const int r = rank_of(sq);
    return c == RED ? (r >= 5) : (r <= 4);
}

// 16-bit move: from in bits 0-6, to in bits 7-13.
struct Move {
    uint16_t data = 0;

    Move() = default;
    constexpr Move(Square from, Square to)
        : data(static_cast<uint16_t>(from | (static_cast<uint16_t>(to) << 7))) {}

    constexpr Square from() const { return static_cast<Square>(data & 127); }
    constexpr Square to() const { return static_cast<Square>((data >> 7) & 127); }

    constexpr bool operator==(Move o) const { return data == o.data; }
    constexpr bool operator!=(Move o) const { return data != o.data; }

    std::string iccs() const;
    static Move from_iccs(const std::string& s);
};

struct Undo {
    uint64_t hash;
    Move move;
    Piece captured;
    uint16_t halfmove;
    uint16_t fullmove;
};

struct MoveList {
    Move moves[MAX_MOVES];
    int size = 0;

    void add(Square from, Square to) {
        moves[size++] = Move(from, to);
    }
    const Move* begin() const { return moves; }
    const Move* end() const { return moves + size; }
};

struct Terminal {
    Outcome outcome = Outcome::ONGOING;
    TerminalReason reason = TerminalReason::NONE;
};

// Trivially copyable: memcpy a Board to fork a self-play / MCTS worker.
// No shared mutable state. Each thread or process must own its own Board.
struct Board {
    Piece squares[N_SQUARES]{};
    Square piece_list[2][MAX_PIECES]{};
    uint8_t n_pieces[2]{};
    uint8_t list_index[N_SQUARES]{};
    Square king_sq[2]{NO_SQUARE, NO_SQUARE};
    Color side = RED;
    uint16_t halfmove = 0;
    uint16_t fullmove = 1;
    uint64_t hash = 0;
    int ply = 0;
    uint64_t hist[MAX_PLY]{};
    Undo undos[MAX_PLY]{};

    Board();
    explicit Board(const std::string& fen);

    void clear();
    void set_fen(const std::string& fen);
    std::string fen() const;
    std::string to_string() const;

    void generate_pseudo_legal(MoveList& out) const;
    int generate_legal(MoveList& out);
    bool has_legal_move();
    bool is_legal(Move m);

    void make_move(Move m);
    void unmake_move();
    Move last_move() const;

    bool in_check() const;
    bool is_attacked(Square sq, Color by) const;
    bool attacks_from(Square from, Square to) const;
    bool is_chasing(Color us) const;

    Terminal terminal();  // uses legal-move generation (non-const)
    Terminal adjudicate_repetition();
    Color side_to_move() const { return side; }
    Piece piece_at(Square sq) const { return squares[sq]; }
    bool is_capture(Move m) const { return squares[m.to()] != EMPTY; }

    uint64_t perft(int depth);
    uint64_t compute_hash() const;
    int repetition_count() const;

    static void init_zobrist();
};

static_assert(std::is_trivially_copyable<Move>::value, "Move must be POD");
static_assert(std::is_trivially_copyable<Board>::value,
              "Board must be memcpy-copyable for MCTS forks and multiprocessing");

inline bool boards_equal_position(const Board& a, const Board& b) {
    if (a.side != b.side) return false;
    for (int i = 0; i < N_SQUARES; ++i) {
        if (a.squares[i] != b.squares[i]) return false;
    }
    return true;
}

std::string square_to_iccs(Square sq);
Square iccs_to_square(const std::string& s);
char piece_to_fen_char(Piece p);
Piece fen_char_to_piece(char c);

}  // namespace xiangqi
