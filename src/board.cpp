#include "xiangqi/board.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>

namespace xiangqi {
namespace {

uint64_t Z_PIECE[15][N_SQUARES];
uint64_t Z_SIDE;
std::once_flag zobrist_once;

void init_zobrist_tables() {
    // Fixed seed so every process/thread derives the same keys.
    std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
    for (int p = 1; p <= 14; ++p) {
        for (int sq = 0; sq < N_SQUARES; ++sq) {
            Z_PIECE[p][sq] = rng();
        }
    }
    Z_SIDE = rng();
}

constexpr int ORTHO_DF[4] = {1, -1, 0, 0};
constexpr int ORTHO_DR[4] = {0, 0, 1, -1};

constexpr int DIAG_DF[4] = {1, 1, -1, -1};
constexpr int DIAG_DR[4] = {1, -1, 1, -1};

// Horse: (file, rank) jump, hobble is one step in the longer direction.
constexpr int HORSE_DF[8] = {1, -1, 1, -1, 2, 2, -2, -2};
constexpr int HORSE_DR[8] = {2, 2, -2, -2, 1, -1, 1, -1};
constexpr int HORSE_HF[8] = {0, 0, 0, 0, 1, 1, -1, -1};
constexpr int HORSE_HR[8] = {1, 1, -1, -1, 0, 0, 0, 0};

void add_if_capturable(MoveList& out, const Board& b, Square from, int rank, int file,
                       Color us) {
    if (!in_board(rank, file)) return;
    const Square to = make_square(rank, file);
    const Piece cap = b.squares[to];
    if (cap == EMPTY || color_of(cap) != us) {
        out.add(from, to);
    }
}

}  // namespace

void Board::init_zobrist() { std::call_once(zobrist_once, init_zobrist_tables); }

std::string square_to_iccs(Square sq) {
    std::string s(2, '\0');
    s[0] = static_cast<char>('a' + file_of(sq));
    s[1] = static_cast<char>('0' + rank_of(sq));
    return s;
}

Square iccs_to_square(const std::string& s) {
    if (s.size() < 2) throw std::invalid_argument("bad square: " + s);
    const int file = s[0] - 'a';
    const int rank = s[1] - '0';
    if (!in_board(rank, file)) throw std::invalid_argument("bad square: " + s);
    return make_square(rank, file);
}

std::string Move::iccs() const { return square_to_iccs(from()) + square_to_iccs(to()); }

Move Move::from_iccs(const std::string& s) {
    if (s.size() < 4) throw std::invalid_argument("bad move: " + s);
    return Move(iccs_to_square(s.substr(0, 2)), iccs_to_square(s.substr(2, 2)));
}

char piece_to_fen_char(Piece p) {
    static constexpr char kChars[] = " KAEHRCPKAEHRCP";
    if (p == EMPTY || p > B_PAWN) return '.';
    char c = kChars[p];
    if (p >= B_KING) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return c;
}

Piece fen_char_to_piece(char c) {
    const bool black = std::islower(static_cast<unsigned char>(c)) != 0;
    switch (std::tolower(static_cast<unsigned char>(c))) {
        case 'k':
            return black ? B_KING : R_KING;
        case 'a':
            return black ? B_ADVISOR : R_ADVISOR;
        case 'e':
        case 'b':  // UCCI / Fairy-Stockfish elephant
            return black ? B_ELEPHANT : R_ELEPHANT;
        case 'h':
        case 'n':  // UCCI horse
            return black ? B_HORSE : R_HORSE;
        case 'r':
            return black ? B_CHARIOT : R_CHARIOT;
        case 'c':
            return black ? B_CANNON : R_CANNON;
        case 'p':
            return black ? B_PAWN : R_PAWN;
        default:
            throw std::invalid_argument(std::string("bad piece char: ") + c);
    }
}

namespace {

const char* piece_glyph(Piece p) {
    switch (p) {
        case R_KING:
            return "帅";
        case R_ADVISOR:
            return "仕";
        case R_ELEPHANT:
            return "相";
        case R_HORSE:
            return "马";
        case R_CHARIOT:
            return "车";
        case R_CANNON:
            return "炮";
        case R_PAWN:
            return "兵";
        case B_KING:
            return "将";
        case B_ADVISOR:
            return "士";
        case B_ELEPHANT:
            return "象";
        case B_HORSE:
            return "馬";
        case B_CHARIOT:
            return "車";
        case B_CANNON:
            return "砲";
        case B_PAWN:
            return "卒";
        default:
            return "·";
    }
}

}  // namespace

Board::Board() { set_fen(START_FEN); }

Board::Board(const std::string& fen) { set_fen(fen); }

void Board::clear() {
    std::fill(std::begin(squares), std::end(squares), EMPTY);
    n_pieces[0] = n_pieces[1] = 0;
    std::fill(std::begin(list_index), std::end(list_index), static_cast<uint8_t>(0xFF));
    king_sq[0] = king_sq[1] = NO_SQUARE;
    side = RED;
    halfmove = 0;
    fullmove = 1;
    hash = 0;
    ply = 0;
}

void Board::set_fen(const std::string& fen) {
    init_zobrist();
    clear();

    std::istringstream ss(fen);
    std::string board_part, stm, castle, ep, half, full;
    if (!(ss >> board_part >> stm)) {
        throw std::invalid_argument("FEN missing board or side to move");
    }
    ss >> castle >> ep >> half >> full;  // optional

    int rank = 9;
    int file = 0;
    for (char c : board_part) {
        if (c == '/') {
            if (file != N_FILES) throw std::invalid_argument("FEN rank length");
            --rank;
            file = 0;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            file += c - '0';
            if (file > N_FILES) throw std::invalid_argument("FEN too many empties");
            continue;
        }
        if (!in_board(rank, file)) throw std::invalid_argument("FEN overflow");
        const Piece p = fen_char_to_piece(c);
        const Square sq = make_square(rank, file);
        squares[sq] = p;
        const Color ccol = color_of(p);
        const uint8_t idx = n_pieces[ccol]++;
        if (idx >= MAX_PIECES) throw std::invalid_argument("too many pieces");
        piece_list[ccol][idx] = sq;
        list_index[sq] = idx;
        if (type_of(p) == PieceType::KING) king_sq[ccol] = sq;
        hash ^= Z_PIECE[p][sq];
        ++file;
    }
    if (rank != 0 || file != N_FILES) {
        throw std::invalid_argument("FEN must contain 10 ranks of 9 files");
    }

    if (stm == "w" || stm == "r" || stm == "R" || stm == "W") {
        side = RED;
    } else if (stm == "b" || stm == "B") {
        side = BLACK;
        hash ^= Z_SIDE;
    } else {
        throw std::invalid_argument("FEN bad side to move");
    }

    if (!half.empty()) halfmove = static_cast<uint16_t>(std::stoi(half));
    if (!full.empty()) fullmove = static_cast<uint16_t>(std::stoi(full));
    ply = 0;
    hist[0] = hash;
}

std::string Board::fen() const {
    std::string out;
    out.reserve(80);
    for (int rank = 9; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < N_FILES; ++file) {
            const Piece p = squares[make_square(rank, file)];
            if (p == EMPTY) {
                ++empty;
            } else {
                if (empty) {
                    out.push_back(static_cast<char>('0' + empty));
                    empty = 0;
                }
                out.push_back(piece_to_fen_char(p));
            }
        }
        if (empty) out.push_back(static_cast<char>('0' + empty));
        if (rank) out.push_back('/');
    }
    out.push_back(' ');
    out.push_back(side == RED ? 'w' : 'b');
    out += " - - ";
    out += std::to_string(halfmove);
    out.push_back(' ');
    out += std::to_string(fullmove);
    return out;
}

std::string Board::to_string() const {
    std::string out;
    out += "  a b c d e f g h i\n";
    for (int rank = 9; rank >= 0; --rank) {
        out.push_back(static_cast<char>('0' + rank));
        out.push_back(' ');
        for (int file = 0; file < N_FILES; ++file) {
            out += piece_glyph(squares[make_square(rank, file)]);
            out.push_back(' ');
        }
        out.push_back('\n');
    }
    out += (side == RED ? "red to move\n" : "black to move\n");
    return out;
}

uint64_t Board::compute_hash() const {
    uint64_t h = 0;
    for (int sq = 0; sq < N_SQUARES; ++sq) {
        const Piece p = squares[sq];
        if (p) h ^= Z_PIECE[p][sq];
    }
    if (side == BLACK) h ^= Z_SIDE;
    return h;
}

int Board::repetition_count() const {
    int n = 0;
    // Count appearances of the current hash, including now.
    // Step by 1: Xiangqi side-to-move is in the hash, so same-side repeats
    // are naturally every even distance; scanning all is still cheap.
    for (int i = ply; i >= 0; --i) {
        if (hist[i] == hash) ++n;
    }
    return n;
}

bool Board::is_attacked(Square sq, Color by) const {
    for (int i = 0; i < n_pieces[by]; ++i) {
        const Square from = piece_list[by][i];
        const Piece p = squares[from];
        const int ff = file_of(from);
        const int rf = rank_of(from);
        const int ft = file_of(sq);
        const int rt = rank_of(sq);
        const int df = ft - ff;
        const int dr = rt - rf;

        switch (type_of(p)) {
            case PieceType::KING: {
                if (df == 0 && dr != 0) {
                    // Flying general: same file, empty path, target is the other king.
                    const int step = dr > 0 ? 1 : -1;
                    bool blocked = false;
                    for (int r = rf + step; r != rt; r += step) {
                        if (squares[make_square(r, ff)] != EMPTY) {
                            blocked = true;
                            break;
                        }
                    }
                    if (!blocked && type_of(squares[sq]) == PieceType::KING) return true;
                }
                if ((std::abs(df) + std::abs(dr)) == 1 && in_palace(from, by) &&
                    in_palace(sq, by)) {
                    return true;
                }
                break;
            }
            case PieceType::ADVISOR:
                if (std::abs(df) == 1 && std::abs(dr) == 1 && in_palace(from, by) &&
                    in_palace(sq, by)) {
                    return true;
                }
                break;
            case PieceType::ELEPHANT:
                if (std::abs(df) == 2 && std::abs(dr) == 2 &&
                    elephant_on_own_side(sq, by) &&
                    squares[make_square(rf + dr / 2, ff + df / 2)] == EMPTY) {
                    return true;
                }
                break;
            case PieceType::HORSE: {
                const int adf = std::abs(df);
                const int adr = std::abs(dr);
                if ((adf == 1 && adr == 2) || (adf == 2 && adr == 1)) {
                    const int hf = (adf == 2) ? (df > 0 ? 1 : -1) : 0;
                    const int hr = (adr == 2) ? (dr > 0 ? 1 : -1) : 0;
                    if (squares[make_square(rf + hr, ff + hf)] == EMPTY) return true;
                }
                break;
            }
            case PieceType::CHARIOT: {
                if (df != 0 && dr != 0) break;
                if (df == 0 && dr == 0) break;
                const int sf = (df == 0) ? 0 : (df > 0 ? 1 : -1);
                const int sr = (dr == 0) ? 0 : (dr > 0 ? 1 : -1);
                int f = ff + sf;
                int r = rf + sr;
                bool blocked = false;
                while (f != ft || r != rt) {
                    if (squares[make_square(r, f)] != EMPTY) {
                        blocked = true;
                        break;
                    }
                    f += sf;
                    r += sr;
                }
                if (!blocked) return true;
                break;
            }
            case PieceType::CANNON: {
                if (df != 0 && dr != 0) break;
                if (df == 0 && dr == 0) break;
                const int sf = (df == 0) ? 0 : (df > 0 ? 1 : -1);
                const int sr = (dr == 0) ? 0 : (dr > 0 ? 1 : -1);
                int screens = 0;
                int f = ff + sf;
                int r = rf + sr;
                while (f != ft || r != rt) {
                    if (squares[make_square(r, f)] != EMPTY) ++screens;
                    f += sf;
                    r += sr;
                }
                if (screens == 1) return true;
                break;
            }
            case PieceType::PAWN: {
                const int fwd = (by == RED) ? 1 : -1;
                if (df == 0 && dr == fwd) return true;
                if (pawn_has_crossed_river(from, by) && dr == 0 && std::abs(df) == 1) {
                    return true;
                }
                break;
            }
        }
    }
    return false;
}

bool Board::in_check() const {
    if (king_sq[side] == NO_SQUARE) return false;
    return is_attacked(king_sq[side], static_cast<Color>(side ^ 1));
}

void Board::generate_pseudo_legal(MoveList& out) const {
    out.size = 0;
    const Color us = side;

    for (int i = 0; i < n_pieces[us]; ++i) {
        const Square from = piece_list[us][i];
        const Piece p = squares[from];
        const int ff = file_of(from);
        const int rf = rank_of(from);

        switch (type_of(p)) {
            case PieceType::KING:
                for (int d = 0; d < 4; ++d) {
                    const int r = rf + ORTHO_DR[d];
                    const int f = ff + ORTHO_DF[d];
                    if (!in_board(r, f)) continue;
                    const Square to = make_square(r, f);
                    if (!in_palace(to, us)) continue;
                    add_if_capturable(out, *this, from, r, f, us);
                }
                break;

            case PieceType::ADVISOR:
                for (int d = 0; d < 4; ++d) {
                    const int r = rf + DIAG_DR[d];
                    const int f = ff + DIAG_DF[d];
                    if (!in_board(r, f)) continue;
                    const Square to = make_square(r, f);
                    if (!in_palace(to, us)) continue;
                    add_if_capturable(out, *this, from, r, f, us);
                }
                break;

            case PieceType::ELEPHANT:
                for (int d = 0; d < 4; ++d) {
                    const int r = rf + 2 * DIAG_DR[d];
                    const int f = ff + 2 * DIAG_DF[d];
                    const int mr = rf + DIAG_DR[d];
                    const int mf = ff + DIAG_DF[d];
                    if (!in_board(r, f) || !in_board(mr, mf)) continue;
                    if (squares[make_square(mr, mf)] != EMPTY) continue;
                    const Square to = make_square(r, f);
                    if (!elephant_on_own_side(to, us)) continue;
                    add_if_capturable(out, *this, from, r, f, us);
                }
                break;

            case PieceType::HORSE:
                for (int d = 0; d < 8; ++d) {
                    const int hr = rf + HORSE_HR[d];
                    const int hf = ff + HORSE_HF[d];
                    if (!in_board(hr, hf)) continue;
                    if (squares[make_square(hr, hf)] != EMPTY) continue;
                    add_if_capturable(out, *this, from, rf + HORSE_DR[d], ff + HORSE_DF[d],
                                      us);
                }
                break;

            case PieceType::CHARIOT:
                for (int d = 0; d < 4; ++d) {
                    int r = rf + ORTHO_DR[d];
                    int f = ff + ORTHO_DF[d];
                    while (in_board(r, f)) {
                        const Square to = make_square(r, f);
                        const Piece cap = squares[to];
                        if (cap == EMPTY) {
                            out.add(from, to);
                        } else {
                            if (color_of(cap) != us) out.add(from, to);
                            break;
                        }
                        r += ORTHO_DR[d];
                        f += ORTHO_DF[d];
                    }
                }
                break;

            case PieceType::CANNON:
                for (int d = 0; d < 4; ++d) {
                    int r = rf + ORTHO_DR[d];
                    int f = ff + ORTHO_DF[d];
                    bool seen = false;
                    while (in_board(r, f)) {
                        const Square to = make_square(r, f);
                        const Piece cap = squares[to];
                        if (!seen) {
                            if (cap == EMPTY) {
                                out.add(from, to);
                            } else {
                                seen = true;
                            }
                        } else {
                            if (cap != EMPTY) {
                                if (color_of(cap) != us) out.add(from, to);
                                break;
                            }
                        }
                        r += ORTHO_DR[d];
                        f += ORTHO_DF[d];
                    }
                }
                break;

            case PieceType::PAWN: {
                const int fwd = (us == RED) ? 1 : -1;
                add_if_capturable(out, *this, from, rf + fwd, ff, us);
                if (pawn_has_crossed_river(from, us)) {
                    add_if_capturable(out, *this, from, rf, ff + 1, us);
                    add_if_capturable(out, *this, from, rf, ff - 1, us);
                }
                break;
            }
        }
    }
}

void Board::make_move(Move m) {
    if (ply + 1 >= MAX_PLY) throw std::runtime_error("MAX_PLY exceeded");
    const Square from = m.from();
    const Square to = m.to();
    const Piece p = squares[from];
    const Piece cap = squares[to];
    const Color us = side;
    const Color them = static_cast<Color>(us ^ 1);

    Undo& u = undos[ply];
    u.hash = hash;
    u.move = m;
    u.captured = cap;
    u.halfmove = halfmove;
    u.fullmove = fullmove;

    if (cap != EMPTY) {
        const uint8_t cidx = list_index[to];
        const uint8_t last = --n_pieces[them];
        const Square last_sq = piece_list[them][last];
        piece_list[them][cidx] = last_sq;
        list_index[last_sq] = cidx;
        hash ^= Z_PIECE[cap][to];
        halfmove = 0;
    } else {
        ++halfmove;
    }
    if (type_of(p) == PieceType::PAWN) halfmove = 0;

    squares[from] = EMPTY;
    squares[to] = p;
    const uint8_t idx = list_index[from];
    piece_list[us][idx] = to;
    list_index[to] = idx;
    list_index[from] = 0xFF;
    hash ^= Z_PIECE[p][from];
    hash ^= Z_PIECE[p][to];
    if (type_of(p) == PieceType::KING) king_sq[us] = to;

    hash ^= Z_SIDE;
    side = them;
    if (us == BLACK) ++fullmove;

    ++ply;
    hist[ply] = hash;
}

Move Board::last_move() const {
    if (ply <= 0) throw std::runtime_error("no move to unmake");
    return undos[ply - 1].move;
}

void Board::unmake_move() {
    if (ply <= 0) throw std::runtime_error("no move to unmake");
    --ply;
    const Undo& u = undos[ply];
    const Move m = u.move;
    const Square from = m.from();
    const Square to = m.to();
    const Color them = side;
    const Color us = static_cast<Color>(them ^ 1);
    const Piece p = squares[to];
    const Piece cap = u.captured;

    side = us;
    halfmove = u.halfmove;
    fullmove = u.fullmove;
    hash = u.hash;

    squares[from] = p;
    squares[to] = cap;
    const uint8_t idx = list_index[to];
    piece_list[us][idx] = from;
    list_index[from] = idx;
    if (type_of(p) == PieceType::KING) king_sq[us] = from;

    if (cap != EMPTY) {
        const uint8_t cidx = n_pieces[them]++;
        piece_list[them][cidx] = to;
        list_index[to] = cidx;
    } else {
        list_index[to] = 0xFF;
    }
}

int Board::generate_legal(MoveList& out) {
    MoveList pseudo;
    generate_pseudo_legal(pseudo);
    out.size = 0;
    const Color us = side;
    for (int i = 0; i < pseudo.size; ++i) {
        const Move m = pseudo.moves[i];
        make_move(m);
        const bool legal = !is_attacked(king_sq[us], side);
        unmake_move();
        if (legal) out.add(m.from(), m.to());
    }
    return out.size;
}

bool Board::has_legal_move() {
    MoveList pseudo;
    generate_pseudo_legal(pseudo);
    const Color us = side;
    for (int i = 0; i < pseudo.size; ++i) {
        const Move m = pseudo.moves[i];
        make_move(m);
        const bool legal = !is_attacked(king_sq[us], side);
        unmake_move();
        if (legal) return true;
    }
    return false;
}

bool Board::is_legal(Move m) {
    // Must be a generated pseudo-legal move of the side to move, then king-safe.
    MoveList pseudo;
    generate_pseudo_legal(pseudo);
    bool found = false;
    for (int i = 0; i < pseudo.size; ++i) {
        if (pseudo.moves[i] == m) {
            found = true;
            break;
        }
    }
    if (!found) return false;
    const Color us = side;
    make_move(m);
    const bool ok = !is_attacked(king_sq[us], side);
    unmake_move();
    return ok;
}

Terminal Board::terminal() {
    Terminal t;
    if (ply >= MAX_PLY - 1) {
        t.outcome = Outcome::DRAW;
        t.reason = TerminalReason::MAX_PLY;
        return t;
    }
    if (repetition_count() >= 3) {
        t.outcome = Outcome::DRAW;
        t.reason = TerminalReason::REPETITION;
        return t;
    }
    if (halfmove >= 120) {
        t.outcome = Outcome::DRAW;
        t.reason = TerminalReason::NO_PROGRESS;
        return t;
    }
    if (!has_legal_move()) {
        const bool check = in_check();
        t.reason = check ? TerminalReason::CHECKMATE : TerminalReason::STALEMATE;
        t.outcome = (side == RED) ? Outcome::BLACK_WIN : Outcome::RED_WIN;
        return t;
    }
    return t;
}

uint64_t Board::perft(int depth) {
    if (depth <= 0) return 1;
    MoveList ml;
    generate_legal(ml);
    if (depth == 1) return static_cast<uint64_t>(ml.size);
    uint64_t nodes = 0;
    for (int i = 0; i < ml.size; ++i) {
        make_move(ml.moves[i]);
        nodes += perft(depth - 1);
        unmake_move();
    }
    return nodes;
}

}  // namespace xiangqi
