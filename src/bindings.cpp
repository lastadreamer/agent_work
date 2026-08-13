#include "xiangqi/board.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using xiangqi::Board;
using xiangqi::Move;
using xiangqi::MoveList;
using xiangqi::Outcome;
using xiangqi::Terminal;
using xiangqi::TerminalReason;

namespace {

Board copy_board(const Board& b) { return b; }

py::bytes board_to_bytes(const Board& b) {
    return py::bytes(reinterpret_cast<const char*>(&b), sizeof(Board));
}

Board board_from_bytes(const py::bytes& data) {
    Board::init_zobrist();
    const std::string raw = data;
    if (raw.size() != sizeof(Board)) {
        throw std::invalid_argument("Board pickle size mismatch");
    }
    Board b;
    std::memcpy(&b, raw.data(), sizeof(Board));
    return b;
}

py::list legal_moves_py(Board& b) {
    MoveList ml;
    b.generate_legal(ml);
    py::list out;
    for (int i = 0; i < ml.size; ++i) {
        out.append(ml.moves[i]);
    }
    return out;
}

py::list divide_py(Board& b, int depth) {
    MoveList ml;
    b.generate_legal(ml);
    py::list out;
    for (int i = 0; i < ml.size; ++i) {
        b.make_move(ml.moves[i]);
        const uint64_t n = (depth <= 1) ? 1ULL : b.perft(depth - 1);
        b.unmake_move();
        out.append(py::make_tuple(ml.moves[i], n));
    }
    return out;
}

struct RandomPlayResult {
    int red_wins = 0;
    int black_wins = 0;
    int draws = 0;
    uint64_t plies = 0;
};

RandomPlayResult play_random_games(int n_games, int max_plies, uint64_t seed) {
    // Splitmix64 — no shared RNG state, safe for threads/processes.
    auto next = [&seed]() {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    };

    RandomPlayResult acc;
    Board b;
    for (int g = 0; g < n_games; ++g) {
        b.set_fen(xiangqi::START_FEN);
        int plies = 0;
        while (plies < max_plies) {
            const Terminal t = b.terminal();
            if (t.outcome != Outcome::ONGOING) {
                if (t.outcome == Outcome::RED_WIN) ++acc.red_wins;
                else if (t.outcome == Outcome::BLACK_WIN) ++acc.black_wins;
                else ++acc.draws;
                break;
            }
            MoveList ml;
            b.generate_legal(ml);
            if (ml.size == 0) break;
            const Move m = ml.moves[static_cast<int>(next() % static_cast<uint64_t>(ml.size))];
            b.make_move(m);
            ++plies;
        }
        if (plies >= max_plies) ++acc.draws;
        acc.plies += static_cast<uint64_t>(plies);
    }
    return acc;
}

}  // namespace

PYBIND11_MODULE(_xiangqi, m) {
    m.doc() = "Xiangqi rules engine (AlphaZero phase 1): legal moves, make/unmake, perft";
    Board::init_zobrist();

    m.attr("RED") = py::int_(xiangqi::RED);
    m.attr("BLACK") = py::int_(xiangqi::BLACK);
    m.attr("EMPTY") = py::int_(xiangqi::EMPTY);
    m.attr("START_FEN") = xiangqi::START_FEN;
    m.attr("N_SQUARES") = xiangqi::N_SQUARES;
    m.attr("BOARD_NBYTES") = py::int_(sizeof(Board));

    py::enum_<Outcome>(m, "Outcome")
        .value("ONGOING", Outcome::ONGOING)
        .value("RED_WIN", Outcome::RED_WIN)
        .value("BLACK_WIN", Outcome::BLACK_WIN)
        .value("DRAW", Outcome::DRAW);

    py::enum_<TerminalReason>(m, "TerminalReason")
        .value("NONE", TerminalReason::NONE)
        .value("CHECKMATE", TerminalReason::CHECKMATE)
        .value("STALEMATE", TerminalReason::STALEMATE)
        .value("REPETITION", TerminalReason::REPETITION)
        .value("NO_PROGRESS", TerminalReason::NO_PROGRESS)
        .value("MAX_PLY", TerminalReason::MAX_PLY);

    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def(py::init<xiangqi::Square, xiangqi::Square>(), py::arg("from_sq"), py::arg("to_sq"))
        .def_static("from_iccs", &Move::from_iccs)
        .def_property_readonly("from_sq", &Move::from)
        .def_property_readonly("to_sq", &Move::to)
        .def_property_readonly("data", [](const Move& mv) { return mv.data; })
        .def("iccs", &Move::iccs)
        .def("__repr__",
             [](const Move& mv) { return "Move('" + mv.iccs() + "')"; })
        .def("__str__", &Move::iccs)
        .def("__eq__", &Move::operator==)
        .def("__hash__", [](const Move& mv) { return py::int_(mv.data); });

    py::class_<Terminal>(m, "Terminal")
        .def_readonly("outcome", &Terminal::outcome)
        .def_readonly("reason", &Terminal::reason)
        .def("__repr__", [](const Terminal& t) {
            return "<Terminal outcome=" + std::to_string(static_cast<int>(t.outcome)) +
                   " reason=" + std::to_string(static_cast<int>(t.reason)) + ">";
        });

    py::class_<Board>(m, "Board")
        .def(py::init<>())
        .def(py::init<const std::string&>(), py::arg("fen"))
        .def("copy", &copy_board)
        .def("__copy__", &copy_board)
        .def("__deepcopy__", [](const Board& b, py::object) { return copy_board(b); })
        .def("fen", &Board::fen)
        .def("set_fen", &Board::set_fen)
        .def("__str__", &Board::to_string)
        .def("__repr__", [](const Board& b) { return "Board('" + b.fen() + "')"; })
        .def("legal_moves", &legal_moves_py)
        .def("has_legal_move", &Board::has_legal_move)
        .def("is_legal", [](Board& b, const Move& mv) { return b.is_legal(mv); })
        .def("push", [](Board& b, const Move& mv) { b.make_move(mv); })
        .def("push_iccs", [](Board& b, const std::string& s) { b.make_move(Move::from_iccs(s)); })
        .def("pop", [](Board& b) { b.unmake_move(); })
        .def("make_move", &Board::make_move)
        .def("unmake_move", &Board::unmake_move)
        .def("last_move", &Board::last_move)
        .def("in_check", &Board::in_check)
        .def("is_attacked",
             [](const Board& b, int sq, int color) {
                 return b.is_attacked(static_cast<xiangqi::Square>(sq),
                                      static_cast<xiangqi::Color>(color));
             },
             py::arg("sq"), py::arg("by_color"))
        .def("side_to_move", &Board::side_to_move)
        .def("piece_at", &Board::piece_at)
        .def("is_capture", &Board::is_capture)
        .def("hash", [](const Board& b) { return b.hash; })
        .def("compute_hash", &Board::compute_hash)
        .def("ply", [](const Board& b) { return b.ply; })
        .def("halfmove_clock", [](const Board& b) { return b.halfmove; })
        .def("fullmove_number", [](const Board& b) { return b.fullmove; })
        .def("repetition_count", &Board::repetition_count)
        .def("terminal", &Board::terminal)
        .def("king_square", [](const Board& b, int color) { return b.king_sq[color]; })
        .def(
            "perft",
            [](Board& b, int depth) {
                py::gil_scoped_release release;
                return b.perft(depth);
            },
            py::arg("depth"))
        .def("divide", &divide_py, py::arg("depth"))
        .def("to_bytes", &board_to_bytes)
        .def_static("from_bytes", &board_from_bytes)
        .def(py::pickle(&board_to_bytes, &board_from_bytes));

    m.def("square_to_iccs", &xiangqi::square_to_iccs);
    m.def("iccs_to_square", &xiangqi::iccs_to_square);
    m.def(
        "play_random_games",
        [](int n_games, int max_plies, uint64_t seed) {
            py::gil_scoped_release release;
            return play_random_games(n_games, max_plies, seed);
        },
        py::arg("n_games"), py::arg("max_plies") = 400, py::arg("seed") = 1ULL,
        "Run independent random games in C++ with the GIL released.");

    py::class_<RandomPlayResult>(m, "RandomPlayResult")
        .def_readonly("red_wins", &RandomPlayResult::red_wins)
        .def_readonly("black_wins", &RandomPlayResult::black_wins)
        .def_readonly("draws", &RandomPlayResult::draws)
        .def_readonly("plies", &RandomPlayResult::plies);
}
