#include "xiangqi/board.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using xiangqi::Board;
using xiangqi::MoveList;
using xiangqi::START_FEN;

struct PerftCase {
    const char* name;
    const char* fen;
    int depth;
    uint64_t nodes;
};

// Chess Programming Wiki: Chinese Chess Perft Results (depths kept small for a CLI default).
static const PerftCase kCases[] = {
    {"start", START_FEN, 4, 3290240ULL},
    {"pos2", "r1ea1a3/4kh3/2h1e4/pHp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2EAKAE2 w - - 0 1", 3,
     43929ULL},
    {"pos3", "1ceak4/9/h2a5/2p1p3p/5cp2/2h2H3/6PCP/3AE4/2C6/3A1K1H1 w - - 0 1", 3, 8620ULL},
    {"pos4", "5a3/3k5/3aR4/9/5r3/5h3/9/3A1A3/5K3/2EC2E2 w - - 0 1", 3, 9850ULL},
    {"pos5", "CRH1k1e2/3ca4/4ea3/9/2hr5/9/9/4E4/4A4/4KA3 w - - 0 1", 3, 14808ULL},
};

static uint64_t run_perft(Board& b, int depth) { return b.perft(depth); }

int main(int argc, char** argv) {
    Board::init_zobrist();

    std::string mode = "check";
    if (argc >= 2) mode = argv[1];

    if (mode == "perft") {
        const int depth = (argc >= 3) ? std::stoi(argv[2]) : 4;
        const std::string fen = (argc >= 4) ? argv[3] : START_FEN;
        Board b(fen);
        const auto t0 = std::chrono::steady_clock::now();
        const uint64_t n = run_perft(b, depth);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double nps = (ms > 0.0) ? (static_cast<double>(n) / (ms / 1000.0)) : 0.0;
        std::cout << n << " nodes in " << ms << " ms (" << static_cast<uint64_t>(nps)
                  << " nps)\n";
        return 0;
    }

    if (mode == "divide") {
        const int depth = (argc >= 3) ? std::stoi(argv[2]) : 2;
        const std::string fen = (argc >= 4) ? argv[3] : START_FEN;
        Board b(fen);
        MoveList ml;
        b.generate_legal(ml);
        uint64_t total = 0;
        for (int i = 0; i < ml.size; ++i) {
            b.make_move(ml.moves[i]);
            const uint64_t n = (depth <= 1) ? 1ULL : b.perft(depth - 1);
            b.unmake_move();
            total += n;
            std::cout << ml.moves[i].iccs() << " " << n << "\n";
        }
        std::cout << "total " << total << "\n";
        return 0;
    }

    if (mode == "check" || mode == "--check") {
        int failed = 0;
        for (const auto& c : kCases) {
            Board b(c.fen);
            const auto t0 = std::chrono::steady_clock::now();
            const uint64_t n = run_perft(b, c.depth);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            const bool ok = n == c.nodes;
            std::cout << c.name << " depth " << c.depth << ": " << n << " (expected "
                      << c.nodes << ") " << ms << " ms " << (ok ? "OK" : "FAIL") << "\n";
            if (!ok) ++failed;
        }
        return failed ? 1 : 0;
    }

    std::cerr << "usage: xiangqi-perft [check|perft <depth> [fen]|divide <depth> [fen]]\n";
    return 2;
}
