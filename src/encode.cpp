#include "xiangqi/encode.hpp"

#include <cstring>

namespace xiangqi {
namespace {

void write_frame(const Board& b, Color us, bool flip, const EncodeSpec& spec, float* dest) {
    const int frame_c = n_frame_planes(spec);
    if (frame_c == 0) return;
    std::memset(dest, 0, static_cast<size_t>(frame_c) * N_SQUARES * sizeof(float));

    const Color them = static_cast<Color>(us ^ 1);
    const int opp_base = spec.our_pieces ? N_PIECE_TYPES : 0;

    for (int sq = 0; sq < N_SQUARES; ++sq) {
        const Piece p = b.squares[sq];
        if (p == EMPTY) continue;
        const Square osq = flip ? flip_square(static_cast<Square>(sq)) : static_cast<Square>(sq);
        const int t = static_cast<int>(type_of(p));
        const Color c = color_of(p);
        if (spec.our_pieces && c == us) {
            dest[t * N_SQUARES + osq] = 1.f;
        }
        if (spec.opp_pieces && c == them) {
            dest[(opp_base + t) * N_SQUARES + osq] = 1.f;
        }
    }
}

void fill_constant_plane(float* dest, float value) {
    for (int i = 0; i < N_SQUARES; ++i) dest[i] = value;
}

}  // namespace

void encode_state(const Board& current, const Board* history, int n_history,
                  const EncodeSpec& spec, float* out) {
    const int T = spec.history_length > 0 ? spec.history_length : 1;
    const int frame_c = n_frame_planes(spec);
    const int extra_c = n_extra_planes(spec);
    const int total_c = T * frame_c + extra_c;
    std::memset(out, 0, static_cast<size_t>(total_c) * N_SQUARES * sizeof(float));

    const Color us = current.side;
    const bool flip = should_flip(current, spec);

    // history[0] is oldest. We want the last (T-1) history frames + current.
    // Slot 0 is the oldest of the T-window (zero if we don't have it).
    const int n_past = T - 1;
    const int have = n_history < 0 ? 0 : n_history;
    const int use_past = have < n_past ? have : n_past;
    const int past_skip = have - use_past;  // drop older than the window

    int slot = n_past - use_past;  // leading zero frames
    for (int i = 0; i < use_past; ++i) {
        write_frame(history[past_skip + i], us, flip, spec, out + slot * frame_c * N_SQUARES);
        ++slot;
    }
    write_frame(current, us, flip, spec, out + (T - 1) * frame_c * N_SQUARES);

    float* extra = out + T * frame_c * N_SQUARES;
    int e = 0;
    if (spec.side_to_move) {
        // Absolute side: 1 if red to move. Useful when perspective is absolute.
        fill_constant_plane(extra + e * N_SQUARES, current.side == RED ? 1.f : 0.f);
        ++e;
    }
    if (spec.halfmove) {
        const float scale = spec.halfmove_scale > 0.f ? spec.halfmove_scale : 120.f;
        fill_constant_plane(extra + e * N_SQUARES, static_cast<float>(current.halfmove) / scale);
        ++e;
    }
    if (spec.fullmove) {
        const float scale = spec.fullmove_scale > 0.f ? spec.fullmove_scale : 400.f;
        fill_constant_plane(extra + e * N_SQUARES, static_cast<float>(current.fullmove) / scale);
        ++e;
    }
    if (spec.ones) {
        fill_constant_plane(extra + e * N_SQUARES, 1.f);
        ++e;
    }
}

int legal_indices(Board& current, const EncodeSpec& spec, int* out_indices) {
    MoveList ml;
    current.generate_legal(ml);
    const bool flip = should_flip(current, spec);
    for (int i = 0; i < ml.size; ++i) {
        out_indices[i] = move_to_index(ml.moves[i], flip);
    }
    return ml.size;
}

}  // namespace xiangqi
