#pragma once

#include "xiangqi/board.hpp"

namespace xiangqi {

constexpr int N_PIECE_TYPES = 7;
constexpr int ACTION_FROM_TO = N_SQUARES * N_SQUARES;  // 8100

// Spatial 180° rotation: red's back rank <-> black's. sq' = 89 - sq.
inline Square flip_square(Square sq) {
    return static_cast<Square>(N_SQUARES - 1 - sq);
}

// Feature layout is derived from this spec (see n_input_planes).
// Default matches config/default.json.
struct EncodeSpec {
    bool perspective_current_player = true;
    bool our_pieces = true;
    bool opp_pieces = true;
    bool side_to_move = false;
    bool halfmove = true;
    bool fullmove = false;
    bool ones = false;
    float halfmove_scale = 120.f;
    float fullmove_scale = 400.f;
    int history_length = 1;
};

inline int n_frame_planes(const EncodeSpec& spec) {
    int n = 0;
    if (spec.our_pieces) n += N_PIECE_TYPES;
    if (spec.opp_pieces) n += N_PIECE_TYPES;
    return n;
}

inline int n_extra_planes(const EncodeSpec& spec) {
    int n = 0;
    if (spec.side_to_move) ++n;
    if (spec.halfmove) ++n;
    if (spec.fullmove) ++n;
    if (spec.ones) ++n;
    return n;
}

inline int n_input_planes(const EncodeSpec& spec) {
    const int t = spec.history_length > 0 ? spec.history_length : 1;
    return t * n_frame_planes(spec) + n_extra_planes(spec);
}

inline bool should_flip(const Board& b, const EncodeSpec& spec) {
    return spec.perspective_current_player && b.side == BLACK;
}

inline int move_to_index(Move m, bool flip) {
    Square from = m.from();
    Square to = m.to();
    if (flip) {
        from = flip_square(from);
        to = flip_square(to);
    }
    return static_cast<int>(from) * N_SQUARES + static_cast<int>(to);
}

inline Move index_to_move(int index, bool flip) {
    Square from = static_cast<Square>(index / N_SQUARES);
    Square to = static_cast<Square>(index % N_SQUARES);
    if (flip) {
        from = flip_square(from);
        to = flip_square(to);
    }
    return Move(from, to);
}

// Write CHW float32: shape (n_input_planes, 10, 9), row-major.
// history: oldest first, not including current. Missing older frames are zeros.
void encode_state(const Board& current, const Board* history, int n_history,
                  const EncodeSpec& spec, float* out);

inline void encode_state(const Board& current, const EncodeSpec& spec, float* out) {
    encode_state(current, nullptr, 0, spec, out);
}

int legal_indices(Board& current, const EncodeSpec& spec, int* out_indices);

}  // namespace xiangqi
