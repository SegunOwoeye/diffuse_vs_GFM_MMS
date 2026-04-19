#pragma once

#include <vector>
#include <array>

#include "src/dim/state.hpp"
#include "src/dim/eos.hpp"
#include "src/dim/eos_params.hpp"
#include "src/dim/riemann/hllc.hpp"


// [0] Solve 1D line update
template<int DIM, int NMAT, typename EOS>
inline void solve_line(
    const std::vector<Conserved<DIM, NMAT>>& U_in,
    std::vector<Conserved<DIM, NMAT>>& U_out,
    double dx,
    double dt,
    const EOSParams<NMAT>& params,
    int dir
)
{
    const int n = static_cast<int>(U_in.size());

    std::vector<Conserved<DIM, NMAT>> F(n + 1);

    std::array<double, DIM> normal{};
    normal[dir] = 1.0;

    // [0.1] Fluxes
    for (int i = 0; i < n - 1; ++i) {
        F[i + 1] = hllc_flux_normal<DIM, NMAT, EOS>(
            U_in[i],
            U_in[i + 1],
            normal,
            params
        );
    }

    // [0.2] Boundary (zero-gradient)
    F[0] = F[1];
    F[n] = F[n - 1];

    // [0.3] Update
    for (int i = 0; i < n; ++i) {
        U_out[i] = U_in[i] - (dt / dx) * (F[i + 1] - F[i]);
    }
}

// [1] Sweep direction over full grid
template<int DIM, int NMAT, typename EOS>
inline void sweep_direction_dispatch(
    int dir,
    const std::vector<Conserved<DIM, NMAT>>& U_in,
    const std::array<int, DIM>& N,
    const std::array<double, DIM>& dx,
    const EOSParams<NMAT>& params,
    double dt,
    std::vector<Conserved<DIM, NMAT>>& U_out
)
{
    const auto stride = compute_strides<DIM>(N);

    std::array<int, DIM> idx{};

    std::vector<Conserved<DIM, NMAT>> line_in;
    std::vector<Conserved<DIM, NMAT>> line_out;

    // loop over all transverse indices
    const int total_lines = U_in.size() / N[dir];

    for (int linear = 0; linear < total_lines; ++linear) {

        // reconstruct base index (excluding dir)
        int tmp = linear;

        for (int d = DIM - 1; d >= 0; --d) {
            if (d == dir) continue;

            idx[d] = tmp % N[d];
            tmp /= N[d];
        }

        // extract
        extract_line<DIM, NMAT>(
            U_in,
            N,
            stride,
            dir,
            idx,
            line_in
        );

        line_out.resize(line_in.size());

        // solve
        solve_line<DIM, NMAT, EOS>(
            line_in,
            line_out,
            dx[dir],
            dt,
            params,
            dir
        );

        // write back
        write_line<DIM, NMAT>(
            U_out,
            N,
            stride,
            dir,
            idx,
            line_out
        );
    }
}