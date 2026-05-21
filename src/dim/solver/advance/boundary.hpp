#pragma once

#include <array>
#include <string>
#include <vector>

#include "src/dim/state.hpp"
#include "src/dim/solver/advance/geometry.hpp"
#include "src/io/config.hpp"

namespace dim {

    template<int DIM>
    inline void apply_boundary_conditions(
        std::vector<State<DIM>>& U,
        const std::array<int, DIM>& N,
        const Config<DIM>& cfg
    )
    {
        const int total = static_cast<int>(U.size());
        const auto stride = compute_strides<DIM>(N);
        const std::vector<State<DIM>> U_old = U;

        #pragma omp parallel for
        for (int linear = 0; linear < total; ++linear) {
            const auto idx = unflatten_index<DIM>(linear, N);
            std::array<int, DIM> src_idx = idx;
            bool is_boundary = false;

            for (int d = 0; d < DIM; ++d) {
                if (N[d] <= 1) {
                    continue;
                }

                if (idx[d] == 0) {
                    src_idx[d] = 1;
                    is_boundary = true;
                }
                else if (idx[d] == N[d] - 1) {
                    src_idx[d] = N[d] - 2;
                    is_boundary = true;
                }
            }

            if (!is_boundary) {
                continue;
            }

            const int src_linear = flatten_index<DIM>(src_idx, stride);
            U[linear] = U_old[src_linear];

            for (int d = 0; d < DIM; ++d) {
                if ((idx[d] == 0 && cfg.bc_lo[d] == "reflective") ||
                    (idx[d] == N[d] - 1 && cfg.bc_hi[d] == "reflective"))
                {
                    U[linear].mom[d] *= -1.0;
                }
            }
        }
    }

} 
