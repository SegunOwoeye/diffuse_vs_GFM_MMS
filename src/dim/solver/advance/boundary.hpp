#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "src/dim/primitives.hpp"
#include "src/dim/state.hpp"
#include "src/dim/solver/advance/geometry.hpp"
#include "src/io/config.hpp"
#include "src/math/numerical_safety.hpp"

namespace dim {
    inline bool is_nonreflective_boundary_name(
        const std::string& name
    )
    {
        return name == "nonreflective" || name == "non_reflective";
    }

    template<int DIM>
    inline std::string boundary_name_for_cell(
        const std::array<int, DIM>& idx,
        const std::array<int, DIM>& N,
        const Config<DIM>& cfg
    )
    {
        for (int d = 0; d < DIM; ++d) {
            if (idx[d] == 0) {
                return cfg.bc_lo[d];
            }

            if (idx[d] == N[d] - 1) {
                return cfg.bc_hi[d];
            }
        }

        return "transmissive";
    }

    template<int DIM>
    inline int boundary_normal_axis(
        const std::array<int, DIM>& idx,
        const std::array<int, DIM>& N
    )
    {
        for (int d = 0; d < DIM; ++d) {
            if (idx[d] == 0 || idx[d] == N[d] - 1) {
                return d;
            }
        }

        return 0;
    }

    template<int DIM>
    inline int boundary_outward_sign(
        const std::array<int, DIM>& idx,
        int axis
    )
    {
        return (idx[axis] == 0) ? -1 : 1;
    }

    template<int DIM>
    inline std::array<int, DIM> second_interior_index(
        const std::array<int, DIM>& idx,
        const std::array<int, DIM>& src_idx,
        const std::array<int, DIM>& N
    )
    {
        std::array<int, DIM> second = src_idx;

        for (int d = 0; d < DIM; ++d) {
            if (idx[d] == 0 && N[d] > 2) {
                second[d] = 2;
            }
            else if (idx[d] == N[d] - 1 && N[d] > 2) {
                second[d] = N[d] - 3;
            }
        }

        return second;
    }

    template<int DIM>
    inline void extrapolate_dim_state(
        State<DIM>& out,
        const State<DIM>& first,
        const State<DIM>& second
    )
    {
        out = first;

        for (int k = 0; k < static_cast<int>(out.partial_mass.size()); ++k) {
            out.partial_mass[k] = first.partial_mass[k] +
                (first.partial_mass[k] - second.partial_mass[k]);
        }

        for (int d = 0; d < DIM; ++d) {
            out.mom[d] = first.mom[d] + (first.mom[d] - second.mom[d]);
        }

        out.E = first.E + (first.E - second.E);

        for (int k = 0; k < static_cast<int>(out.alpha.size()); ++k) {
            out.alpha[k] = first.alpha[k] + (first.alpha[k] - second.alpha[k]);
        }
    }

    template<int DIM>
    inline State<DIM> local_nonreflective_state(
        const std::vector<State<DIM>>& U_old,
        const std::array<int, DIM>& idx,
        const std::array<int, DIM>& src_idx,
        const std::array<int, DIM>& N,
        const std::array<int, DIM>& stride,
        const EOSParams& params
    )
    {
        const int src_linear = flatten_index<DIM>(src_idx, stride);
        const std::array<int, DIM> second_idx =
            second_interior_index<DIM>(idx, src_idx, N);
        const int second_linear = flatten_index<DIM>(second_idx, stride);

        State<DIM> state = U_old[src_linear];
        State<DIM> extrapolated = U_old[src_linear];
        extrapolate_dim_state<DIM>(
            extrapolated,
            U_old[src_linear],
            U_old[second_linear]
        );

        const int axis = boundary_normal_axis<DIM>(idx, N);
        const int outward_sign = boundary_outward_sign<DIM>(idx, axis);
        const double extrapolated_rho =
            std::max(total_density<DIM>(extrapolated), 1e-12);
        const double extrapolated_normal_velocity =
            outward_sign * safe_div(extrapolated.mom[axis], extrapolated_rho);

        if (extrapolated_normal_velocity > 0.0) {
            const double rho = std::max(total_density<DIM>(state), 1e-12);
            double mom2_before = 0.0;
            double mom2_after = 0.0;

            for (int d = 0; d < DIM; ++d) {
                mom2_before += state.mom[d] * state.mom[d];
            }

            state.mom[axis] = extrapolated.mom[axis];

            for (int d = 0; d < DIM; ++d) {
                mom2_after += state.mom[d] * state.mom[d];
            }

            state.E += 0.5 * (mom2_after - mom2_before) / rho;
        }

        repair_state<DIM>(state, params);
        return state;
    }

    template<int DIM>
    inline void apply_boundary_conditions(
        std::vector<State<DIM>>& U,
        const std::array<int, DIM>& N,
        const Config<DIM>& cfg,
        const EOSParams& params
    )
    {
        const int total = static_cast<int>(U.size());
        const auto stride = compute_strides<DIM>(N);

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
            const std::string bc =
                boundary_name_for_cell<DIM>(idx, N, cfg);

            if (is_nonreflective_boundary_name(bc)) {
                U[linear] = local_nonreflective_state<DIM>(
                    U,
                    idx,
                    src_idx,
                    N,
                    stride,
                    params
                );
                continue;
            }

            U[linear] = U[src_linear];

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
