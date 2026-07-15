#pragma once

#include <cmath>
#include <vector>

#include "src/euler/state.hpp"
#include "src/euler/reconstruction/muscl.hpp"
#include "src/sim/solver/solver_context.hpp"
#include "src/sim/grid/grid_utils.hpp"
#include "src/math/numerical_safety.hpp"

template<int DIM>
inline BoundaryConditionType boundary_condition_for_cell(
    const std::array<int, DIM>& idx,
    const SolverContext<DIM>& ctx
)
{
    for (int d = 0; d < DIM; ++d) {
        if (idx[d] == 0) {
            return ctx.bc_lo[d];
        }

        if (idx[d] == ctx.N[d] - 1) {
            return ctx.bc_hi[d];
        }
    }

    return BoundaryConditionType::transmissive;
}

template<int DIM>
inline int boundary_normal_axis(
    const std::array<int, DIM>& idx,
    const SolverContext<DIM>& ctx
)
{
    for (int d = 0; d < DIM; ++d) {
        if (idx[d] == 0 || idx[d] == ctx.N[d] - 1) {
            return d;
        }
    }

    return 0;
}

template<int DIM>
inline int boundary_outward_sign(
    const std::array<int, DIM>& idx,
    const SolverContext<DIM>& ctx,
    int axis
)
{
    return (idx[axis] == 0) ? -1 : 1;
}

template<int DIM>
inline std::array<int, DIM> second_interior_index(
    const std::array<int, DIM>& idx,
    const std::array<int, DIM>& src_idx,
    const SolverContext<DIM>& ctx
)
{
    std::array<int, DIM> second = src_idx;

    for (int d = 0; d < DIM; ++d) {
        if (idx[d] == 0 && ctx.N[d] > 2) {
            second[d] = 2;
        }
        else if (idx[d] == ctx.N[d] - 1 && ctx.N[d] > 2) {
            second[d] = ctx.N[d] - 3;
        }
    }

    return second;
}

template<int DIM>
inline bool has_physical_energy(
    const Conserved<DIM>& U
)
{
    if (!std::isfinite(U.rho) || U.rho <= 0.0 || !std::isfinite(U.E)) {
        return false;
    }

    double mom2 = 0.0;
    for (int d = 0; d < DIM; ++d) {
        if (!std::isfinite(U.mom[d])) {
            return false;
        }

        mom2 += U.mom[d] * U.mom[d];
    }

    return U.E > 0.5 * mom2 / std::max(U.rho, 1e-12);
}

template<int DIM, typename EOS>
inline Conserved<DIM> local_nonreflective_state(
    const std::vector<Conserved<DIM>>& U_old,
    const std::array<int, DIM>& idx,
    const std::array<int, DIM>& src_idx,
    const SolverContext<DIM>& ctx
)
{
    const int src_id = flatten_index<DIM>(src_idx, ctx.N);
    const std::array<int, DIM> second_idx =
        second_interior_index<DIM>(idx, src_idx, ctx);
    const int second_id = flatten_index<DIM>(second_idx, ctx.N);

    Conserved<DIM> state = U_old[src_id];
    const Conserved<DIM> extrapolated =
        U_old[src_id] + (U_old[src_id] - U_old[second_id]);

    const int axis = boundary_normal_axis<DIM>(idx, ctx);
    const int outward_sign = boundary_outward_sign<DIM>(idx, ctx, axis);
    const double extrapolated_normal_velocity =
        outward_sign * safe_div(extrapolated.mom[axis], extrapolated.rho);

    if (has_physical_energy<DIM>(extrapolated) &&
        extrapolated_normal_velocity > 0.0) {
        const double rho = std::max(state.rho, 1e-12);
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

    const int mat = ctx.material_id.empty() ? 0 : ctx.material_id[src_id];
    enforce_positive_conserved<DIM, EOS>(
        state,
        ctx.material_params[mat]
    );

    return state;
}

template<int DIM, typename EOS>
inline void apply_boundary_conditions(
    std::vector<Conserved<DIM>>& U,
    const SolverContext<DIM>& ctx
)
{
    const int Ntot = static_cast<int>(U.size());

    for (int id = 0; id < Ntot; ++id) {
        const auto idx = unflatten_index<DIM>(id, ctx.level_set_grid);
        std::array<int, DIM> src_idx = idx;
        bool is_boundary = false;

        for (int d = 0; d < DIM; ++d) {
            if (idx[d] == 0) {
                src_idx[d] = 1;
                is_boundary = true;
            }
            else if (idx[d] == ctx.N[d] - 1) {
                src_idx[d] = ctx.N[d] - 2;
                is_boundary = true;
            }
        }

        if (!is_boundary) {
            continue;
        }

        const int src_id = flatten_index<DIM>(src_idx, ctx.N);
        const BoundaryConditionType bc =
            boundary_condition_for_cell<DIM>(idx, ctx);

        if (bc == BoundaryConditionType::nonreflective) {
            U[id] = local_nonreflective_state<DIM, EOS>(
                U,
                idx,
                src_idx,
                ctx
            );
            continue;
        }

        U[id] = U[src_id];

        for (int d = 0; d < DIM; ++d) {
            if (idx[d] == 0 && ctx.bc_lo[d] == BoundaryConditionType::reflective) {
                U[id].mom[d] *= -1.0;
            }
            else if (idx[d] == ctx.N[d] - 1 && ctx.bc_hi[d] == BoundaryConditionType::reflective) {
                U[id].mom[d] *= -1.0;
            }
        }
    }
}
