#pragma once

#include <vector>

#include "src/euler/state.hpp"
#include "src/euler/solver/solver_context.hpp"
#include "src/euler/grid/grid_utils.hpp"

template<int DIM>
inline void apply_boundary_conditions(
    std::vector<Conserved<DIM>>& U,
    const SolverContext<DIM>& ctx
)
{
    const int Ntot = static_cast<int>(U.size());

    for (int id = 0; id < Ntot; ++id) {
        const auto idx = unflatten_index<DIM>(id, ctx.level_set_grid);

        for (int d = 0; d < DIM; ++d) {

            if (idx[d] == 0) {
                const auto nb = offset_index<DIM>(idx, d, +1);
                int nb_id = flatten_index<DIM>(nb, ctx.N);

                U[id] = U[nb_id];
                if (ctx.bc_lo[d] == BoundaryConditionType::reflective)
                    U[id].mom[d] *= -1.0;
            }

            if (idx[d] == ctx.N[d] - 1) {
                const auto nb = offset_index<DIM>(idx, d, -1);
                int nb_id = flatten_index<DIM>(nb, ctx.N);

                U[id] = U[nb_id];
                if (ctx.bc_hi[d] == BoundaryConditionType::reflective)
                    U[id].mom[d] *= -1.0;
            }
        }
    }
}