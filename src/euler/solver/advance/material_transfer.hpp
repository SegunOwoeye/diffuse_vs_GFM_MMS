#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdexcept>
#include <vector>

#include "src/euler/primitives.hpp"
#include "src/euler/solver/solver_context.hpp"
#include "src/euler/state.hpp"
#include "src/euler/thermo_compute.hpp"

template<int DIM, typename EOS>
inline std::vector<Conserved<DIM>> transfer_reassigned_material_states(
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_before,
    const std::vector<int>& material_after,
    const SolverContext<DIM>& ctx
)
{
    if (material_before.size() != material_after.size() ||
        material_before.size() != U.size()) {
        throw std::runtime_error(
            "transfer_reassigned_material_states: size mismatch"
        );
    }

    std::vector<Conserved<DIM>> U_transferred = U;
    const int nmat = static_cast<int>(ctx.material_params.size());

    for (int id = 0; id < static_cast<int>(U.size()); ++id) {
        const int old_mat = material_before[id];
        const int new_mat = material_after[id];

        if (old_mat < 0 || old_mat >= nmat ||
            new_mat < 0 || new_mat >= nmat) {
            throw std::runtime_error(
                "transfer_reassigned_material_states: invalid material id"
            );
        }
    }

    #pragma omp parallel for
    for (int id = 0; id < static_cast<int>(U.size()); ++id) {
        const int old_mat = material_before[id];
        const int new_mat = material_after[id];

        if (old_mat == new_mat) {
            continue;
        }

        const ThermoState<DIM> T =
            compute_thermo<DIM, EOS>(U[id], ctx.material_params[old_mat]);

        Primitive<DIM> P{};
        P.rho = T.rho;
        P.vel = T.vel;
        P.p = T.p;

        U_transferred[id] =
            prim_to_cons<DIM, EOS>(P, ctx.material_params[new_mat]);
    }

    return U_transferred;
}
