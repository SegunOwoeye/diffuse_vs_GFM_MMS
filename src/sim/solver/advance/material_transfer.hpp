#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "src/sim/grid/grid_utils.hpp"
#include "src/euler/primitives.hpp"
#include "src/sim/solver/solver_context.hpp"
#include "src/euler/state.hpp"
#include "src/euler/thermo_compute.hpp"
#include "src/math/numerical_safety.hpp"

template<int DIM>
inline int find_reassigned_material_donor(
    int id,
    int new_mat,
    const std::vector<int>& material_before,
    const std::vector<int>& material_after,
    const SolverContext<DIM>& ctx
)
{
    const std::array<int, DIM> idx =
        unflatten_index<DIM>(id, ctx.level_set_grid);

    int max_grid_extent = 0;
    for (int d = 0; d < DIM; ++d) {
        max_grid_extent = std::max(max_grid_extent, ctx.N[d]);
    }

    const int max_radius = std::min(max_grid_extent - 1, 8);

    for (int radius = 1; radius <= max_radius; ++radius) {
        std::array<int, DIM> offset{};
        offset.fill(-radius);

        int best_id = -1;
        int best_dist2 = std::numeric_limits<int>::max();

        while (true) {
            bool is_origin = true;
            bool on_shell = false;
            int dist2 = 0;

            std::array<int, DIM> nb_idx = idx;
            for (int d = 0; d < DIM; ++d) {
                is_origin = is_origin && (offset[d] == 0);
                on_shell = on_shell || (std::abs(offset[d]) == radius);
                dist2 += offset[d] * offset[d];
                nb_idx[d] += offset[d];
            }

            if (!is_origin && on_shell &&
                dist2 < best_dist2 &&
                is_valid_index<DIM>(nb_idx, ctx.level_set_grid)) {
                const int nb_id = flatten_index<DIM>(nb_idx, ctx.level_set_grid);

                if (material_after[nb_id] == new_mat &&
                    material_before[nb_id] == new_mat) {
                    best_id = nb_id;
                    best_dist2 = dist2;
                }
            }

            int d = 0;
            for (; d < DIM; ++d) {
                ++offset[d];
                if (offset[d] <= radius) {
                    break;
                }
                offset[d] = -radius;
            }

            if (d == DIM) {
                break;
            }
        }

        if (best_id >= 0) {
            return best_id;
        }
    }

    return -1;
}

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

        const int donor_id =
            find_reassigned_material_donor<DIM>(
                id,
                new_mat,
                material_before,
                material_after,
                ctx
            );

        const ThermoState<DIM> old_cell =
            compute_thermo<DIM, EOS>(U[id], ctx.material_params[old_mat]);

        Primitive<DIM> P{};
        P.rho = require_positive(
            old_cell.rho,
            "transfer_reassigned_material_states: old rho",
            1e-12
        );
        P.vel = old_cell.vel;
        P.p = require_positive(
            old_cell.p,
            "transfer_reassigned_material_states: old pressure",
            1e-12
        );

        if (donor_id >= 0) {
            const ThermoState<DIM> donor =
                compute_thermo<DIM, EOS>(U[donor_id], ctx.material_params[new_mat]);
            const EOSParams& new_params = ctx.material_params[new_mat];
            const double donor_entropy =
                EOS::entropy_invariant(
                    require_positive(
                        donor.rho,
                        "transfer_reassigned_material_states: donor rho",
                        1e-12
                    ),
                    require_positive(
                        donor.p,
                        "transfer_reassigned_material_states: donor pressure",
                        1e-12
                    ),
                    new_params
                );

            P.rho =
                require_positive(
                    EOS::density_from_p_invariant(P.p, donor_entropy, new_params),
                    "transfer_reassigned_material_states: transferred rho",
                    1e-12
                );
        }

        U_transferred[id] =
            prim_to_cons<DIM, EOS>(P, ctx.material_params[new_mat]);
    }

    return U_transferred;
}
