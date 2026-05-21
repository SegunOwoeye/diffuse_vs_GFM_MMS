#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "src/euler/conservative.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/gfm/ghost.hpp"
#include "src/euler/primitives.hpp"
#include "src/euler/gfm/tracked_interface.hpp"
#include "src/euler/level_set/level_set_core.hpp"
#include "src/euler/solver/solver_context.hpp"
#include "src/euler/thermo_compute.hpp"
#include "src/math/numerical_safety.hpp"
#include "src/math/vector_ops.hpp"


/*
    [0] Real Ghost Fluid Method workspace.

    The rGFM construction returns corrected real states next to interfaces,
    per-material extension states for ghost-material fluxing, and interface
    normal-speed fields used by level-set transport.
*/
template<int DIM>
struct RGFMInterfaceData {
    std::vector<Conserved<DIM>> U_real;
    std::vector<std::vector<Conserved<DIM>>> material_states;
    std::vector<std::vector<double>> normal_speed_fields;
};


// [0.1] Boundary-cell handle used while pairing opposite sides of an interface.
template<int DIM>
struct RGFMBoundaryCell {
    int id = -1;
    std::array<int, DIM> idx{};
};


/*
    [0.2] Primitive extension buffer for one material.

    fixed cells are trusted real/interface samples, assigned cells are available
    to extrapolate from, and distance/normal orient the extension relative to
    the closest tracked material interface.
*/
template<int DIM>
struct RGFMPrimitiveField {
    std::vector<double> rho;
    std::vector<std::array<double, DIM>> vel;
    std::vector<double> normal_velocity;
    std::vector<double> tangential_velocity;
    std::vector<double> p;
    std::vector<char> fixed;
    std::vector<char> assigned;
    std::vector<double> distance;
    std::vector<std::array<double, DIM>> normal;
};


// [0.3] Detect cells that touch a zero level set or sign-changing neighbour
template<int DIM>
inline bool rgfm_is_bordering_cell(
    int id,
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    double phi_tol
)
{
    const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);
    const double phi_id = phi[id];

    if (std::abs(phi_id) <= phi_tol) {
        return true;
    }

    for (int dir = 0; dir < DIM; ++dir) {
        for (const int step : {-1, 1}) {
            std::array<int, DIM> nb_idx{};

            if (!try_offset_index<DIM>(idx, dir, step, grid, nb_idx)) {
                continue;
            }

            const int nb_id = flatten_index<DIM>(nb_idx, grid);
            const double phi_nb = phi[nb_id];

            if (phi_id * phi_nb <= 0.0 ||
                (std::abs(phi_id) <= phi_tol && std::abs(phi_nb) <= phi_tol)) {
                return true;
            }
        }
    }

    return false;
}


// [1] Detect cells whose direct neighbours belong to a different material
template<int DIM>
inline bool rgfm_is_material_boundary_cell(
    int id,
    const std::vector<int>& material_id,
    int negative_material_id,
    const LevelSetGrid<DIM>& grid
)
{
    const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);
    const bool is_negative_side = material_id[id] == negative_material_id;

    for (int dir = 0; dir < DIM; ++dir) {
        for (const int step : {-1, 1}) {
            std::array<int, DIM> nb_idx{};

            if (!try_offset_index<DIM>(idx, dir, step, grid, nb_idx)) {
                continue;
            }

            const int nb_id = flatten_index<DIM>(nb_idx, grid);
            const bool nb_is_negative_side = material_id[nb_id] == negative_material_id;

            if (is_negative_side != nb_is_negative_side) {
                return true;
            }
        }
    }

    return false;
}


// [2] Squared physical distance between two grid indices
template<int DIM>
inline double rgfm_grid_distance2(
    const std::array<int, DIM>& a,
    const std::array<int, DIM>& b,
    const std::array<double, DIM>& dx
)
{
    double dist2 = 0.0;

    for (int d = 0; d < DIM; ++d) {
        const double delta = static_cast<double>(a[d] - b[d]) * dx[d];
        dist2 += delta * delta;
    }

    return dist2;
}


/*
    [3] Match a boundary cell with an opposite-side partner

    A local search is tried first so nearby interface pairs win on coarse or
    curved grids. The score favours aligned normals and then physical distance.
*/
template<int DIM>
inline int rgfm_find_normal_matched_partner(
    const RGFMBoundaryCell<DIM>& cell,
    const std::vector<RGFMBoundaryCell<DIM>>& candidates,
    const std::vector<std::array<double, DIM>>& normals,
    const std::array<double, DIM>& dx
)
{
    if (candidates.empty()) {
        return -1;
    }

    const std::array<double, DIM> n_cell = normals[cell.id];
    const double n_cell_norm = norm<DIM>(n_cell);

    int best_id = -1;
    double best_score = std::numeric_limits<double>::max();

    double dx_min = dx[0];

    for (int d = 1; d < DIM; ++d) {
        dx_min = std::min(dx_min, dx[d]);
    }

    const double local_radius2 = 9.0 * dx_min * dx_min;
    bool found_local_candidate = false;

    auto visit_candidates = [&](bool local_only) {
        for (const auto& candidate : candidates) {
            const double dist2 = rgfm_grid_distance2<DIM>(
                cell.idx,
                candidate.idx,
                dx
            );

            if (local_only && dist2 > local_radius2) {
                continue;
            }

            found_local_candidate = found_local_candidate || local_only;

            const std::array<double, DIM> n_candidate = normals[candidate.id];
            const double n_candidate_norm = norm<DIM>(n_candidate);
            double normal_score = 1.0;

            if (n_cell_norm > 1e-14 && n_candidate_norm > 1e-14) {
                normal_score = 1.0 - dot<DIM>(n_cell, n_candidate);
            }

            const double grid_dist2 = safe_div(dist2, dx_min * dx_min, 1e-14);
            const double score = normal_score + 0.05 * grid_dist2;

            if (score < best_score) {
                best_score = score;
                best_id = candidate.id;
            }
        }
    };

    visit_candidates(true);

    if (found_local_candidate) {
        return best_id;
    }

    visit_candidates(false);
    return best_id;
}



// [4] Extend the normal interface speed away from fixed interface samples
template<int DIM>
inline void rgfm_extend_normal_speed(
    std::vector<double>& speed,
    const std::vector<char>& fixed,
    const std::vector<double>& phi,
    const LevelSetGrid<DIM>& grid,
    int iterations = 10
)
{
    const int Ntot = static_cast<int>(speed.size());
    std::vector<double> work = speed;

    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp parallel for if(!omp_in_parallel() && Ntot > 512)
        for (int id = 0; id < Ntot; ++id) {
            if (fixed[id] != 0) {
                continue;
            }

            const std::array<int, DIM> idx = unflatten_index<DIM>(id, grid);
            const double phi_abs = std::abs(phi[id]);

            double sum = 0.0;
            int count = 0;

            for (int dir = 0; dir < DIM; ++dir) {
                for (const int step : {-1, 1}) {
                    std::array<int, DIM> nb_idx{};

                    if (!try_offset_index<DIM>(idx, dir, step, grid, nb_idx)) {
                        continue;
                    }

                    const int nb_id = flatten_index<DIM>(nb_idx, grid);

                    if (std::abs(phi[nb_id]) <= phi_abs) {
                        sum += speed[nb_id];
                        count += 1;
                    }
                }
            }

            if (count > 0) {
                work[id] = sum / static_cast<double>(count);
            }
        }

        speed.swap(work);
    }
}


// [5] Convert a conserved state to a positive primitive state for one material
template<int DIM>
inline Primitive<DIM> rgfm_cons_to_primitive_for_material(
    const Conserved<DIM>& U,
    const EOSParams& params
)
{
    Primitive<DIM> P = cons_to_prim<DIM, IdealGasEOS>(U, params);
    P.rho = require_positive(P.rho, "rgfm: primitive rho", 1e-12);
    P.p = require_positive(P.p, "rgfm: primitive p", 1e-12);
    return P;
}


// [6] Store primitive fields and cache normal/tangential velocity components
template<int DIM>
inline void rgfm_store_primitive(
    RGFMPrimitiveField<DIM>& field,
    int id,
    const Primitive<DIM>& P,
    bool fixed
)
{
    field.rho[id] = require_positive(P.rho, "rgfm_store_primitive: rho", 1e-12);
    field.vel[id] = P.vel;
    field.p[id] = require_positive(P.p, "rgfm_store_primitive: p", 1e-12);

    if constexpr (DIM == 2) {
        std::array<double, DIM> n = field.normal[id];

        if (norm<DIM>(n) < 1e-14) {
            n[0] = 1.0;
            n[1] = 0.0;
        }

        n = normalize<DIM>(n, 1e-14);
        const std::array<double, DIM> t{{-n[1], n[0]}};
        field.normal_velocity[id] = dot<DIM>(P.vel, n);
        field.tangential_velocity[id] = dot<DIM>(P.vel, t);
    }

    field.assigned[id] = 1;

    if (fixed) {
        field.fixed[id] = 1;
    }
}


/*
    [7] Signed distance from a material to the nearest tracked interface

    Negative distance denotes the requested material. The fallback gives every
    cell a one-cell signed distance if no explicit interface is available.
*/
template<int DIM>
inline std::vector<double> rgfm_material_distance(
    int material,
    const std::vector<int>& material_id,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<TrackedInterface>& tracked_interfaces,
    const SolverContext<DIM>& ctx
)
{
    const int Ntot = static_cast<int>(material_id.size());
    std::vector<double> distance(Ntot, std::numeric_limits<double>::max());

    for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
        const int neg_mat = tracked_interfaces[k].negative_material_id;

        #pragma omp parallel for if(!omp_in_parallel() && Ntot > 512)
        for (int id = 0; id < Ntot; ++id) {
            const double signed_phi =
                (material == neg_mat) ? phi_list[k][id] : -phi_list[k][id];

            if (std::abs(signed_phi) < std::abs(distance[id])) {
                distance[id] = signed_phi;
            }
        }
    }

    double fallback = ctx.dx[0];

    for (int d = 1; d < DIM; ++d) {
        fallback = std::min(fallback, ctx.dx[d]);
    }

    #pragma omp parallel for if(!omp_in_parallel() && Ntot > 512)
    for (int id = 0; id < Ntot; ++id) {
        if (!std::isfinite(distance[id])) {
            distance[id] = (material_id[id] == material) ? -fallback : fallback;
        }
    }

    return distance;
}


// [8] Interface normals oriented outward from material
template<int DIM>
inline std::vector<std::array<double, DIM>> rgfm_material_normals(
    int material,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<TrackedInterface>& tracked_interfaces,
    const std::vector<std::vector<std::array<double, DIM>>>& normals_list,
    int Ntot
)
{
    std::vector<std::array<double, DIM>> normal(Ntot);
    std::vector<double> best_distance(Ntot, std::numeric_limits<double>::max());

    for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
        const int neg_mat = tracked_interfaces[k].negative_material_id;

        #pragma omp parallel for if(!omp_in_parallel() && Ntot > 512)
        for (int id = 0; id < Ntot; ++id) {
            const double signed_phi =
                (material == neg_mat) ? phi_list[k][id] : -phi_list[k][id];
            const double abs_phi = std::abs(signed_phi);

            if (abs_phi >= best_distance[id]) {
                continue;
            }

            best_distance[id] = abs_phi;
            normal[id] = (material == neg_mat)
                ? normals_list[k][id]
                : scale<DIM>(-1.0, normals_list[k][id]);
        }
    }

    #pragma omp parallel for if(!omp_in_parallel() && Ntot > 512)
    for (int id = 0; id < Ntot; ++id) {
        if (norm<DIM>(normal[id]) < 1e-14) {
            normal[id][0] = 1.0;
        }
    }

    return normal;
}


/*
    [9] Extrapolate a material's primitive field away from fixed interface samples

    The normal-directed pass behaves like a fast-marching fill and intentionally
    remains serial because each assignment can unlock later cells. The fallback
    pass fills any remaining holes from already assigned neighbours.
*/
template<int DIM>
inline void rgfm_extrapolate_primitive_field(
    RGFMPrimitiveField<DIM>& field,
    const LevelSetGrid<DIM>& grid,
    int passes = 8
)
{
    const int Ntot = static_cast<int>(field.rho.size());
    std::vector<int> order(Ntot);

    for (int id = 0; id < Ntot; ++id) {
        order[id] = id;
    }

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return std::abs(field.distance[a]) < std::abs(field.distance[b]);
    });

    for (int pass = 0; pass < passes; ++pass) {
        bool changed = false;

        for (const int id : order) {
            if (field.assigned[id] != 0) {
                continue;
            }

            const auto idx = unflatten_index<DIM>(id, grid);
            const double dist_id = std::abs(field.distance[id]);

            double rho_sum = 0.0;
            std::array<double, DIM> vel_sum{};
            double un_sum = 0.0;
            double ut_sum = 0.0;
            double p_sum = 0.0;
            int count = 0;

            double weight_sum = 0.0;
            std::array<double, DIM> n = field.normal[id];

            if (norm<DIM>(n) > 1e-14) {
                n = normalize<DIM>(n, 1e-14);
            }

            for (int dir = 0; dir < DIM; ++dir) {
                const double nd = n[dir];

                if (std::abs(nd) <= 1e-14) {
                    continue;
                }

                const int step = (nd >= 0.0) ? -1 : 1;
                std::array<int, DIM> nb_idx{};

                if (!try_offset_index<DIM>(idx, dir, step, grid, nb_idx)) {
                    continue;
                }

                const int nb_id = flatten_index<DIM>(nb_idx, grid);

                if (field.assigned[nb_id] == 0) {
                    continue;
                }

                if (std::abs(field.distance[nb_id]) > dist_id + 1e-12) {
                    continue;
                }

                const double weight = std::abs(nd) / grid.dx[dir];
                rho_sum += weight * field.rho[nb_id];
                p_sum += weight * field.p[nb_id];
                un_sum += weight * field.normal_velocity[nb_id];
                ut_sum += weight * field.tangential_velocity[nb_id];

                for (int d = 0; d < DIM; ++d) {
                    vel_sum[d] += weight * field.vel[nb_id][d];
                }

                weight_sum += weight;
            }

            if (weight_sum <= 0.0) {
                continue;
            }

            field.rho[id] = require_positive(
                safe_div(rho_sum, weight_sum, 1e-14),
                "rgfm_extrapolate_primitive_field: rho",
                1e-12
            );
            field.p[id] = require_positive(
                safe_div(p_sum, weight_sum, 1e-14),
                "rgfm_extrapolate_primitive_field: pressure",
                1e-12
            );
            field.normal_velocity[id] = safe_div(un_sum, weight_sum, 1e-14);
            field.tangential_velocity[id] = safe_div(ut_sum, weight_sum, 1e-14);

            for (int d = 0; d < DIM; ++d) {
                field.vel[id][d] = safe_div(vel_sum[d], weight_sum, 1e-14);
            }

            field.assigned[id] = 1;
            changed = true;
        }

        if (!changed) {
            break;
        }
    }

    for (int pass = 0; pass < passes; ++pass) {
        bool changed = false;

        for (int id = 0; id < Ntot; ++id) {
            if (field.assigned[id] != 0) {
                continue;
            }

            const auto idx = unflatten_index<DIM>(id, grid);
            double rho_sum = 0.0;
            std::array<double, DIM> vel_sum{};
            double un_sum = 0.0;
            double ut_sum = 0.0;
            double p_sum = 0.0;
            int count = 0;

            for (int dir = 0; dir < DIM; ++dir) {
                for (const int step : {-1, 1}) {
                    std::array<int, DIM> nb_idx{};

                    if (!try_offset_index<DIM>(idx, dir, step, grid, nb_idx)) {
                        continue;
                    }

                    const int nb_id = flatten_index<DIM>(nb_idx, grid);

                    if (field.assigned[nb_id] == 0) {
                        continue;
                    }

                    rho_sum += field.rho[nb_id];
                    p_sum += field.p[nb_id];
                    un_sum += field.normal_velocity[nb_id];
                    ut_sum += field.tangential_velocity[nb_id];

                    for (int d = 0; d < DIM; ++d) {
                        vel_sum[d] += field.vel[nb_id][d];
                    }

                    count += 1;
                }
            }

            if (count == 0) {
                continue;
            }

            const double denom = static_cast<double>(count);
            field.rho[id] = require_positive(
                safe_div(rho_sum, denom, 1e-14),
                "rgfm_extrapolate_primitive_field: fallback rho",
                1e-12
            );
            field.p[id] = require_positive(
                safe_div(p_sum, denom, 1e-14),
                "rgfm_extrapolate_primitive_field: fallback pressure",
                1e-12
            );
            field.normal_velocity[id] = safe_div(un_sum, denom, 1e-14);
            field.tangential_velocity[id] = safe_div(ut_sum, denom, 1e-14);

            for (int d = 0; d < DIM; ++d) {
                field.vel[id][d] = safe_div(vel_sum[d], denom, 1e-14);
            }

            field.assigned[id] = 1;
            changed = true;
        }

        if (!changed) {
            break;
        }
    }
}


/*
[10] Build all rGFM data needed by one split update

    Track :
        [10.1] Seed each material with its real primitive cells
        [10.2] Find and pair opposite-side material-boundary cells
        [10.3] Solve normal mixed-material Riemann problems at those pairs
        [10.4] Extend normal speeds and material ghost states away from interfaces
*/
template<int DIM, typename EOS>
inline RGFMInterfaceData<DIM> build_rgfm_interface_data(
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<TrackedInterface>& tracked_interfaces,
    const std::vector<std::vector<std::array<double, DIM>>>& normals_list,
    const SolverContext<DIM>& ctx
)
{
    const int Ntot = static_cast<int>(U.size());

    if (static_cast<int>(material_id.size()) != Ntot) {
        throw std::runtime_error("build_rgfm_interface_data: material size mismatch");
    }

    RGFMInterfaceData<DIM> data;
    data.U_real = U;
    data.material_states.assign(
        ctx.material_params.size(),
        std::vector<Conserved<DIM>>(Ntot)
    );
    data.normal_speed_fields.assign(
        phi_list.size(),
        std::vector<double>(Ntot, 0.0)
    );

    std::vector<RGFMPrimitiveField<DIM>> material_fields(ctx.material_params.size());
    std::vector<std::vector<double>> material_interface_score(
        ctx.material_params.size(),
        std::vector<double>(Ntot, std::numeric_limits<double>::max())
    );

    for (int mat = 0; mat < static_cast<int>(ctx.material_params.size()); ++mat) {
        auto& field = material_fields[mat];
        field.rho.assign(Ntot, 1.0);
        field.vel.assign(Ntot, std::array<double, DIM>{});
        field.normal_velocity.assign(Ntot, 0.0);
        field.tangential_velocity.assign(Ntot, 0.0);
        field.p.assign(Ntot, 1.0);
        field.fixed.assign(Ntot, 0);
        field.assigned.assign(Ntot, 0);
        field.normal = rgfm_material_normals<DIM>(
            mat,
            phi_list,
            tracked_interfaces,
            normals_list,
            Ntot
        );
        field.distance = rgfm_material_distance<DIM>(
            mat,
            material_id,
            phi_list,
            tracked_interfaces,
            ctx
        );

        #pragma omp parallel for if(!omp_in_parallel() && Ntot > 512)
        for (int id = 0; id < Ntot; ++id) {
            if (material_id[id] != mat) {
                continue;
            }

            Primitive<DIM> P = cons_to_prim<DIM, EOS>(
                U[id],
                ctx.material_params[mat]
            );
            rgfm_store_primitive<DIM>(field, id, P, true);
        }
    }

    for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
        if (static_cast<int>(phi_list[k].size()) != Ntot ||
            static_cast<int>(normals_list[k].size()) != Ntot) {
            throw std::runtime_error("build_rgfm_interface_data: level-set size mismatch");
        }

        std::vector<std::array<double, DIM>> velocity(Ntot);

        #pragma omp parallel for if(!omp_in_parallel() && Ntot > 512)
        for (int id = 0; id < Ntot; ++id) {
            const int mat = material_id[id];
            const ThermoState<DIM> T =
                compute_thermo<DIM, EOS>(U[id], ctx.material_params[mat]);
            velocity[id] = T.vel;
            data.normal_speed_fields[k][id] = dot<DIM>(T.vel, normals_list[k][id]);
        }

        const int neg_mat = tracked_interfaces[k].negative_material_id;
        std::vector<RGFMBoundaryCell<DIM>> negative_cells;
        std::vector<RGFMBoundaryCell<DIM>> positive_cells;

        for (int id = 0; id < Ntot; ++id) {
            if (!rgfm_is_material_boundary_cell<DIM>(
                    id,
                    material_id,
                    neg_mat,
                    ctx.level_set_grid)) {
                continue;
            }

            RGFMBoundaryCell<DIM> cell;
            cell.id = id;
            cell.idx = unflatten_index<DIM>(id, ctx.level_set_grid);

            if (material_id[id] == neg_mat) {
                negative_cells.push_back(cell);
            }
            else {
                positive_cells.push_back(cell);
            }
        }

        std::vector<char> fixed_speed(Ntot, 0);
        std::vector<double> state_score(Ntot, std::numeric_limits<double>::max());

        auto apply_pair = [&](int neg_id, int pos_id) {
            if (neg_id < 0 || pos_id < 0) {
                return;
            }

            if (material_id[neg_id] != neg_mat ||
                material_id[pos_id] == neg_mat) {
                return;
            }

            std::array<double, DIM> normal = normals_list[k][neg_id];

            if (norm<DIM>(normal) < 1e-14) {
                normal = normals_list[k][pos_id];
            }

            if (norm<DIM>(normal) < 1e-14) {
                return;
            }

            if (phi_list[k][neg_id] > phi_list[k][pos_id]) {
                normal = scale<DIM>(-1.0, normal);
            }

            const auto states = rgfm_interface_states_normal<DIM, EOS, EOS>(
                U[neg_id],
                U[pos_id],
                normal,
                ctx.material_params[material_id[neg_id]],
                ctx.material_params[material_id[pos_id]]
            );

            const double score_neg = std::abs(phi_list[k][neg_id]);
            const double score_pos = std::abs(phi_list[k][pos_id]);

            if (score_neg < state_score[neg_id]) {
                data.U_real[neg_id] = states.left;
                state_score[neg_id] = score_neg;
            }

            if (score_pos < state_score[pos_id]) {
                data.U_real[pos_id] = states.right;
                state_score[pos_id] = score_pos;
            }

            const int neg_mat = material_id[neg_id];
            const int pos_mat = material_id[pos_id];
            const Primitive<DIM> P_neg = cons_to_prim<DIM, EOS>(
                states.left,
                ctx.material_params[neg_mat]
            );
            const Primitive<DIM> P_pos = cons_to_prim<DIM, EOS>(
                states.right,
                ctx.material_params[pos_mat]
            );

            if constexpr (DIM == 1) {
                const int direction_to_pos = (pos_id > neg_id) ? 1 : -1;

                for (int m = 0; m < 4; ++m) {
                    const int id = neg_id + m * direction_to_pos;
                    if (id < 0 || id >= Ntot) {
                        continue;
                    }

                    rgfm_store_primitive<DIM>(
                        material_fields[neg_mat],
                        id,
                        P_neg,
                        true
                    );
                    material_interface_score[neg_mat][id] = 0.0;
                }

                for (int m = 0; m < 4; ++m) {
                    const int id = pos_id - m * direction_to_pos;
                    if (id < 0 || id >= Ntot) {
                        continue;
                    }

                    rgfm_store_primitive<DIM>(
                        material_fields[pos_mat],
                        id,
                        P_pos,
                        true
                    );
                    material_interface_score[pos_mat][id] = 0.0;
                }
            }

            if (score_neg < material_interface_score[neg_mat][neg_id]) {
                rgfm_store_primitive<DIM>(
                    material_fields[neg_mat],
                    neg_id,
                    P_neg,
                    true
                );
                material_interface_score[neg_mat][neg_id] = score_neg;
            }

            if (score_pos < material_interface_score[pos_mat][pos_id]) {
                rgfm_store_primitive<DIM>(
                    material_fields[pos_mat],
                    pos_id,
                    P_pos,
                    true
                );
                material_interface_score[pos_mat][pos_id] = score_pos;
            }

            data.normal_speed_fields[k][neg_id] = states.un_star;
            data.normal_speed_fields[k][pos_id] = states.un_star;
            fixed_speed[neg_id] = 1;
            fixed_speed[pos_id] = 1;
        };

        for (const auto& cell : negative_cells) {
            const int partner = rgfm_find_normal_matched_partner<DIM>(
                cell,
                positive_cells,
                normals_list[k],
                ctx.dx
            );

            apply_pair(cell.id, partner);
        }

        for (const auto& cell : positive_cells) {
            const int partner = rgfm_find_normal_matched_partner<DIM>(
                cell,
                negative_cells,
                normals_list[k],
                ctx.dx
            );

            apply_pair(partner, cell.id);
        }

        rgfm_extend_normal_speed<DIM>(
            data.normal_speed_fields[k],
            fixed_speed,
            phi_list[k],
            ctx.level_set_grid
        );
    }

    for (int mat = 0; mat < static_cast<int>(ctx.material_params.size()); ++mat) {
        auto& field = material_fields[mat];

        rgfm_extrapolate_primitive_field<DIM>(
            field,
            ctx.level_set_grid
        );

        #pragma omp parallel for if(!omp_in_parallel() && Ntot > 512)
        for (int id = 0; id < Ntot; ++id) {
            Primitive<DIM> P{};
            P.rho = field.rho[id];
            if constexpr (DIM == 2) {
                std::array<double, DIM> n = field.normal[id];

                if (norm<DIM>(n) < 1e-14) {
                    n[0] = 1.0;
                    n[1] = 0.0;
                }

                n = normalize<DIM>(n, 1e-14);
                const std::array<double, DIM> t{{-n[1], n[0]}};
                P.vel = add<DIM>(
                    scale<DIM>(field.normal_velocity[id], n),
                    scale<DIM>(field.tangential_velocity[id], t)
                );
            }
            else {
                P.vel = field.vel[id];
            }
            P.p = field.p[id];
            data.material_states[mat][id] = prim_to_cons<DIM, EOS>(
                P,
                ctx.material_params[mat]
            );
        }
    }

    return data;
}








