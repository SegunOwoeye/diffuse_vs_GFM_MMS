#pragma once

#include <vector>
#include <array>
#include <limits>
#include <stdexcept>
#include <cmath>

#include "src/euler/state.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/solver/cfl.hpp"
#include "src/euler/solver/solver_context.hpp"
#include "src/euler/thermo_compute.hpp"

#include "src/euler/reconstruction/muscl.hpp"
#include "src/euler/riemann/hllc.hpp"
#include "src/euler/gfm/ghost.hpp"
#include "src/euler/gfm/interface_utils.hpp"
#include "src/euler/level_set/level_set.hpp"

#include "src/euler/grid/grid_utils.hpp"
#include "src/math/vector_ops.hpp"


// [0] Step Result
template<int DIM>
struct StepResult {
    std::vector<Conserved<DIM>> U_new;
    std::vector<std::vector<double>> phi_list_new;
    double dt;
};


// [1] Build velocity field from conserved state
template<int DIM, typename EOS>
inline std::vector<std::array<double, DIM>> build_velocity_field(
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::vector<EOSParams>& material_params
)
{
    const int Ntot = static_cast<int>(U.size());

    std::vector<std::array<double, DIM>> vel(Ntot);

    for (int i = 0; i < Ntot; ++i) {
        const int mat = material_id.empty() ? 0 : material_id[i];

        const EOSParams& params =
            material_params.empty()
            ? EOSParams{}
            : material_params[mat];

        const ThermoState<DIM> T =
            compute_thermo<DIM, EOS>(U[i], params);

        vel[i] = T.vel;
    }

    return vel;
}


// [2] Build axis normal for sweep direction
template<int DIM>
inline std::array<double, DIM> axis_normal(
    int dir
)
{
    std::array<double, DIM> n{};
    n[dir] = 1.0;
    return n;
}


// [3] Build face normal from neighbouring cell normals
template<int DIM>
inline std::array<double, DIM> build_face_normal(
    const std::vector<std::array<double, DIM>>& normals,
    int idL,
    int idR,
    int dir,
    double tol = 1e-14
)
{
    std::array<double, DIM> n_face{};

    for (int d = 0; d < DIM; ++d) {
        n_face[d] = 0.5 * (normals[idL][d] + normals[idR][d]);
    }

    if (norm<DIM>(n_face) < tol) {
        return axis_normal<DIM>(dir);
    }

    return normalize<DIM>(n_face, tol);
}


// [4] Build projected normal speed field (extension / fallback field)
template<int DIM>
inline std::vector<double> build_projected_normal_speed_field(
    const std::vector<std::array<double, DIM>>& vel,
    const std::vector<std::array<double, DIM>>& normals
)
{
    const int Ntot = static_cast<int>(vel.size());

    if (static_cast<int>(normals.size()) != Ntot) {
        throw std::runtime_error("build_projected_normal_speed_field: size mismatch");
    }

    std::vector<double> Vn(Ntot, 0.0);

    for (int i = 0; i < Ntot; ++i) {
        Vn[i] = dot<DIM>(vel[i], normals[i]);
    }

    return Vn;
}


// [5] Select active interface on a face
template<int DIM>
inline int select_face_interface(
    int idL,
    int idR,
    int matL,
    int matR,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<int>& phi_material_ids
)
{
    if (matL == matR) {
        return -1;
    }

    int best_k = -1;
    double best_score = std::numeric_limits<double>::max();

    for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
        const int tagged_mat = phi_material_ids[k];

        if (tagged_mat != matL && tagged_mat != matR) {
            continue;
        }

        const double phiL = phi_list[k][idL];
        const double phiR = phi_list[k][idR];

        if (is_interface_face(phiL, phiR)) {
            const double score = std::abs(phiL) + std::abs(phiR);

            if (score < best_score) {
                best_score = score;
                best_k = k;
            }
        }
    }

    if (best_k >= 0) {
        return best_k;
    }

    for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
        const int tagged_mat = phi_material_ids[k];

        if (tagged_mat != matL && tagged_mat != matR) {
            continue;
        }

        const double phiL = phi_list[k][idL];
        const double phiR = phi_list[k][idR];
        const double score = std::abs(phiL) + std::abs(phiR);

        if (score < best_score) {
            best_score = score;
            best_k = k;
        }
    }

    return best_k;
}


// [6] Extract one grid line in direction DIR
template<int DIM, int DIR>
inline void extract_line(
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::array<int, DIM>& N,
    const std::array<int, DIM>& base_idx,
    std::vector<Conserved<DIM>>& U_line,
    std::vector<int>& mat_line,
    std::vector<int>& id_line
)
{
    const int L = N[DIR];

    U_line.resize(L);
    mat_line.resize(L);
    id_line.resize(L);

    for (int i = 0; i < L; ++i) {
        std::array<int, DIM> idx = base_idx;
        idx[DIR] = i;

        const int id = flatten_index<DIM>(idx, N);

        U_line[i] = U[id];
        mat_line[i] = material_id.empty() ? 0 : material_id[id];
        id_line[i] = id;
    }
}


// [7] Dispatch MUSCL reconstruction by sweep direction
template<int DIM, typename EOS>
inline void reconstruct_line_interfaces_dispatch(
    int dir,
    const std::vector<Conserved<DIM>>& U_line,
    const std::vector<int>& mat_line,
    const std::vector<EOSParams>& material_params,
    double dt,
    double dx,
    std::vector<Conserved<DIM>>& UL_face,
    std::vector<Conserved<DIM>>& UR_face,
    double rho_floor = 1e-10,
    double p_floor = 1e-10
)
{
    if constexpr (DIM >= 1) {
        if (dir == 0) {
            reconstruct_line_interfaces_material_aware<DIM, 0, EOS>(
                U_line,
                mat_line,
                material_params,
                dt,
                dx,
                UL_face,
                UR_face,
                rho_floor,
                p_floor
            );
            return;
        }
    }

    if constexpr (DIM >= 2) {
        if (dir == 1) {
            reconstruct_line_interfaces_material_aware<DIM, 1, EOS>(
                U_line,
                mat_line,
                material_params,
                dt,
                dx,
                UL_face,
                UR_face,
                rho_floor,
                p_floor
            );
            return;
        }
    }

    if constexpr (DIM >= 3) {
        if (dir == 2) {
            reconstruct_line_interfaces_material_aware<DIM, 2, EOS>(
                U_line,
                mat_line,
                material_params,
                dt,
                dx,
                UL_face,
                UR_face,
                rho_floor,
                p_floor
            );
            return;
        }
    }

    throw std::runtime_error("reconstruct_line_interfaces_dispatch: invalid direction");
}


// [8] Accumulate interface-normal speed from one extracted line
template<int DIM, typename EOS>
inline void accumulate_interface_normal_speed_line(
    int dir,
    const std::vector<Conserved<DIM>>& U_line,
    const std::vector<int>& mat_line,
    const std::vector<int>& id_line,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<int>& phi_material_ids,
    const std::vector<std::vector<std::array<double, DIM>>>& normals_list,
    const SolverContext<DIM>& ctx,
    double dt,
    std::vector<std::vector<double>>& Vn_sum,
    std::vector<std::vector<int>>& Vn_count
)
{
    const int L = static_cast<int>(U_line.size());

    if (L < 2) {
        return;
    }

    std::vector<Conserved<DIM>> UL_face;
    std::vector<Conserved<DIM>> UR_face;

    reconstruct_line_interfaces_dispatch<DIM, EOS>(
        dir,
        U_line,
        mat_line,
        ctx.material_params,
        dt,
        ctx.dx[dir],
        UL_face,
        UR_face
    );

    for (int i = 0; i < L - 1; ++i) {
        const int idL = id_line[i];
        const int idR = id_line[i + 1];

        const int matL = mat_line[i];
        const int matR = mat_line[i + 1];

        if (matL == matR) {
            continue;
        }

        const int k = select_face_interface<DIM>(
            idL,
            idR,
            matL,
            matR,
            phi_list,
            phi_material_ids
        );

        if (k < 0) {
            continue;
        }

        const EOSParams& paramsL = ctx.material_params[matL];
        const EOSParams& paramsR = ctx.material_params[matR];

        std::array<double, DIM> n_face{};

        if constexpr (DIM == 1) {
            if (ctx.use_axis_normals_in_1d) {
                n_face = axis_normal<DIM>(dir);
            }
            else {
                n_face = build_face_normal<DIM>(normals_list[k], idL, idR, dir);
            }
        }
        else {
            n_face = build_face_normal<DIM>(normals_list[k], idL, idR, dir);
        }

        enforce_positive_conserved<DIM, EOS>(UL_face[i], paramsL);
        enforce_positive_conserved<DIM, EOS>(UR_face[i], paramsR);

        const double Vn_face = interface_normal_speed_mcrs<DIM, EOS, EOS>(
            UL_face[i],
            UR_face[i],
            n_face,
            paramsL,
            paramsR
        );

        Vn_sum[k][idL] += Vn_face;
        Vn_sum[k][idR] += Vn_face;

        Vn_count[k][idL] += 1;
        Vn_count[k][idR] += 1;
    }
}


// [9] Recursive accumulation over all transverse index combinations
template<int DIM, typename EOS, int DIR>
inline void accumulate_interface_normal_speed_recursive(
    int depth,
    std::array<int, DIM>& base_idx,
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<int>& phi_material_ids,
    const std::vector<std::vector<std::array<double, DIM>>>& normals_list,
    const SolverContext<DIM>& ctx,
    double dt,
    std::vector<std::vector<double>>& Vn_sum,
    std::vector<std::vector<int>>& Vn_count
)
{
    if (depth == DIM) {
        std::vector<Conserved<DIM>> U_line;
        std::vector<int> mat_line;
        std::vector<int> id_line;

        extract_line<DIM, DIR>(
            U,
            material_id,
            ctx.N,
            base_idx,
            U_line,
            mat_line,
            id_line
        );

        accumulate_interface_normal_speed_line<DIM, EOS>(
            DIR,
            U_line,
            mat_line,
            id_line,
            phi_list,
            phi_material_ids,
            normals_list,
            ctx,
            dt,
            Vn_sum,
            Vn_count
        );

        return;
    }

    if (depth == DIR) {
        base_idx[depth] = 0;

        accumulate_interface_normal_speed_recursive<DIM, EOS, DIR>(
            depth + 1,
            base_idx,
            U,
            material_id,
            phi_list,
            phi_material_ids,
            normals_list,
            ctx,
            dt,
            Vn_sum,
            Vn_count
        );

        return;
    }

    for (int i = 0; i < ctx.N[depth]; ++i) {
        base_idx[depth] = i;

        accumulate_interface_normal_speed_recursive<DIM, EOS, DIR>(
            depth + 1,
            base_idx,
            U,
            material_id,
            phi_list,
            phi_material_ids,
            normals_list,
            ctx,
            dt,
            Vn_sum,
            Vn_count
        );
    }
}


// [10] Build per-level-set normal speed fields
//
// Starts from projected bulk velocity as an extension field and then
// overwrites interface-adjacent cells with HLLC/MCRS-consistent interface speed.
template<int DIM, typename EOS>
inline std::vector<std::vector<double>> build_interface_normal_speed_fields(
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::vector<std::array<double, DIM>>& vel,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<int>& phi_material_ids,
    const std::vector<std::vector<std::array<double, DIM>>>& normals_list,
    const SolverContext<DIM>& ctx,
    double dt
)
{
    const int Ntot = static_cast<int>(U.size());
    const int nphi = static_cast<int>(phi_list.size());

    std::vector<std::vector<double>> Vn_fields(nphi, std::vector<double>(Ntot, 0.0));

    // [10.1] Start from projected bulk velocity as a global extension field
    for (int k = 0; k < nphi; ++k) {
        Vn_fields[k] = build_projected_normal_speed_field<DIM>(vel, normals_list[k]);
    }

    // [10.2] Accumulate interface-consistent speed where interfaces exist
    std::vector<std::vector<double>> Vn_sum(nphi, std::vector<double>(Ntot, 0.0));
    std::vector<std::vector<int>> Vn_count(nphi, std::vector<int>(Ntot, 0));

    std::array<int, DIM> base_idx{};

    if constexpr (DIM >= 1) {
        accumulate_interface_normal_speed_recursive<DIM, EOS, 0>(
            0,
            base_idx,
            U,
            material_id,
            phi_list,
            phi_material_ids,
            normals_list,
            ctx,
            dt,
            Vn_sum,
            Vn_count
        );
    }

    if constexpr (DIM >= 2) {
        base_idx = {};
        accumulate_interface_normal_speed_recursive<DIM, EOS, 1>(
            0,
            base_idx,
            U,
            material_id,
            phi_list,
            phi_material_ids,
            normals_list,
            ctx,
            dt,
            Vn_sum,
            Vn_count
        );
    }

    if constexpr (DIM >= 3) {
        base_idx = {};
        accumulate_interface_normal_speed_recursive<DIM, EOS, 2>(
            0,
            base_idx,
            U,
            material_id,
            phi_list,
            phi_material_ids,
            normals_list,
            ctx,
            dt,
            Vn_sum,
            Vn_count
        );
    }

    // [10.3] Overwrite interface-adjacent cells with averaged interface speed
    for (int k = 0; k < nphi; ++k) {
        for (int id = 0; id < Ntot; ++id) {
            if (Vn_count[k][id] > 0) {
                Vn_fields[k][id] = Vn_sum[k][id] / static_cast<double>(Vn_count[k][id]);
            }
        }
    }

    return Vn_fields;
}


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

            // --- LOW SIDE ---
            if (idx[d] == 0) {
                const auto nb = offset_index<DIM>(idx, d, +1);
                const int nb_id = flatten_index<DIM>(nb, ctx.N);

                if (ctx.bc_lo[d] == BoundaryConditionType::transmissive) {
                    U[id] = U[nb_id];
                }
                else if (ctx.bc_lo[d] == BoundaryConditionType::reflective) {
                    U[id] = U[nb_id];
                    U[id].mom[d] *= -1.0;
                }
            }

            // --- HIGH SIDE ---
            if (idx[d] == ctx.N[d] - 1) {
                const auto nb = offset_index<DIM>(idx, d, -1);
                const int nb_id = flatten_index<DIM>(nb, ctx.N);

                if (ctx.bc_hi[d] == BoundaryConditionType::transmissive) {
                    U[id] = U[nb_id];
                }
                else if (ctx.bc_hi[d] == BoundaryConditionType::reflective) {
                    U[id] = U[nb_id];
                    U[id].mom[d] *= -1.0;
                }
            }
        }
    }
}


// [11] Advance one extracted line by one split sweep
template<int DIM, typename EOS>
inline void advance_line(
    int dir,
    const std::vector<Conserved<DIM>>& U_line,
    const std::vector<int>& mat_line,
    const std::vector<int>& id_line,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<int>& phi_material_ids,
    const std::vector<std::vector<std::array<double, DIM>>>& normals_list,
    const SolverContext<DIM>& ctx,
    double dt,
    std::vector<Conserved<DIM>>& U_out
)
{
    const int L = static_cast<int>(U_line.size());

    if (L < 2) {
        return;
    }

    std::vector<Conserved<DIM>> UL_face;
    std::vector<Conserved<DIM>> UR_face;

    reconstruct_line_interfaces_dispatch<DIM, EOS>(
        dir,
        U_line,
        mat_line,
        ctx.material_params,
        dt,
        ctx.dx[dir],
        UL_face,
        UR_face
    );

    for (int i = 0; i < L - 1; ++i) {
        const int matL = mat_line[i];
        const int matR = mat_line[i + 1];

        const EOSParams& paramsL = ctx.material_params[matL];
        const EOSParams& paramsR = ctx.material_params[matR];

        enforce_positive_conserved<DIM, EOS>(UL_face[i], paramsL);
        enforce_positive_conserved<DIM, EOS>(UR_face[i], paramsR);

        const Primitive<DIM> PL = cons_to_prim<DIM, EOS>(UL_face[i], paramsL);
        const Primitive<DIM> PR = cons_to_prim<DIM, EOS>(UR_face[i], paramsR);

        if (!std::isfinite(PL.rho) || !std::isfinite(PL.p) ||
            !std::isfinite(PR.rho) || !std::isfinite(PR.p)) {
            throw std::runtime_error("advance_line: non-finite reconstructed state");
        }

        if (PL.rho <= 0.0 || PL.p <= 0.0 ||
            PR.rho <= 0.0 || PR.p <= 0.0) {
            throw std::runtime_error("advance_line: non-physical reconstructed state");
        }
    }

    std::vector<Conserved<DIM>> F(L - 1);
    std::vector<bool> is_interface(L - 1, false);
    std::vector<Conserved<DIM>> F_left_interface(L - 1);
    std::vector<Conserved<DIM>> F_right_interface(L - 1);

    for (int i = 0; i < L - 1; ++i) {
        const int idL = id_line[i];
        const int idR = id_line[i + 1];

        const int matL = mat_line[i];
        const int matR = mat_line[i + 1];

        const EOSParams& paramsL = ctx.material_params[matL];
        const EOSParams& paramsR = ctx.material_params[matR];

        if (matL == matR) {
            const std::array<double, DIM> n_axis = axis_normal<DIM>(dir);

            F[i] = hllc_flux_normal<DIM, EOS>(
                UL_face[i],
                UR_face[i],
                n_axis,
                paramsL,
                paramsR
            );
        }
        else {
            is_interface[i] = true;

            const int k = select_face_interface<DIM>(
                idL,
                idR,
                matL,
                matR,
                phi_list,
                phi_material_ids
            );

            if (k < 0) {
                throw std::runtime_error("advance_line: failed to identify active interface");
            }

            const double phiL = phi_list[k][idL];
            const double phiR = phi_list[k][idR];

            // --- [1] Interface position (CRITICAL) ---
            const double denom = phiL - phiR;
            double alpha = 0.5;

            if (std::abs(denom) > 1e-14) {
                alpha = std::clamp(phiL / denom, 0.0, 1.0);
            }

            // --- [2] Interface normal ---
            std::array<double, DIM> n_face{};

            if constexpr (DIM == 1) {
                if (ctx.use_axis_normals_in_1d) {
                    n_face = axis_normal<DIM>(dir);
                } else {
                    n_face = build_face_normal<DIM>(normals_list[k], idL, idR, dir);
                }
            } else {
                n_face = build_face_normal<DIM>(normals_list[k], idL, idR, dir);
            }

            // --- [3] USE CELL-CENTERED STATES (FIXED) ---
            const auto ghosts = ghost_states_normal<DIM, EOS, EOS>(
                U_line[i],
                U_line[i + 1],
                n_face,
                paramsL,
                paramsR
            );

            // --- [4] Compute one-sided fluxes ---
            const Conserved<DIM> FL = hllc_flux_normal<DIM, EOS>(
                U_line[i],
                ghosts.first,
                n_face,
                paramsL,
                paramsL
            );

            const Conserved<DIM> FR = hllc_flux_normal<DIM, EOS>(
                ghosts.second,
                U_line[i + 1],
                n_face,
                paramsR,
                paramsR
            );

            // [5] 
            F_left_interface[i] = FL;
            F_right_interface[i] = FR;

            F[i] = F_left_interface[i];
        }
    }

    const double factor = dt / ctx.dx[dir];

    for (int i = 1; i < L - 1; ++i) {
        Conserved<DIM> F_left{};
        Conserved<DIM> F_right{};

        if (is_interface[i - 1]) {
            F_left = F_right_interface[i - 1];
        }
        else {
            F_left = F[i - 1];
        }

        if (is_interface[i]) {
            F_right = F_left_interface[i];
        }
        else {
            F_right = F[i];
        }

        Conserved<DIM> U_next = U_line[i] - factor * (F_right - F_left);

        const EOSParams& params_i = ctx.material_params[mat_line[i]];
        enforce_positive_conserved<DIM, EOS>(U_next, params_i);

        U_out[id_line[i]] = U_next;
    }

}


// [12] Recursive line sweep over all transverse index combinations
template<int DIM, typename EOS, int DIR>
inline void sweep_lines_recursive(
    int depth,
    std::array<int, DIM>& base_idx,
    const std::vector<Conserved<DIM>>& U_in,
    const std::vector<int>& material_id,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<int>& phi_material_ids,
    const std::vector<std::vector<std::array<double, DIM>>>& normals_list,
    const SolverContext<DIM>& ctx,
    double dt,
    std::vector<Conserved<DIM>>& U_out
)
{
    if (depth == DIM) {
        std::vector<Conserved<DIM>> U_line;
        std::vector<int> mat_line;
        std::vector<int> id_line;

        extract_line<DIM, DIR>(
            U_in,
            material_id,
            ctx.N,
            base_idx,
            U_line,
            mat_line,
            id_line
        );

        advance_line<DIM, EOS>(
            DIR,
            U_line,
            mat_line,
            id_line,
            phi_list,
            phi_material_ids,
            normals_list,
            ctx,
            dt,
            U_out
        );

        return;
    }

    if (depth == DIR) {
        base_idx[depth] = 0;

        sweep_lines_recursive<DIM, EOS, DIR>(
            depth + 1,
            base_idx,
            U_in,
            material_id,
            phi_list,
            phi_material_ids,
            normals_list,
            ctx,
            dt,
            U_out
        );

        return;
    }

    for (int i = 0; i < ctx.N[depth]; ++i) {
        base_idx[depth] = i;

        sweep_lines_recursive<DIM, EOS, DIR>(
            depth + 1,
            base_idx,
            U_in,
            material_id,
            phi_list,
            phi_material_ids,
            normals_list,
            ctx,
            dt,
            U_out
        );
    }
}


// [13] Dispatch one full split sweep direction
template<int DIM, typename EOS>
inline void sweep_direction_dispatch(
    int dir,
    const std::vector<Conserved<DIM>>& U_in,
    const std::vector<int>& material_id,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<int>& phi_material_ids,
    const std::vector<std::vector<std::array<double, DIM>>>& normals_list,
    const SolverContext<DIM>& ctx,
    double dt,
    std::vector<Conserved<DIM>>& U_out
)
{
    std::array<int, DIM> base_idx{};

    if constexpr (DIM >= 1) {
        if (dir == 0) {
            sweep_lines_recursive<DIM, EOS, 0>(
                0,
                base_idx,
                U_in,
                material_id,
                phi_list,
                phi_material_ids,
                normals_list,
                ctx,
                dt,
                U_out
            );
            return;
        }
    }

    if constexpr (DIM >= 2) {
        if (dir == 1) {
            sweep_lines_recursive<DIM, EOS, 1>(
                0,
                base_idx,
                U_in,
                material_id,
                phi_list,
                phi_material_ids,
                normals_list,
                ctx,
                dt,
                U_out
            );
            return;
        }
    }

    if constexpr (DIM >= 3) {
        if (dir == 2) {
            sweep_lines_recursive<DIM, EOS, 2>(
                0,
                base_idx,
                U_in,
                material_id,
                phi_list,
                phi_material_ids,
                normals_list,
                ctx,
                dt,
                U_out
            );
            return;
        }
    }

    throw std::runtime_error("sweep_direction_dispatch: invalid direction");
}


// [14] Advance One Step
template<int DIM, typename EOS>
inline StepResult<DIM> advance_one_step(
    const std::vector<Conserved<DIM>>& U,
    const SolverContext<DIM>& ctx
)
{
    const int Ntot = static_cast<int>(U.size());

    if (Ntot == 0) {
        throw std::runtime_error("advance_one_step: empty state");
    }

    ctx.validate(Ntot);

    // [14.1] Build current material map from the current level sets if requested
    std::vector<int> material_id_current = ctx.material_id;

    if (ctx.reassign_material_from_phi) {
        assign_material_ids_from_phi<DIM>(
            ctx.phi_list,
            ctx.phi_material_ids,
            ctx.background_material_id,
            material_id_current,
            ctx.level_set_grid
        );
    }

    // [14.2] CFL based on the current material map
    const double dt = compute_dt_cfl_materials<DIM, EOS>(
        U,
        material_id_current,
        ctx.material_params,
        ctx.dx,
        ctx.cfl,
        ctx.dt_max
    );

    // [14.3] Start from current level sets
    std::vector<std::vector<double>> phi_list_work = ctx.phi_list;

    // [14.4] Build normals from current level sets for transport
    std::vector<std::vector<std::array<double, DIM>>> normals_current(ctx.phi_list.size());

    for (int k = 0; k < static_cast<int>(ctx.phi_list.size()); ++k) {
        if constexpr (DIM == 1) {
            if (ctx.use_axis_normals_in_1d) {
                normals_current[k].assign(
                    Ntot,
                    axis_normal<DIM>(0)
                );
            }
            else {
                normals_current[k] = compute_normals<DIM>(
                    ctx.phi_list[k],
                    ctx.level_set_grid
                );
            }
        }
        else {
            normals_current[k] = compute_normals<DIM>(
                ctx.phi_list[k],
                ctx.level_set_grid
            );
        }
    }

    // [14.5] Advect level sets using interface-consistent normal speed fields
    if (ctx.advect_level_set && !phi_list_work.empty()) {
        const std::vector<std::array<double, DIM>> vel =
            build_velocity_field<DIM, EOS>(
                U,
                material_id_current,
                ctx.material_params
            );

        const std::vector<std::vector<double>> Vn_fields =
            build_interface_normal_speed_fields<DIM, EOS>(
                U,
                material_id_current,
                vel,
                ctx.phi_list,
                ctx.phi_material_ids,
                normals_current,
                ctx,
                dt
            );

        for (int k = 0; k < static_cast<int>(phi_list_work.size()); ++k) {
            phi_list_work[k] = advect_phi_normal_speed<DIM>(
                ctx.phi_list[k],
                Vn_fields[k],
                ctx.level_set_grid,
                dt
            );
        }
    }

    //
    static int step_counter = 0;
    step_counter++;

    if (ctx.reinit_enabled &&
        (step_counter % ctx.reinit_frequency == 0))
    {
        for (int k = 0; k < static_cast<int>(phi_list_work.size()); ++k) {
            phi_list_work[k] = reinitialise_phi<DIM>(
                phi_list_work[k],
                ctx.level_set_grid,
                ctx.reinit_iterations
            );
        }
    }


    // [14.6] Rebuild working material map from advected level sets
    std::vector<int> material_id_work = material_id_current;

    if (ctx.reassign_material_from_phi) {
        assign_material_ids_from_phi<DIM>(
            phi_list_work,
            ctx.phi_material_ids,
            ctx.background_material_id,
            material_id_work,
            ctx.level_set_grid
        );
    }

    // [14.7] Build normals from advected level sets
    std::vector<std::vector<std::array<double, DIM>>> normals_list(phi_list_work.size());

    for (int k = 0; k < static_cast<int>(phi_list_work.size()); ++k) {
        if constexpr (DIM == 1) {
            if (ctx.use_axis_normals_in_1d) {
                normals_list[k].assign(
                    Ntot,
                    axis_normal<DIM>(0)
                );
            }
            else {
                normals_list[k] = compute_normals<DIM>(
                    phi_list_work[k],
                    ctx.level_set_grid
                );
            }
        }
        else {
            normals_list[k] = compute_normals<DIM>(
                phi_list_work[k],
                ctx.level_set_grid
            );
        }
    }

    // [14.8] Directionally split update using rebuilt material map
    std::vector<Conserved<DIM>> U_stage = U;

    for (int dir = 0; dir < DIM; ++dir) {
        std::vector<Conserved<DIM>> U_next = U_stage;

        sweep_direction_dispatch<DIM, EOS>(
            dir,
            U_stage,
            material_id_work,
            phi_list_work,
            ctx.phi_material_ids,
            normals_list,
            ctx,
            dt,
            U_next
        );

        apply_boundary_conditions<DIM>(U_next, ctx);

        U_stage.swap(U_next);
    }

    return {U_stage, phi_list_work, dt};
}


