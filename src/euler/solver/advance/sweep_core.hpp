#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>
#include <array>
#include <stdexcept>

#include "src/euler/solver/advance/line_ops.hpp"
#include "src/euler/solver/advance/geometry.hpp"
#include "src/euler/solver/advance/interface_speed.hpp"

#include "src/euler/riemann/hllc.hpp"
#include "src/euler/gfm/ghost.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"


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
    std::vector<EOSParams> cell_params(U_line.size());

    for (int i = 0; i < static_cast<int>(U_line.size()); ++i) {
        cell_params[i] = ctx.material_params[mat_line[i]];
    }

    reconstruct_line_interfaces_dispatch<DIM, EOS>(
        dir,
        U_line,
        cell_params,
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

            // Interface position
            const double denom = phiL - phiR;
            double alpha = 0.5;

            if (std::abs(denom) > 1e-14) {
                alpha = std::clamp(phiL / denom, 0.0, 1.0);
            }

            // Interface normal
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

            // Use cell centrered states
            const auto ghosts = ghost_states_normal<DIM, EOS, EOS>(
                UL_face[i],
                UR_face[i],
                n_face,
                paramsL,
                paramsR
            );

            // Compute one-sided fluxes (uses HLLC) 
            const Conserved<DIM> FL = hllc_flux_normal<DIM, EOS>(
                UL_face[i],
                ghosts.first,
                n_face,
                paramsL,
                paramsL
            );

            const Conserved<DIM> FR = hllc_flux_normal<DIM, EOS>(
                ghosts.second,
                UR_face[i],
                n_face,
                paramsR,
                paramsR
            );


            // [5] Flux averaging
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


// [15] Flat OpenMP-parallel line sweep (replaces recursive traversal)
template<int DIM, typename EOS>
inline void sweep_lines_flat(
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
    const int num_lines = compute_num_lines<DIM>(dir, ctx.N);

    #pragma omp parallel for
    for (int line_id = 0; line_id < num_lines; ++line_id) {

        std::array<int, DIM> base_idx{};
        line_id_to_base_idx<DIM>(line_id, dir, ctx.N, base_idx);

        std::vector<Conserved<DIM>> U_line;
        std::vector<int> mat_line;
        std::vector<int> id_line;

        extract_line_dispatch<DIM>(
            dir,
            U_in,
            material_id,
            ctx.N,
            base_idx,
            U_line,
            mat_line,
            id_line
        );

        advance_line<DIM, EOS>(
            dir,
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
    }
}


// [16] Dispatch one full split sweep direction (flat version)
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
    sweep_lines_flat<DIM, EOS>(
        dir,
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
