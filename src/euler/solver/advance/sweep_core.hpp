#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>
#include <array>
#include <stdexcept>
#include <string>

#include "src/euler/solver/advance/line_ops.hpp"
#include "src/euler/solver/advance/geometry.hpp"
#include "src/euler/solver/advance/interface_speed.hpp"

#include "src/euler/gfm/tracked_interface.hpp"
#include "src/euler/riemann/hllc.hpp"
#include "src/euler/gfm/ghost.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/flux.hpp"
#include "src/math/numerical_safety.hpp"


// [1] Advance one extracted line by one split sweep
template<int DIM, typename EOS>
inline void advance_line(
    int dir,
    const std::vector<Conserved<DIM>>& U_line,
    const std::vector<int>& mat_line,
    const std::vector<int>& id_line,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<TrackedInterface>& tracked_interfaces,
    const std::vector<std::vector<std::array<double, DIM>>>& normals_list,
    const SolverContext<DIM>& ctx,
    double dt,
    std::vector<Conserved<DIM>>& U_out,
    const std::vector<std::vector<Conserved<DIM>>>* material_states = nullptr
)
{
    const int L = static_cast<int>(U_line.size());
    const double phi_tol = geometry_tolerance(ctx.dx[dir]);

    if (L < 2) {
        return;
    }

    if (material_states != nullptr) {
        const int nmat = static_cast<int>(ctx.material_params.size());

        if (static_cast<int>(material_states->size()) != nmat) {
            throw std::runtime_error("advance_line: material state count mismatch");
        }

        std::vector<std::vector<Conserved<DIM>>> flux_by_material(
            nmat,
            std::vector<Conserved<DIM>>(L-1)
        );

        const std::array<double, DIM> n_axis = axis_normal<DIM>(dir);

        for (int mat = 0; mat < nmat; ++mat) {
            std::vector<Conserved<DIM>> U_mat_line(L);
            std::vector<EOSParams> mat_params(L, ctx.material_params[mat]);

            for (int i = 0; i < L; ++i) {
                U_mat_line[i] = (*material_states)[mat][id_line[i]];
            }

            std::vector<Conserved<DIM>> UL_face;
            std::vector<Conserved<DIM>> UR_face;

            reconstruct_line_interfaces_dispatch<DIM, EOS>(
                dir,
                U_mat_line,
                mat_params,
                dt,
                ctx.dx[dir],
                UL_face,
                UR_face,
                nullptr
            );

            for (int i = 0; i < L - 1; ++i) {
                enforce_positive_conserved<DIM, EOS>(
                    UL_face[i],
                    ctx.material_params[mat]
                );
                enforce_positive_conserved<DIM, EOS>(
                    UR_face[i],
                    ctx.material_params[mat]
                );

                flux_by_material[mat][i] = hllc_flux_normal<DIM, EOS>(
                    UL_face[i],
                    UR_face[i],
                    n_axis,
                    ctx.material_params[mat],
                    ctx.material_params[mat]
                );
            }
        }

        const double factor = dt / ctx.dx[dir];

        for (int i = 1; i < L - 1; ++i) {
            const int mat = mat_line[i];
            Conserved<DIM> U_next =
                (*material_states)[mat][id_line[i]]
                - factor * (
                    flux_by_material[mat][i]
                  - flux_by_material[mat][i - 1]
                );

            enforce_positive_conserved<DIM, EOS>(
                U_next,
                ctx.material_params[mat]
            );

            U_out[id_line[i]] = U_next;
        }

        return;
    }

    std::vector<EOSParams> cell_params(U_line.size());

    for (int i = 0; i < static_cast<int>(U_line.size()); ++i) {
        cell_params[i] = ctx.material_params[mat_line[i]];
    }

    std::vector<Conserved<DIM>> U_rgfm = U_line;
    std::vector<bool> is_interface(L - 1, false);
    std::vector<RGFMInterfaceStates<DIM>> interface_states(L - 1);

    for (int i = 0; i < L - 1; ++i) {
        const int matL = mat_line[i];
        const int matR = mat_line[i + 1];

        if (matL == matR) {
            continue;
        }

        const int idL = id_line[i];
        const int idR = id_line[i + 1];

        const int k = select_face_interface<DIM>(
            idL,
            idR,
            matL,
            matR,
            phi_list,
            tracked_interfaces,
            phi_tol
        );

        if (k < 0) {
            std::string msg =
                "advance_line: failed to identify active interface"
                " face_ids=(" + std::to_string(idL) + "," + std::to_string(idR) + ")"
                " mats=(" + std::to_string(matL) + "," + std::to_string(matR) + ")";

            if (!phi_list.empty()) {
                msg += " phi_samples=";

                for (int kk = 0; kk < static_cast<int>(phi_list.size()); ++kk) {
                    msg +=
                        "[" + std::to_string(kk) +
                        ":neg=" + std::to_string(tracked_interfaces[kk].negative_material_id) +
                        ",phiL=" + std::to_string(phi_list[kk][idL]) +
                        ",phiR=" + std::to_string(phi_list[kk][idR]) + "]";
                }
            }

            throw std::runtime_error(msg);
        }

        const std::array<double, DIM> n_axis = axis_normal<DIM>(dir);
        std::array<double, DIM> n_interface =
            build_face_normal<DIM>(normals_list[k], idL, idR, dir);

        if (dot<DIM>(n_interface, n_axis) < 0.0) {
            n_interface = scale<DIM>(-1.0, n_interface);
        }

        interface_states[i] = rgfm_interface_states_normal<DIM, EOS, EOS>(
            U_line[i],
            U_line[i + 1],
            n_interface,
            cell_params[i],
            cell_params[i + 1]
        );

        U_rgfm[i] = interface_states[i].left;
        U_rgfm[i + 1] = interface_states[i].right;
        is_interface[i] = true;
    }

    std::vector<Conserved<DIM>> UL_face;
    std::vector<Conserved<DIM>> UR_face;

    reconstruct_line_interfaces_dispatch<DIM, EOS>(
        dir,
        U_rgfm,
        cell_params,
        dt,
        ctx.dx[dir],
        UL_face,
        UR_face,
        &mat_line
    );

    for (int i = 0; i < L - 1; ++i) {
        if (!is_interface[i]) {
            continue;
        }

        UL_face[i] = interface_states[i].left;
        UR_face[i] = interface_states[i].right;
    }

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
            const std::array<double, DIM> n_face = axis_normal<DIM>(dir);

            const Conserved<DIM> FL = compute_flux_normal<DIM, EOS>(
                interface_states[i].left,
                n_face,
                paramsL
            );

            const Conserved<DIM> FR = compute_flux_normal<DIM, EOS>(
                interface_states[i].right,
                n_face,
                paramsR
            );


            /* 
            rGFM imposes pI/uI/rhoI on the real nodes before fluxing.
            the mixed face therefore uses material-specific physical fluxes.
            */
            F_left_interface[i] = FL;
            F_right_interface[i] = FR;

            F[i] = F_left_interface[i];
        }
    }

    const double factor = dt/ctx.dx[dir];

    for (int i = 1; i < L - 1; ++i) {
        Conserved<DIM> F_left{};
        Conserved<DIM> F_right{};

        if (is_interface[i-1]) {
            F_left = F_right_interface[i-1];
        }
        else {
            F_left = F[i-1];
        }

        if (is_interface[i]) {
            F_right = F_left_interface[i];
        }
        else {
            F_right = F[i];
        }

        Conserved<DIM> U_next = U_rgfm[i] -factor * (F_right-F_left);

        const EOSParams& params_i = ctx.material_params[mat_line[i]];
        enforce_positive_conserved<DIM, EOS>(U_next, params_i);

        U_out[id_line[i]] = U_next;
    }

}


// [2] Flat OpenMP-parallel line sweep
template<int DIM, typename EOS>
inline void sweep_lines_flat(
    int dir,
    const std::vector<Conserved<DIM>>& U_in,
    const std::vector<int>& material_id,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<TrackedInterface>& tracked_interfaces,
    const std::vector<std::vector<std::array<double, DIM>>>& normals_list,
    const SolverContext<DIM>& ctx,
    double dt,
    std::vector<Conserved<DIM>>& U_out,
    const std::vector<std::vector<Conserved<DIM>>>* material_states = nullptr
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
            tracked_interfaces,
            normals_list,
            ctx,
            dt,
            U_out,
            material_states
        );
    }
}


// [3] Dispatch one full split sweep direction 
template<int DIM, typename EOS>
inline void sweep_direction_dispatch(
    int dir,
    const std::vector<Conserved<DIM>>& U_in,
    const std::vector<int>& material_id,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<TrackedInterface>& tracked_interfaces,
    const std::vector<std::vector<std::array<double, DIM>>>& normals_list,
    const SolverContext<DIM>& ctx,
    double dt,
    std::vector<Conserved<DIM>>& U_out,
    const std::vector<std::vector<Conserved<DIM>>>* material_states = nullptr
)
{
    sweep_lines_flat<DIM, EOS>(
        dir,
        U_in,
        material_id,
        phi_list,
        tracked_interfaces,
        normals_list,
        ctx,
        dt,
        U_out,
        material_states
    );
}
