#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>
#include <array>
#include <limits>
#include <stdexcept>

#include "src/euler/solver/advance/line_ops.hpp"
#include "src/euler/solver/advance/geometry.hpp"

#include "src/euler/gfm/tracked_interface.hpp"
#include "src/euler/solver/solver_context.hpp"
#include "src/euler/gfm/interface_utils.hpp"
#include "src/euler/gfm/ghost.hpp"
#include "src/euler/eos.hpp"
#include "src/math/vector_ops.hpp"


// [1] Build projected normal speed field
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

    #pragma omp parallel for
    for (int i = 0; i < Ntot; ++i) {
        Vn[i] = dot<DIM>(vel[i], normals[i]);
    }

    return Vn;
}

// [2] Select active interface on a face
template<int DIM>
inline int select_face_interface(
    int idL,
    int idR,
    int matL,
    int matR,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<TrackedInterface>& tracked_interfaces,
    double phi_tol
)
{
    if (matL == matR) {
        return -1;
    }

    int best_k = -1;
    double best_score = std::numeric_limits<double>::max();

    for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
        const double phiL = phi_list[k][idL];
        const double phiR = phi_list[k][idR];

        const int neg_mat = tracked_interfaces[k].negative_material_id;
        const bool tracked_material_on_face = (neg_mat == matL) || (neg_mat == matR);
        const bool brackets_or_touches_interface =
            ((phiL <= phi_tol) && (phiR >= -phi_tol)) ||
            ((phiR <= phi_tol) && (phiL >= -phi_tol));

        if (!tracked_material_on_face || !brackets_or_touches_interface) {
            continue;
        }

        const double score = std::abs(phiL) + std::abs(phiR);

        if (score < best_score) {
            best_score = score;
            best_k = k;
        }
    }

    return best_k;
}


// [3] Accumulate interface-normal speed from one extracted line
template<int DIM, typename EOS>
inline void accumulate_interface_normal_speed_line(
    int dir,
    const std::vector<Conserved<DIM>>& U_line,
    const std::vector<int>& mat_line,
    const std::vector<int>& id_line,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<TrackedInterface>& tracked_interfaces,
    const std::vector<std::vector<std::array<double, DIM>>>& normals_list,
    const SolverContext<DIM>& ctx,
    double dt,
    std::vector<std::vector<double>>& Vn_sum,
    std::vector<std::vector<int>>& Vn_count
)
{
    const int L = static_cast<int>(U_line.size());
    const double phi_tol = 1e-3 * ctx.dx[dir];

    if (L < 2) {
        return;
    }

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
            tracked_interfaces,
            phi_tol
        );

        if (k < 0) {
            throw std::runtime_error(
                "accumulate_interface_normal_speed_line: failed to identify tracked interface for mixed-material face"
            );
        }

        const EOSParams& paramsL = ctx.material_params[matL];
        const EOSParams& paramsR = ctx.material_params[matR];

        const std::array<double, DIM> n_flux = axis_normal<DIM>(dir);
        const std::array<double, DIM> n_transport =
            build_face_normal<DIM>(normals_list[k], idL, idR, dir);

        Conserved<DIM> UL_real = U_line[i];
        Conserved<DIM> UR_real = U_line[i + 1];

        enforce_positive_conserved<DIM, EOS>(UL_real, paramsL);
        enforce_positive_conserved<DIM, EOS>(UR_real, paramsR);

        const double Vn_flux = interface_normal_speed_mcrs<DIM, EOS, EOS>(
            UL_real,
            UR_real,
            n_flux,
            paramsL,
            paramsR
        );
        const double Vn_face = Vn_flux * dot<DIM>(n_transport, n_flux);

        Vn_sum[k][idL] += Vn_face;
        Vn_sum[k][idR] += Vn_face;

        Vn_count[k][idL] += 1;
        Vn_count[k][idR] += 1;
    }
}

// [4] Recursive accumulation over all transverse index combinations
template<int DIM, typename EOS, int DIR>
inline void accumulate_interface_normal_speed_recursive(
    int depth,
    std::array<int, DIM>& base_idx,
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<TrackedInterface>& tracked_interfaces,
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
            tracked_interfaces,
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
            tracked_interfaces,
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
            tracked_interfaces,
            normals_list,
            ctx,
            dt,
            Vn_sum,
            Vn_count
        );
    }
}

/*
[5] Build per-level-set normal speed fields

    Starts from projected bulk velocity as an extension field and then
    overwrites interface-adjacent cells with HLLC/MCRS-consistent interface speed.
*/
template<int DIM, typename EOS>
inline std::vector<std::vector<double>> build_interface_normal_speed_fields(
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::vector<std::array<double, DIM>>& vel,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<TrackedInterface>& tracked_interfaces,
    const std::vector<std::vector<std::array<double, DIM>>>& normals_list,
    const SolverContext<DIM>& ctx,
    double dt
)
{
    const int Ntot = static_cast<int>(U.size());
    const int nphi = static_cast<int>(phi_list.size());

    std::vector<std::vector<double>> Vn_fields(nphi, std::vector<double>(Ntot, 0.0));

    // [5.1] Start from projected bulk velocity as a global extension field
    for (int k = 0; k < nphi; ++k) {
        Vn_fields[k] = build_projected_normal_speed_field<DIM>(vel, normals_list[k]);
    }

    // [5.2] Accumulate interface-consistent speed where interfaces exist
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
            tracked_interfaces,
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
            tracked_interfaces,
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
            tracked_interfaces,
            normals_list,
            ctx,
            dt,
            Vn_sum,
            Vn_count
        );
    }

    // [5.3] Overwrite interface-adjacent cells with averaged interface speed
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < nphi; ++k) {
        for (int id = 0; id < Ntot; ++id) {
            if (Vn_count[k][id] > 0) {
                Vn_fields[k][id] = Vn_sum[k][id] / static_cast<double>(Vn_count[k][id]);
            }
        }
    }

    return Vn_fields;
}
