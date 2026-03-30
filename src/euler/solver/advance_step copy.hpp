#pragma once

#include <vector>
#include <array>
#include <stdexcept>

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
#include "src/euler/level_set/level_set_core.hpp"

#include "src/euler/grid/grid_utils.hpp"
#include "src/math/vector_ops.hpp"


// [0] Step Result
template<int DIM>
struct StepResult {
    std::vector<Conserved<DIM>> U_new;
    std::vector<double> phi_new;
    double dt;
};


// [1] Build Velocity Field from Conserved State
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


// [4] Extract one grid line in direction DIR
template<int DIM, int DIR>
inline void extract_line(
    const std::vector<Conserved<DIM>>& U,
    const std::vector<double>& phi,
    const std::vector<int>& material_id,
    const std::array<int, DIM>& N,
    const std::array<int, DIM>& base_idx,
    std::vector<Conserved<DIM>>& U_line,
    std::vector<double>& phi_line,
    std::vector<int>& mat_line,
    std::vector<int>& id_line
)
{
    const int L = N[DIR];

    U_line.resize(L);
    phi_line.resize(L);
    mat_line.resize(L);
    id_line.resize(L);

    for (int i = 0; i < L; ++i) {
        std::array<int, DIM> idx = base_idx;
        idx[DIR] = i;

        const int id = flatten_index<DIM>(idx, N);

        U_line[i] = U[id];
        phi_line[i] = phi[id];
        mat_line[i] = material_id.empty() ? 0 : material_id[id];
        id_line[i] = id;
    }
}


// [5] Dispatch MUSCL reconstruction by sweep direction
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


// [6] Advance one extracted line by one split sweep
template<int DIM, typename EOS>
inline void advance_line(
    int dir,
    const std::vector<Conserved<DIM>>& U_line,
    const std::vector<double>& phi_line,
    const std::vector<int>& mat_line,
    const std::vector<int>& id_line,
    const std::vector<std::array<double, DIM>>& normals,
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

    std::vector<Conserved<DIM>> F(L - 1);

    // [6.1] Compute interface fluxes
    for (int i = 0; i < L - 1; ++i) {
        const int idL = id_line[i];
        const int idR = id_line[i + 1];

        const int matL = mat_line[i];
        const int matR = mat_line[i + 1];

        const EOSParams& paramsL = ctx.material_params[matL];
        const EOSParams& paramsR = ctx.material_params[matR];

        const double phiL = phi_line[i];
        const double phiR = phi_line[i + 1];

        if (!is_interface_face(phiL, phiR)) {
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
            const std::array<double, DIM> n_face =
                build_face_normal<DIM>(normals, idL, idR, dir);

            const auto ghosts =
                ghost_states_normal<DIM, EOS, EOS>(
                    UL_face[i],
                    UR_face[i],
                    n_face,
                    paramsL,
                    paramsR
                );

            F[i] = hllc_flux_normal<DIM, EOS>(
                ghosts.first,
                ghosts.second,
                n_face,
                paramsL,
                paramsR
            );
        }
    }

    // [6.2] Conservative update on interior cells of the line
    const double factor = dt / ctx.dx[dir];

    for (int i = 1; i < L - 1; ++i) {
        Conserved<DIM> U_next = U_line[i] - factor * (F[i] - F[i - 1]);

        const EOSParams& params_i = ctx.material_params[mat_line[i]];

        enforce_positive_conserved<DIM, EOS>(U_next, params_i);

        U_out[id_line[i]] = U_next;
    }

    // [6.3] Simple boundary copy on the line
    if (L >= 2) {
        U_out[id_line.front()] = U_out[id_line[1]];
        U_out[id_line.back()] = U_out[id_line[L - 2]];
    }
}


// [7] Recursive line sweep over all transverse index combinations
template<int DIM, typename EOS, int DIR>
inline void sweep_lines_recursive(
    int depth,
    std::array<int, DIM>& base_idx,
    const std::vector<Conserved<DIM>>& U_in,
    const std::vector<double>& phi,
    const std::vector<std::array<double, DIM>>& normals,
    const SolverContext<DIM>& ctx,
    double dt,
    std::vector<Conserved<DIM>>& U_out
)
{
    if (depth == DIM) {
        std::vector<Conserved<DIM>> U_line;
        std::vector<double> phi_line;
        std::vector<int> mat_line;
        std::vector<int> id_line;

        extract_line<DIM, DIR>(
            U_in,
            phi,
            ctx.material_id,
            ctx.N,
            base_idx,
            U_line,
            phi_line,
            mat_line,
            id_line
        );

        advance_line<DIM, EOS>(
            DIR,
            U_line,
            phi_line,
            mat_line,
            id_line,
            normals,
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
            phi,
            normals,
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
            phi,
            normals,
            ctx,
            dt,
            U_out
        );
    }
}


// [8] Dispatch one full split sweep direction
template<int DIM, typename EOS>
inline void sweep_direction_dispatch(
    int dir,
    const std::vector<Conserved<DIM>>& U_in,
    const std::vector<double>& phi,
    const std::vector<std::array<double, DIM>>& normals,
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
                phi,
                normals,
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
                phi,
                normals,
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
                phi,
                normals,
                ctx,
                dt,
                U_out
            );
            return;
        }
    }

    throw std::runtime_error("sweep_direction_dispatch: invalid direction");
}


// [9] Advance One Step (Dimension-Agnostic, Split Scheme)
template<int DIM, typename EOS>
inline StepResult<DIM> advance_one_step(
    const std::vector<Conserved<DIM>>& U,
    const std::vector<double>& phi,
    const SolverContext<DIM>& ctx
)
{
    const int Ntot = static_cast<int>(U.size());

    if (Ntot == 0) {
        throw std::runtime_error("advance_one_step: empty state");
    }

    if (static_cast<int>(phi.size()) != Ntot) {
        throw std::runtime_error("advance_one_step: phi size mismatch");
    }

    ctx.validate(Ntot);

    // [9.1] Compute timestep
    const double dt = compute_dt_cfl_materials<DIM, EOS>(
        U,
        ctx.material_id,
        ctx.material_params,
        ctx.dx,
        ctx.cfl,
        ctx.dt_max
    );

    std::vector<Conserved<DIM>> U_stage = U;

    // [9.2] Geometry for interface treatment
    LevelSetGrid<DIM> grid{ctx.N, ctx.dx};

    std::vector<double> phi_geom = reinitialise_phi<DIM>(
        phi,
        grid,
        ctx.reinit_iterations
    );

    const std::vector<std::array<double, DIM>> normals =
        compute_normals<DIM>(phi_geom, grid);

    // [9.3] Directional sweeps
    for (int dir = 0; dir < DIM; ++dir) {
        std::vector<Conserved<DIM>> U_next = U_stage;

        sweep_direction_dispatch<DIM, EOS>(
            dir,
            U_stage,
            phi_geom,
            normals,
            ctx,
            dt,
            U_next
        );

        U_stage.swap(U_next);
    }

    // [9.4] Level set update
    const std::vector<std::array<double, DIM>> vel =
        build_velocity_field<DIM, EOS>(
            U_stage,
            ctx.material_id,
            ctx.material_params
        );

    std::vector<double> phi_adv =
        advect_phi<DIM>(phi, vel, grid, dt);

    std::vector<double> phi_new =
        reinitialise_phi<DIM>(
            phi_adv,
            grid,
            ctx.reinit_iterations
        );

    return {U_stage, phi_new, dt};
}