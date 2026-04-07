#pragma once

#include <vector>
#include <array>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "src/euler/state.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/solver/cfl.hpp"
#include "src/euler/solver/solver_context.hpp"
#include "src/euler/thermo_compute.hpp"
#include "src/euler/level_set/level_set.hpp"

// Split modules
#include "src/euler/solver/advance/geometry.hpp"
#include "src/euler/solver/advance/line_ops.hpp"
#include "src/euler/solver/advance/interface_speed.hpp"
#include "src/euler/solver/advance/sweep_core.hpp"
#include "src/euler/solver/advance/boundary.hpp"



// [0] Step Result
template<int DIM>
struct StepResult {
    std::vector<Conserved<DIM>> U_new;
    std::vector<std::vector<double>> phi_list_new;
    double dt;
};



// [1] Velocity field
template<int DIM, typename EOS>
inline std::vector<std::array<double, DIM>> build_velocity_field(
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::vector<EOSParams>& material_params
)
{
    const int Ntot = static_cast<int>(U.size());

    std::vector<std::array<double, DIM>> vel(Ntot);

    #pragma omp parallel for
    for (int i = 0; i < Ntot; ++i) {

        const int mat = material_id.empty() ? 0 : material_id[i];

        const EOSParams& params = material_params.empty()
            ? EOSParams{}
            : material_params[mat];

        const ThermoState<DIM> T = compute_thermo<DIM, EOS>(U[i], params);

        vel[i] = T.vel;
    }

    return vel;
}



// [2] Advance One Step
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

    
    // [2.1] Material map
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

    
    // [2.2] CFL timestep
    const double dt = compute_dt_cfl_materials<DIM, EOS>(
        U,
        material_id_current,
        ctx.material_params,
        ctx.dx,
        ctx.cfl,
        ctx.dt_max
    );

    
    // [2.3] Working level sets
    std::vector<std::vector<double>> phi_list_work = ctx.phi_list;

    
    // [2.4] Normals 
    std::vector<std::vector<std::array<double, DIM>>> normals_current(
        ctx.phi_list.size()
    );

    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(ctx.phi_list.size()); ++k) {

        if constexpr (DIM == 1) {
            if (ctx.use_axis_normals_in_1d) {
                normals_current[k].assign(Ntot, axis_normal<DIM>(0));
            } else {
                normals_current[k] =
                    compute_normals<DIM>(
                        ctx.phi_list[k],
                        ctx.level_set_grid
                    );
            }
        } else {
            normals_current[k] =
                compute_normals<DIM>(
                    ctx.phi_list[k],
                    ctx.level_set_grid
                );
        }
    }

    
    // [2.5] Level-set advection
    if (ctx.advect_level_set && !phi_list_work.empty()) {

        const auto vel = build_velocity_field<DIM, EOS>(
            U,
            material_id_current,
            ctx.material_params
        );

        const auto Vn_fields =
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

        #pragma omp parallel for
        for (int k = 0; k < static_cast<int>(phi_list_work.size()); ++k) {

            phi_list_work[k] =
                advect_phi_normal_speed<DIM>(
                    ctx.phi_list[k],
                    Vn_fields[k],
                    ctx.level_set_grid,
                    dt
                );
        }
    }

    
    // [2.6] Reinitialisation
    static int step_counter = 0;
    step_counter++;

    if (ctx.reinit_enabled &&
        (step_counter % ctx.reinit_frequency == 0))
    {
        for (int k = 0; k < static_cast<int>(phi_list_work.size()); ++k) {

            phi_list_work[k] =
                reinitialise_phi<DIM>(
                    phi_list_work[k],
                    ctx.level_set_grid,
                    ctx.reinit_iterations
                );
        }
    }

    
    // [2.7] Updated material map
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

    
    // [2.8] Normals 
    std::vector<std::vector<std::array<double, DIM>>> normals_list(
        phi_list_work.size()
    );

    for (int k = 0; k < static_cast<int>(phi_list_work.size()); ++k) {

        if constexpr (DIM == 1) {
            if (ctx.use_axis_normals_in_1d) {
                normals_list[k].assign(Ntot, axis_normal<DIM>(0));
            } else {
                normals_list[k] = compute_normals<DIM>(
                        phi_list_work[k],
                        ctx.level_set_grid
                    );
            }
        } else {
            normals_list[k] =
                compute_normals<DIM>(
                    phi_list_work[k],
                    ctx.level_set_grid
                );
        }
    }

    
    // [2.9] Directional splitting
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