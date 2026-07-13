#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/app/dimension.hpp"
#include "src/core/phase_timings.hpp"
#include "src/euler/conservative.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/primitives.hpp"
#include "src/euler/state.hpp"
#include "src/euler/thermo_compute.hpp"
#include "src/io/config.hpp"
#include "src/io/config_loader.hpp"
#include "src/setup/initial_conditions.hpp"
#include "src/sim/solver/advance/sweep_core.hpp"
#include "src/sim/solver/solver_context.hpp"

using EOS = IdealGasEOS;

namespace {

constexpr int ghost_width = 2;

struct Slab {
    int start = 0;
    int owned = 0;
};

struct MpiRuntime {
    int rank = 0;
    int size = 1;
};

template<int DIM>
struct CellInitialState {
    Primitive<DIM> primitive{};
    int material_id = 0;
};

BoundaryConditionType parse_boundary_condition(const std::string& value)
{
    if (value == "reflective" || value == "reflection") {
        return BoundaryConditionType::reflective;
    }
    if (value == "nonreflective") {
        return BoundaryConditionType::nonreflective;
    }
    return BoundaryConditionType::transmissive;
}

std::vector<int> parse_resolution(const std::string& text, int expected_dim)
{
    std::vector<int> values;
    std::string token;
    std::stringstream stream(text);

    while (std::getline(stream, token, 'x')) {
        if (!token.empty()) {
            values.push_back(std::stoi(token));
        }
    }

    if (static_cast<int>(values.size()) != expected_dim) {
        throw std::runtime_error("resolution override dimension does not match APP_DIM");
    }

    return values;
}

Slab compute_slab(int global_cells, int rank, int size)
{
    const int base = global_cells / size;
    const int remainder = global_cells % size;

    Slab slab{};
    slab.owned = base + (rank < remainder ? 1 : 0);
    slab.start = rank * base + std::min(rank, remainder);
    return slab;
}

template<int DIM>
int total_cells(const std::array<int, DIM>& N)
{
    int total = 1;
    for (int d = 0; d < DIM; ++d) {
        total *= N[d];
    }
    return total;
}

template<int DIM>
int flat_id(const std::array<int, DIM>& idx, const std::array<int, DIM>& N)
{
    int id = idx[0];
    int stride = N[0];
    for (int d = 1; d < DIM; ++d) {
        id += idx[d] * stride;
        stride *= N[d];
    }
    return id;
}

template<int DIM>
std::array<int, DIM> unflatten_id(int linear, const std::array<int, DIM>& N)
{
    std::array<int, DIM> idx{};
    int stride = 1;
    for (int d = 0; d < DIM; ++d) {
        idx[d] = (linear / stride) % N[d];
        stride *= N[d];
    }
    return idx;
}

template<int DIM>
int plane_cell_count(const std::array<int, DIM>& N, int axis)
{
    int count = 1;
    for (int d = 0; d < DIM; ++d) {
        if (d != axis) {
            count *= N[d];
        }
    }
    return count;
}

template<int DIM>
std::array<int, DIM> plane_index(
    int plane_linear,
    int axis,
    int axis_value,
    const std::array<int, DIM>& N)
{
    std::array<int, DIM> idx{};
    idx[axis] = axis_value;

    int tmp = plane_linear;
    for (int d = 0; d < DIM; ++d) {
        if (d == axis) {
            continue;
        }
        idx[d] = tmp % N[d];
        tmp /= N[d];
    }

    return idx;
}

std::vector<EOSParams> build_material_params(const std::vector<MaterialConfig>& materials)
{
    std::vector<EOSParams> params(materials.size());
    for (const auto& material : materials) {
        if (material.id < 0 || material.id >= static_cast<int>(materials.size())) {
            throw std::runtime_error("material ids must be contiguous for MPI SM");
        }
        params[material.id] = build_eos_params_from_material(materials, material.id);
    }
    return params;
}

template<int DIM>
CellInitialState<DIM> initial_state_at_point(
    const Config<DIM>& cfg,
    const std::array<double, DIM>& x)
{
    if (cfg.initial_condition == "regions" ||
        cfg.initial_condition == "planar_regions") {
        if (cfg.regions.empty()) {
            throw std::runtime_error("No regions defined");
        }

        const bool use_planar_regions = cfg.initial_condition == "planar_regions";
        const auto planar_normal = use_planar_regions
            ? normalised_planar_normal<DIM>(cfg)
            : std::array<double, DIM>{};

        for (const auto& region : cfg.regions) {
            const bool inside_region = use_planar_regions
                ? point_in_planar_region<DIM>(x, region, planar_normal)
                : point_in_region<DIM>(x, region);

            if (!inside_region) {
                continue;
            }

            Primitive<DIM> primitive{};
            primitive.rho = region.rho;
            primitive.vel = region.vel;
            primitive.p = region.p;
            return {primitive, region.material_id};
        }

        throw std::runtime_error("Cell not covered by any region");
    }

    if (cfg.initial_condition == "explosion" ||
        cfg.initial_condition == "double_explosion") {
        if (cfg.explosion_radius <= 0.0) {
            throw std::runtime_error("Explosion radius must be positive");
        }

        bool inside = false;
        if (cfg.initial_condition == "double_explosion") {
            if (cfg.explosion_centers.empty()) {
                throw std::runtime_error("double_explosion requires explosion_centers");
            }
            for (const auto& center : cfg.explosion_centers) {
                if (point_in_explosion_region<DIM>(x, center, cfg.explosion_radius)) {
                    inside = true;
                    break;
                }
            }
        }
        else {
            inside = point_in_explosion_region<DIM>(
                x,
                cfg.explosion_center,
                cfg.explosion_radius
            );
        }

        Primitive<DIM> primitive{};
        if (inside) {
            primitive.rho = cfg.rho_in;
            primitive.vel = cfg.vel_in;
            primitive.p = cfg.p_in;
            return {primitive, cfg.material_in};
        }

        primitive.rho = cfg.rho_out;
        primitive.vel = cfg.vel_out;
        primitive.p = cfg.p_out;
        return {primitive, cfg.material_out};
    }

    if (cfg.initial_condition == "shock_bubble") {
        if (cfg.shock_axis < 0 || cfg.shock_axis >= DIM) {
            throw std::runtime_error("Invalid shock axis");
        }
        if (cfg.bubble_radius <= 0.0) {
            throw std::runtime_error("Bubble radius must be positive");
        }

        Primitive<DIM> primitive{};
        if (point_in_explosion_region<DIM>(x, cfg.bubble_center, cfg.bubble_radius)) {
            primitive.rho = cfg.rho_bubble;
            primitive.vel = cfg.vel_bubble;
            primitive.p = cfg.p_bubble;
            return {primitive, cfg.material_bubble};
        }

        if (x[cfg.shock_axis] < cfg.shock_position) {
            primitive.rho = cfg.rho_left;
            primitive.vel = cfg.vel_left;
            primitive.p = cfg.p_left;
            return {primitive, cfg.material_left};
        }

        primitive.rho = cfg.rho_right;
        primitive.vel = cfg.vel_right;
        primitive.p = cfg.p_right;
        return {primitive, cfg.material_right};
    }

    if (cfg.initial_condition == "coated_shock_bubble") {
        if (cfg.shock_axis < 0 || cfg.shock_axis >= DIM) {
            throw std::runtime_error("Invalid shock axis");
        }
        if (cfg.bubble_radius <= 0.0 || cfg.film_radius <= cfg.bubble_radius) {
            throw std::runtime_error("Invalid coated_shock_bubble radii");
        }

        double radius_sq = 0.0;
        for (int d = 0; d < DIM; ++d) {
            const double diff = x[d] - cfg.bubble_center[d];
            radius_sq += diff * diff;
        }
        const double radius = std::sqrt(radius_sq);

        Primitive<DIM> primitive{};
        if (radius <= cfg.bubble_radius) {
            primitive.rho = cfg.rho_bubble;
            primitive.vel = cfg.vel_bubble;
            primitive.p = cfg.p_bubble;
            return {primitive, cfg.material_bubble};
        }
        if (radius <= cfg.film_radius) {
            primitive.rho = cfg.rho_film;
            primitive.vel = cfg.vel_film;
            primitive.p = cfg.p_film;
            return {primitive, cfg.material_film};
        }
        if (x[cfg.shock_axis] < cfg.shock_position) {
            primitive.rho = cfg.rho_left;
            primitive.vel = cfg.vel_left;
            primitive.p = cfg.p_left;
            return {primitive, cfg.material_left};
        }

        primitive.rho = cfg.rho_right;
        primitive.vel = cfg.vel_right;
        primitive.p = cfg.p_right;
        return {primitive, cfg.material_right};
    }

    throw std::runtime_error("Unsupported initial_condition: " + cfg.initial_condition);
}

template<int DIM>
void initialise_local_slab(
    std::vector<Conserved<DIM>>& U,
    std::vector<int>& material_id,
    const Config<DIM>& cfg,
    const std::vector<EOSParams>& material_params,
    const std::array<int, DIM>& local_N,
    const std::array<double, DIM>& dx,
    const Slab& slab,
    int axis)
{
    U.assign(total_cells<DIM>(local_N), Conserved<DIM>{});
    material_id.assign(U.size(), 0);

    for (int local_axis = ghost_width; local_axis < ghost_width + slab.owned; ++local_axis) {
        const int global_axis = slab.start + local_axis - ghost_width;

        for (int plane_linear = 0; plane_linear < plane_cell_count<DIM>(local_N, axis); ++plane_linear) {
            std::array<int, DIM> idx =
                plane_index<DIM>(plane_linear, axis, local_axis, local_N);

            std::array<double, DIM> centre{};
            for (int d = 0; d < DIM; ++d) {
                const int global_index = (d == axis) ? global_axis : idx[d];
                centre[d] = cfg.domain_min[d] + (static_cast<double>(global_index) + 0.5) * dx[d];
            }

            const CellInitialState<DIM> state = initial_state_at_point<DIM>(cfg, centre);
            if (state.material_id < 0 ||
                state.material_id >= static_cast<int>(material_params.size())) {
                throw std::runtime_error("Invalid material id in MPI local initializer");
            }

            const int id = flat_id<DIM>(idx, local_N);
            U[id] = prim_to_cons<DIM, EOS>(
                state.primitive,
                material_params[state.material_id]
            );
            material_id[id] = state.material_id;
        }
    }
}

template<int DIM>
void pack_axis_plane(
    const std::vector<Conserved<DIM>>& U,
    const std::array<int, DIM>& N,
    int axis,
    int axis_value,
    std::vector<double>& buffer)
{
    const int values_per_cell = DIM + 2;
    const int cells = plane_cell_count<DIM>(N, axis);
    buffer.assign(static_cast<std::size_t>(cells) * values_per_cell, 0.0);

    int out = 0;
    for (int plane_linear = 0; plane_linear < cells; ++plane_linear) {
        const auto idx = plane_index<DIM>(plane_linear, axis, axis_value, N);
        const auto& state = U[flat_id<DIM>(idx, N)];
        buffer[out++] = state.rho;
        for (int d = 0; d < DIM; ++d) {
            buffer[out++] = state.mom[d];
        }
        buffer[out++] = state.E;
    }
}

template<int DIM>
void unpack_axis_plane(
    std::vector<Conserved<DIM>>& U,
    const std::array<int, DIM>& N,
    int axis,
    int axis_value,
    const std::vector<double>& buffer)
{
    const int cells = plane_cell_count<DIM>(N, axis);

    int in = 0;
    for (int plane_linear = 0; plane_linear < cells; ++plane_linear) {
        const auto idx = plane_index<DIM>(plane_linear, axis, axis_value, N);
        auto& state = U[flat_id<DIM>(idx, N)];
        state.rho = buffer[in++];
        for (int d = 0; d < DIM; ++d) {
            state.mom[d] = buffer[in++];
        }
        state.E = buffer[in++];
    }
}

template<int DIM>
void apply_boundary_copy(
    std::vector<Conserved<DIM>>& U,
    const std::array<int, DIM>& N,
    const std::array<int, DIM>& dst,
    const std::array<int, DIM>& src,
    int axis,
    BoundaryConditionType bc)
{
    Conserved<DIM> state = U[flat_id<DIM>(src, N)];
    if (bc == BoundaryConditionType::reflective) {
        state.mom[axis] *= -1.0;
    }
    U[flat_id<DIM>(dst, N)] = state;
}

template<int DIM>
void apply_non_decomposed_boundaries(
    std::vector<Conserved<DIM>>& U,
    const std::array<int, DIM>& N,
    const SolverContext<DIM>& ctx,
    int decomp_axis)
{
    const int cells = total_cells<DIM>(N);

    for (int axis = 0; axis < DIM; ++axis) {
        if (axis == decomp_axis) {
            continue;
        }

        for (int linear = 0; linear < cells; ++linear) {
            const auto idx = unflatten_id<DIM>(linear, N);

            if (idx[axis] == 0) {
                auto src = idx;
                src[axis] = 1;
                apply_boundary_copy<DIM>(U, N, idx, src, axis, ctx.bc_lo[axis]);
            }
            else if (idx[axis] == N[axis] - 1) {
                auto src = idx;
                src[axis] = N[axis] - 2;
                apply_boundary_copy<DIM>(U, N, idx, src, axis, ctx.bc_hi[axis]);
            }
        }
    }
}

template<int DIM>
void apply_decomposed_physical_boundaries(
    std::vector<Conserved<DIM>>& U,
    const std::array<int, DIM>& N,
    const SolverContext<DIM>& ctx,
    const MpiRuntime& mpi,
    int axis)
{
    const int cells = plane_cell_count<DIM>(N, axis);
    const int owned_lo = ghost_width;
    const int owned_hi = ghost_width + (N[axis] - 2 * ghost_width) - 1;

    if (mpi.rank == 0) {
        for (int plane_linear = 0; plane_linear < cells; ++plane_linear) {
            auto dst = plane_index<DIM>(plane_linear, axis, owned_lo, N);
            auto src = dst;
            src[axis] = owned_lo + 1;
            apply_boundary_copy<DIM>(U, N, dst, src, axis, ctx.bc_lo[axis]);

            for (int g = 0; g < ghost_width; ++g) {
                auto ghost = dst;
                ghost[axis] = g;
                apply_boundary_copy<DIM>(U, N, ghost, dst, axis, ctx.bc_lo[axis]);
            }
        }
    }

    if (mpi.rank == mpi.size - 1) {
        for (int plane_linear = 0; plane_linear < cells; ++plane_linear) {
            auto dst = plane_index<DIM>(plane_linear, axis, owned_hi, N);
            auto src = dst;
            src[axis] = owned_hi - 1;
            apply_boundary_copy<DIM>(U, N, dst, src, axis, ctx.bc_hi[axis]);

            for (int g = 0; g < ghost_width; ++g) {
                auto ghost = dst;
                ghost[axis] = owned_hi + 1 + g;
                apply_boundary_copy<DIM>(U, N, ghost, dst, axis, ctx.bc_hi[axis]);
            }
        }
    }
}

template<int DIM>
void exchange_axis_ghosts(
    std::vector<Conserved<DIM>>& U,
    const std::array<int, DIM>& N,
    const MpiRuntime& mpi,
    int axis)
{
    const int left = mpi.rank - 1;
    const int right = mpi.rank + 1;
    const int owned_lo = ghost_width;
    const int owned_hi = ghost_width + (N[axis] - 2 * ghost_width) - 1;
    const int plane_values = plane_cell_count<DIM>(N, axis) * (DIM + 2);

    std::vector<double> send_left;
    std::vector<double> recv_left(plane_values);
    std::vector<double> send_right;
    std::vector<double> recv_right(plane_values);

    for (int offset = 0; offset < ghost_width; ++offset) {
        if (left >= 0) {
            pack_axis_plane<DIM>(U, N, axis, owned_lo + offset, send_left);
            MPI_Sendrecv(
                send_left.data(), plane_values, MPI_DOUBLE, left, 101 + offset,
                recv_left.data(), plane_values, MPI_DOUBLE, left, 201 + offset,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
            unpack_axis_plane<DIM>(U, N, axis, owned_lo - 1 - offset, recv_left);
        }

        if (right < mpi.size) {
            pack_axis_plane<DIM>(U, N, axis, owned_hi - offset, send_right);
            MPI_Sendrecv(
                send_right.data(), plane_values, MPI_DOUBLE, right, 201 + offset,
                recv_right.data(), plane_values, MPI_DOUBLE, right, 101 + offset,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
            unpack_axis_plane<DIM>(U, N, axis, owned_hi + 1 + offset, recv_right);
        }
    }
}

template<int DIM>
void refresh_boundaries(
    std::vector<Conserved<DIM>>& U,
    const std::array<int, DIM>& N,
    const SolverContext<DIM>& ctx,
    const MpiRuntime& mpi,
    int axis)
{
    apply_non_decomposed_boundaries<DIM>(U, N, ctx, axis);
    exchange_axis_ghosts<DIM>(U, N, mpi, axis);
    apply_decomposed_physical_boundaries<DIM>(U, N, ctx, mpi, axis);
}

template<int DIM>
double compute_local_owned_dt(
    const std::vector<Conserved<DIM>>& U,
    const std::array<int, DIM>& N,
    const std::array<double, DIM>& dx,
    const EOSParams& params,
    double cfl,
    double dt_max,
    int axis)
{
    double dt = dt_max;
    const int cells = total_cells<DIM>(N);

    #pragma omp parallel for reduction(min:dt)
    for (int linear = 0; linear < cells; ++linear) {
        const auto idx = unflatten_id<DIM>(linear, N);
        if (idx[axis] < ghost_width || idx[axis] >= N[axis] - ghost_width) {
            continue;
        }

        const auto& state = U[linear];
        const ThermoState<DIM> thermo = compute_thermo<DIM, EOS>(state, params);
        for (int d = 0; d < DIM; ++d) {
            const double wave_speed = std::abs(thermo.vel[d]) + thermo.c;
            if (wave_speed > 1.0e-14) {
                dt = std::min(dt, cfl * dx[d] / wave_speed);
            }
        }
    }

    return dt;
}

template<int DIM>
void write_rank_csv(
    const std::filesystem::path& path,
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::array<int, DIM>& local_N,
    const std::array<double, DIM>& domain_min,
    const std::array<double, DIM>& dx,
    const Slab& slab,
    int axis,
    const std::vector<EOSParams>& material_params)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("cannot write " + path.string());
    }

    for (int d = 0; d < DIM; ++d) {
        file << "x" << d << ",";
    }
    file << "rho,";
    for (int d = 0; d < DIM; ++d) {
        file << "u" << d << ",";
    }
    file << "p,e,mat\n";
    file << std::setprecision(16);

    for (int local_axis = ghost_width; local_axis < ghost_width + slab.owned; ++local_axis) {
        const int global_axis = slab.start + local_axis - ghost_width;

        for (int plane_linear = 0; plane_linear < plane_cell_count<DIM>(local_N, axis); ++plane_linear) {
            const auto idx = plane_index<DIM>(plane_linear, axis, local_axis, local_N);
            const int id = flat_id<DIM>(idx, local_N);
            const int mat = material_id.empty() ? 0 : material_id[id];
            const Primitive<DIM> primitive =
                cons_to_prim<DIM, EOS>(U[id], material_params[mat]);
            double kinetic = 0.0;
            for (int d = 0; d < DIM; ++d) {
                kinetic += U[id].mom[d] * U[id].mom[d];
            }
            const double specific_internal_energy =
                (U[id].E - 0.5 * kinetic / U[id].rho) / U[id].rho;

            for (int d = 0; d < DIM; ++d) {
                const int global_index = (d == axis) ? global_axis : idx[d];
                const double coordinate =
                    domain_min[d] + (static_cast<double>(global_index) + 0.5) * dx[d];
                file << coordinate << ",";
            }

            file << primitive.rho << ",";
            for (int d = 0; d < DIM; ++d) {
                file << primitive.vel[d] << ",";
            }
            file << primitive.p << "," << specific_internal_energy << "," << mat << "\n";
        }
    }
}

template<int DIM>
void write_runtime_report(
    const std::filesystem::path& path,
    const MpiRuntime& mpi,
    const std::array<int, DIM>& N,
    int steps,
    double final_time,
    double wall_seconds,
    const SolverPhaseTimings& timings)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("cannot write " + path.string());
    }

    const double cells = static_cast<double>(total_cells<DIM>(N));
    const double cell_updates = cells * static_cast<double>(steps);
    file << "interface_method = SM_MPI\n";
    file << "dimension = " << DIM << "\n";
    file << "mpi_ranks = " << mpi.size << "\n";
    file << "N = [";
    for (int d = 0; d < DIM; ++d) {
        if (d > 0) {
            file << ", ";
        }
        file << N[d];
    }
    file << "]\n";
    file << "steps = " << steps << "\n";
    file << "cells = " << cells << "\n";
    file << "final_time = " << final_time << "\n";
    file << "wall_time_seconds = " << wall_seconds << "\n";
    file << "cell_updates_per_second = " << (cell_updates / std::max(wall_seconds, 1.0e-30)) << "\n";
    file << "cost_per_cell_update_seconds = " << (wall_seconds / std::max(cell_updates, 1.0)) << "\n";
    file << "phase_cfl_seconds = " << timings.cfl_seconds << "\n";
    file << "phase_sweep_seconds = " << timings.sweep_seconds << "\n";
    file << "phase_boundary_seconds = " << timings.boundary_seconds << "\n";
    file << "phase_output_seconds = " << timings.output_seconds << "\n";
}

template<int DIM>
int run_mpi_sm(int argc, char** argv, const MpiRuntime& mpi)
{
    if (argc < 2) {
        throw std::runtime_error("Usage: ./sm_mpi_main <config_file> [--resolution N[xN...]] [--output-dir path]");
    }

    std::string config_file = argv[1];
    std::string output_dir_override;
    std::vector<int> resolution_override;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--resolution" && i + 1 < argc) {
            resolution_override = parse_resolution(argv[++i], DIM);
        }
        else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir_override = argv[++i];
        }
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    Config<DIM> cfg = load_config<DIM>(config_file);
    const bool timing_only = std::getenv("QUANT_TIMING_ONLY") != nullptr;
    if (cfg.interface_method != "SM") {
        throw std::runtime_error("sm_mpi_main supports only interface_method = SM");
    }
    if (cfg.time_update != "split") {
        throw std::runtime_error("sm_mpi_main currently supports only split time update");
    }
    if (cfg.materials.size() != 1) {
        throw std::runtime_error("sm_mpi_main prototype currently supports exactly one material");
    }

    const auto material_params = build_material_params(cfg.materials);

    if (!resolution_override.empty()) {
        std::array<int, DIM> N{};
        for (int d = 0; d < DIM; ++d) {
            N[d] = resolution_override[d];
        }
        cfg.N_list = {N};
    }

    const int decomp_axis = DIM - 1;

    for (const auto& global_N : cfg.N_list) {
        const Slab slab = compute_slab(global_N[decomp_axis], mpi.rank, mpi.size);
        if (slab.owned <= 0) {
            throw std::runtime_error("number of MPI ranks exceeds decomposed-axis cells");
        }

        std::array<int, DIM> local_N = global_N;
        local_N[decomp_axis] = slab.owned + 2 * ghost_width;

        std::array<double, DIM> dx{};
        for (int d = 0; d < DIM; ++d) {
            dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / global_N[d];
        }

        std::vector<Conserved<DIM>> U;
        std::vector<int> material_id;
        initialise_local_slab<DIM>(
            U,
            material_id,
            cfg,
            material_params,
            local_N,
            dx,
            slab,
            decomp_axis
        );

        SolverContext<DIM> ctx{};
        ctx.N = local_N;
        ctx.dx = dx;
        ctx.cfl = cfg.cfl;
        ctx.time_update = cfg.time_update;
        ctx.material_id = material_id;
        ctx.material_params = material_params;
        ctx.background_material_id = 0;
        for (int d = 0; d < DIM; ++d) {
            ctx.bc_lo[d] = parse_boundary_condition(cfg.bc_lo[d]);
            ctx.bc_hi[d] = parse_boundary_condition(cfg.bc_hi[d]);
        }
        ctx.initialise_level_set_grid();

        refresh_boundaries<DIM>(U, local_N, ctx, mpi, decomp_axis);

        double time = 0.0;
        int step = 0;
        SolverPhaseTimings timings{};
        const auto wall_start = std::chrono::steady_clock::now();

        while (time < cfg.tfinal - 1.0e-14) {
            const double dt_max = cfg.tfinal - time;

            const auto cfl_start = std::chrono::steady_clock::now();
            const double local_dt = compute_local_owned_dt<DIM>(
                U,
                local_N,
                dx,
                material_params[0],
                cfg.cfl,
                dt_max,
                decomp_axis
            );
            double dt = dt_max;
            MPI_Allreduce(&local_dt, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            const auto cfl_end = std::chrono::steady_clock::now();
            timings.cfl_seconds += std::chrono::duration<double>(cfl_end - cfl_start).count();

            std::vector<Conserved<DIM>> stage = U;
            const auto sweep_start = std::chrono::steady_clock::now();

            for (int dir = 0; dir < DIM; ++dir) {
                std::vector<Conserved<DIM>> next(stage.size());
                sweep_direction_dispatch<DIM, EOS>(
                    dir,
                    stage,
                    material_id,
                    {},
                    {},
                    {},
                    ctx,
                    dt,
                    next,
                    nullptr
                );

                const auto boundary_start = std::chrono::steady_clock::now();
                refresh_boundaries<DIM>(next, local_N, ctx, mpi, decomp_axis);
                const auto boundary_end = std::chrono::steady_clock::now();
                timings.boundary_seconds +=
                    std::chrono::duration<double>(boundary_end - boundary_start).count();

                stage.swap(next);
            }

            const auto sweep_end = std::chrono::steady_clock::now();
            timings.sweep_seconds +=
                std::chrono::duration<double>(sweep_end - sweep_start).count();

            U.swap(stage);
            time += dt;
            ++step;

            if (dt <= 0.0) {
                throw std::runtime_error("non-positive MPI timestep");
            }
        }

        const auto wall_end = std::chrono::steady_clock::now();
        const double local_wall =
            std::chrono::duration<double>(wall_end - wall_start).count();
        double max_wall = local_wall;
        MPI_Reduce(&local_wall, &max_wall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        const std::filesystem::path output_root =
            output_dir_override.empty()
                ? std::filesystem::path(cfg.output_dir) / (cfg.output_prefix + "_mpi")
                : std::filesystem::path(output_dir_override);

        if (!timing_only) {
            const auto output_start = std::chrono::steady_clock::now();
            const std::filesystem::path rank_file =
                output_root /
                ("rank_" + std::to_string(mpi.rank) + "_" + cfg.output_prefix + ".csv");
            write_rank_csv<DIM>(
                rank_file,
                U,
                material_id,
                local_N,
                cfg.domain_min,
                dx,
                slab,
                decomp_axis,
                material_params
            );
            const auto output_end = std::chrono::steady_clock::now();
            timings.output_seconds += std::chrono::duration<double>(output_end - output_start).count();
        }

        SolverPhaseTimings reduced_timings{};
        MPI_Reduce(&timings.cfl_seconds, &reduced_timings.cfl_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&timings.sweep_seconds, &reduced_timings.sweep_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&timings.boundary_seconds, &reduced_timings.boundary_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&timings.output_seconds, &reduced_timings.output_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (mpi.rank == 0) {
            write_runtime_report<DIM>(
                output_root / (cfg.output_prefix + "_mpi_runtime.txt"),
                mpi,
                global_N,
                step,
                time,
                max_wall,
                reduced_timings
            );

            std::cout << "MPI SM run completed: dim=" << DIM
                      << " ranks=" << mpi.size
                      << " steps=" << step
                      << " wall=" << max_wall << "s"
                      << " output=" << output_root.string() << "\n";
        }
    }

    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    MpiRuntime mpi{};
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);

    try {
        const int result = run_mpi_sm<DIM_>(argc, argv, mpi);
        MPI_Finalize();
        return result;
    }
    catch (const std::exception& exc) {
        std::cerr << "[rank " << mpi.rank << "] ERROR: " << exc.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return 1;
}
