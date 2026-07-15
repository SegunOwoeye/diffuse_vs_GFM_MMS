#ifndef ENABLE_LEGACY_MPI_APPS
#error "Legacy MPI app entry points are disabled. Use the serial/OpenMP report workflow instead, or define ENABLE_LEGACY_MPI_APPS explicitly for historical MPI experiments."
#endif

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
#include "src/app/material/material_builder.hpp"
#include "src/core/phase_timings.hpp"
#include "src/dim/primitives.hpp"
#include "src/dim/solver/advance/sweep_core.hpp"
#include "src/io/config.hpp"
#include "src/io/config_loader.hpp"
#include "src/sim/solver/solver_context.hpp"
#include "src/setup/dim_initial_conditions.hpp"

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

std::string format_mpi_time_tag(double time)
{
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(6) << time;
    std::string text = stream.str();
    for (char& c : text) {
        if (c == '.') {
            c = 'p';
        }
        else if (c == '+') {
            c = 'p';
        }
        else if (c == '-') {
            c = 'm';
        }
    }
    return "t" + text;
}

int ghost_lo(const MpiRuntime& mpi)
{
    return (mpi.rank > 0) ? ghost_width : 0;
}

int ghost_hi(const MpiRuntime& mpi)
{
    return (mpi.rank + 1 < mpi.size) ? ghost_width : 0;
}

template<int DIM>
int owned_lo_axis(const MpiRuntime& mpi)
{
    return ghost_lo(mpi);
}

template<int DIM>
int owned_hi_axis(
    const std::array<int, DIM>& N,
    const MpiRuntime& mpi,
    int axis)
{
    return N[axis] - ghost_hi(mpi) - 1;
}

BoundaryConditionType parse_boundary_condition(const std::string& value)
{
    if (value == "reflective" || value == "reflection") {
        return BoundaryConditionType::reflective;
    }
    if (value == "nonreflective" || value == "non_reflective") {
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

template<int DIM>
dim::Primitive<DIM> primitive_at_point(
    const Config<DIM>& cfg,
    const std::array<double, DIM>& x,
    const dim::EOSParams& params)
{
    if (cfg.initial_condition == "regions" ||
        cfg.initial_condition == "planar_regions") {
        return dim::diffuse_region_primitive<DIM>(cfg, x, params);
    }
    if (cfg.initial_condition == "shock_bubble") {
        return dim::diffuse_shock_bubble_primitive<DIM>(cfg, x, params);
    }
    if (cfg.initial_condition == "coated_shock_bubble") {
        return dim::diffuse_coated_shock_bubble_primitive<DIM>(cfg, x, params);
    }
    throw std::runtime_error("dim_mpi_main: unsupported initial_condition: " + cfg.initial_condition);
}

template<int DIM>
void initialise_local_slab(
    std::vector<dim::State<DIM>>& U,
    const Config<DIM>& cfg,
    const dim::EOSParams& params,
    const std::array<int, DIM>& local_N,
    const std::array<double, DIM>& dx,
    const Slab& slab,
    const MpiRuntime& mpi,
    int axis)
{
    U.assign(total_cells<DIM>(local_N), dim::make_state<DIM>(params.nmat()));

    const int owned_lo = owned_lo_axis<DIM>(mpi);
    for (int local_axis = owned_lo; local_axis < owned_lo + slab.owned; ++local_axis) {
        const int global_axis = slab.start + local_axis - owned_lo;

        for (int plane_linear = 0; plane_linear < plane_cell_count<DIM>(local_N, axis); ++plane_linear) {
            const auto idx = plane_index<DIM>(plane_linear, axis, local_axis, local_N);

            std::array<double, DIM> centre{};
            for (int d = 0; d < DIM; ++d) {
                const int global_index = (d == axis) ? global_axis : idx[d];
                centre[d] = cfg.domain_min[d] + (static_cast<double>(global_index) + 0.5) * dx[d];
            }

            const int id = flat_id<DIM>(idx, local_N);
            U[id] = dim::prim_to_cons<DIM>(
                primitive_at_point<DIM>(cfg, centre, params),
                params
            );
        }
    }
}

template<int DIM>
void pack_axis_plane(
    const std::vector<dim::State<DIM>>& U,
    const std::array<int, DIM>& N,
    int axis,
    int axis_value,
    int nmat,
    std::vector<double>& buffer)
{
    const int values_per_cell = nmat + DIM + 1 + std::max(nmat - 1, 0);
    const int cells = plane_cell_count<DIM>(N, axis);
    buffer.assign(static_cast<std::size_t>(cells) * values_per_cell, 0.0);

    int out = 0;
    for (int plane_linear = 0; plane_linear < cells; ++plane_linear) {
        const auto idx = plane_index<DIM>(plane_linear, axis, axis_value, N);
        const auto& state = U[flat_id<DIM>(idx, N)];
        for (double value : state.partial_mass) {
            buffer[out++] = value;
        }
        for (int d = 0; d < DIM; ++d) {
            buffer[out++] = state.mom[d];
        }
        buffer[out++] = state.E;
        for (double value : state.alpha) {
            buffer[out++] = value;
        }
    }
}

template<int DIM>
void unpack_axis_plane(
    std::vector<dim::State<DIM>>& U,
    const std::array<int, DIM>& N,
    int axis,
    int axis_value,
    int nmat,
    const std::vector<double>& buffer)
{
    const int cells = plane_cell_count<DIM>(N, axis);
    int in = 0;
    for (int plane_linear = 0; plane_linear < cells; ++plane_linear) {
        const auto idx = plane_index<DIM>(plane_linear, axis, axis_value, N);
        auto& state = U[flat_id<DIM>(idx, N)];
        state = dim::make_state<DIM>(nmat);
        for (int k = 0; k < nmat; ++k) {
            state.partial_mass[k] = buffer[in++];
        }
        for (int d = 0; d < DIM; ++d) {
            state.mom[d] = buffer[in++];
        }
        state.E = buffer[in++];
        for (int k = 0; k < nmat - 1; ++k) {
            state.alpha[k] = buffer[in++];
        }
    }
}

template<int DIM>
void apply_boundary_copy(
    std::vector<dim::State<DIM>>& U,
    const std::array<int, DIM>& N,
    const std::array<int, DIM>& dst,
    const std::array<int, DIM>& src,
    int axis,
    BoundaryConditionType bc)
{
    dim::State<DIM> state = U[flat_id<DIM>(src, N)];
    if (bc == BoundaryConditionType::reflective) {
        state.mom[axis] *= -1.0;
    }
    U[flat_id<DIM>(dst, N)] = state;
}

template<int DIM>
void apply_non_decomposed_boundaries(
    std::vector<dim::State<DIM>>& U,
    const std::array<int, DIM>& N,
    const std::array<BoundaryConditionType, DIM>& bc_lo,
    const std::array<BoundaryConditionType, DIM>& bc_hi,
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
                apply_boundary_copy<DIM>(U, N, idx, src, axis, bc_lo[axis]);
            }
            else if (idx[axis] == N[axis] - 1) {
                auto src = idx;
                src[axis] = N[axis] - 2;
                apply_boundary_copy<DIM>(U, N, idx, src, axis, bc_hi[axis]);
            }
        }
    }
}

template<int DIM>
void apply_decomposed_physical_boundaries(
    std::vector<dim::State<DIM>>& U,
    const std::array<int, DIM>& N,
    const std::array<BoundaryConditionType, DIM>& bc_lo,
    const std::array<BoundaryConditionType, DIM>& bc_hi,
    const MpiRuntime& mpi,
    int axis)
{
    const int cells = plane_cell_count<DIM>(N, axis);
    const int owned_lo = owned_lo_axis<DIM>(mpi);
    const int owned_hi = owned_hi_axis<DIM>(N, mpi, axis);

    if (mpi.rank == 0) {
        for (int plane_linear = 0; plane_linear < cells; ++plane_linear) {
            auto dst = plane_index<DIM>(plane_linear, axis, owned_lo, N);
            auto src = dst;
            src[axis] = owned_lo + 1;
            apply_boundary_copy<DIM>(U, N, dst, src, axis, bc_lo[axis]);
        }
    }

    if (mpi.rank == mpi.size - 1) {
        for (int plane_linear = 0; plane_linear < cells; ++plane_linear) {
            auto dst = plane_index<DIM>(plane_linear, axis, owned_hi, N);
            auto src = dst;
            src[axis] = owned_hi - 1;
            apply_boundary_copy<DIM>(U, N, dst, src, axis, bc_hi[axis]);
        }
    }
}

template<int DIM>
void exchange_axis_ghosts(
    std::vector<dim::State<DIM>>& U,
    const std::array<int, DIM>& N,
    const MpiRuntime& mpi,
    int axis,
    int nmat)
{
    const int left = mpi.rank - 1;
    const int right = mpi.rank + 1;
    const int owned_lo = owned_lo_axis<DIM>(mpi);
    const int owned_hi = owned_hi_axis<DIM>(N, mpi, axis);
    const int values_per_cell = nmat + DIM + 1 + std::max(nmat - 1, 0);
    const int plane_values = plane_cell_count<DIM>(N, axis) * values_per_cell;

    std::vector<double> send_left;
    std::vector<double> recv_left(plane_values);
    std::vector<double> send_right;
    std::vector<double> recv_right(plane_values);

    for (int offset = 0; offset < ghost_width; ++offset) {
        if (left >= 0) {
            pack_axis_plane<DIM>(U, N, axis, owned_lo + offset, nmat, send_left);
            MPI_Sendrecv(
                send_left.data(), plane_values, MPI_DOUBLE, left, 101 + offset,
                recv_left.data(), plane_values, MPI_DOUBLE, left, 201 + offset,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
            unpack_axis_plane<DIM>(U, N, axis, owned_lo - 1 - offset, nmat, recv_left);
        }

        if (right < mpi.size) {
            pack_axis_plane<DIM>(U, N, axis, owned_hi - offset, nmat, send_right);
            MPI_Sendrecv(
                send_right.data(), plane_values, MPI_DOUBLE, right, 201 + offset,
                recv_right.data(), plane_values, MPI_DOUBLE, right, 101 + offset,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
            unpack_axis_plane<DIM>(U, N, axis, owned_hi + 1 + offset, nmat, recv_right);
        }
    }
}

template<int DIM>
void refresh_boundaries(
    std::vector<dim::State<DIM>>& U,
    const std::array<int, DIM>& N,
    const std::array<BoundaryConditionType, DIM>& bc_lo,
    const std::array<BoundaryConditionType, DIM>& bc_hi,
    const MpiRuntime& mpi,
    int axis,
    int nmat)
{
    apply_non_decomposed_boundaries<DIM>(U, N, bc_lo, bc_hi, axis);
    exchange_axis_ghosts<DIM>(U, N, mpi, axis, nmat);
    apply_decomposed_physical_boundaries<DIM>(U, N, bc_lo, bc_hi, mpi, axis);
}

template<int DIM>
double compute_local_owned_dt(
    const std::vector<dim::State<DIM>>& U,
    const std::array<int, DIM>& N,
    const std::array<double, DIM>& dx,
    const dim::EOSParams& params,
    const std::string& lambda_model,
    double cfl,
    double dt_max,
    const MpiRuntime& mpi,
    int axis)
{
    double dt = dt_max;
    const int cells = total_cells<DIM>(N);
    const int owned_lo = owned_lo_axis<DIM>(mpi);
    const int owned_hi = owned_hi_axis<DIM>(N, mpi, axis);

    #pragma omp parallel for reduction(min:dt)
    for (int linear = 0; linear < cells; ++linear) {
        const auto idx = unflatten_id<DIM>(linear, N);
        if (idx[axis] < owned_lo || idx[axis] > owned_hi) {
            continue;
        }

        const dim::Primitive<DIM> P = dim::cons_to_prim<DIM>(U[linear], params);
        const double c = dim::MixtureEOS::generic_mixture_sound_speed(
            dim::total_density(U[linear]),
            P.p,
            P.alpha,
            P.rho,
            params,
            lambda_model
        );
        for (int d = 0; d < DIM; ++d) {
            const double wave_speed = std::abs(P.vel[d]) + c;
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
    const std::vector<dim::State<DIM>>& U,
    const std::array<int, DIM>& local_N,
    const std::array<double, DIM>& domain_min,
    const std::array<double, DIM>& dx,
    const Slab& slab,
    const MpiRuntime& mpi,
    int axis,
    const dim::EOSParams& params)
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
    file << "p,e";
    for (int k = 0; k < params.nmat(); ++k) {
        file << ",alpha" << k;
    }
    for (int k = 0; k < params.nmat(); ++k) {
        file << ",rho_mat" << k;
    }
    for (int k = 0; k < params.nmat(); ++k) {
        file << ",mass" << k;
    }
    file << "\n";
    file << std::setprecision(16);

    const int owned_lo = owned_lo_axis<DIM>(mpi);
    for (int local_axis = owned_lo; local_axis < owned_lo + slab.owned; ++local_axis) {
        const int global_axis = slab.start + local_axis - owned_lo;

        for (int plane_linear = 0; plane_linear < plane_cell_count<DIM>(local_N, axis); ++plane_linear) {
            const auto idx = plane_index<DIM>(plane_linear, axis, local_axis, local_N);
            const int id = flat_id<DIM>(idx, local_N);
            const dim::Primitive<DIM> P = dim::cons_to_prim<DIM>(U[id], params);
            const double rho = dim::total_density(U[id]);
            double velocity_squared = 0.0;
            for (int d = 0; d < DIM; ++d) {
                velocity_squared += P.vel[d] * P.vel[d];
            }
            const double internal_energy =
                std::max((U[id].E - 0.5 * rho * velocity_squared) / std::max(rho, 1.0e-30), 0.0);

            for (int d = 0; d < DIM; ++d) {
                const int global_index = (d == axis) ? global_axis : idx[d];
                const double coordinate =
                    domain_min[d] + (static_cast<double>(global_index) + 0.5) * dx[d];
                file << coordinate << ",";
            }

            file << rho << ",";
            for (int d = 0; d < DIM; ++d) {
                file << P.vel[d] << ",";
            }
            file << P.p << "," << internal_energy;
            for (double value : P.alpha) {
                file << "," << value;
            }
            for (double value : P.rho) {
                file << "," << value;
            }
            for (double value : U[id].partial_mass) {
                file << "," << value;
            }
            file << "\n";
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
    file << "interface_method = DIM_MPI\n";
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
int run_mpi_dim(int argc, char** argv, const MpiRuntime& mpi)
{
    if (argc < 2) {
        throw std::runtime_error("Usage: ./dim_mpi_main <config_file> [--resolution N[xN...]] [--output-dir path]");
    }

    const std::string config_file = argv[1];
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
    if (cfg.interface_method != "DIM") {
        throw std::runtime_error("dim_mpi_main supports only interface_method = DIM");
    }
    if (cfg.time_update != "split") {
        throw std::runtime_error("dim_mpi_main currently supports only split time update");
    }
    if (cfg.barton_solid_material >= 0) {
        throw std::runtime_error("dim_mpi_main does not support Barton DIM extension cases");
    }

    const dim::EOSParams material_params = build_dim_material_params(cfg.materials);

    if (!resolution_override.empty()) {
        std::array<int, DIM> N{};
        for (int d = 0; d < DIM; ++d) {
            N[d] = resolution_override[d];
        }
        cfg.N_list = {N};
    }

    std::array<BoundaryConditionType, DIM> bc_lo{};
    std::array<BoundaryConditionType, DIM> bc_hi{};
    for (int d = 0; d < DIM; ++d) {
        bc_lo[d] = parse_boundary_condition(cfg.bc_lo[d]);
        bc_hi[d] = parse_boundary_condition(cfg.bc_hi[d]);
    }

    const int decomp_axis = DIM - 1;

    for (const auto& global_N : cfg.N_list) {
        const Slab slab = compute_slab(global_N[decomp_axis], mpi.rank, mpi.size);
        if (slab.owned <= 0) {
            throw std::runtime_error("number of MPI ranks exceeds decomposed-axis cells");
        }

        std::array<int, DIM> local_N = global_N;
        local_N[decomp_axis] = slab.owned + ghost_lo(mpi) + ghost_hi(mpi);

        std::array<double, DIM> dx{};
        for (int d = 0; d < DIM; ++d) {
            dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / global_N[d];
        }

        std::vector<dim::State<DIM>> U;
        initialise_local_slab<DIM>(
            U,
            cfg,
            material_params,
            local_N,
            dx,
            slab,
            mpi,
            decomp_axis
        );
        refresh_boundaries<DIM>(
            U,
            local_N,
            bc_lo,
            bc_hi,
            mpi,
            decomp_axis,
            material_params.nmat()
        );

        const std::filesystem::path output_root =
            output_dir_override.empty()
                ? std::filesystem::path(cfg.output_dir) / (cfg.output_prefix + "_dim_mpi")
                : std::filesystem::path(output_dir_override);

        double time = 0.0;
        int step = 0;
        std::size_t next_output_index = 0;
        SolverPhaseTimings timings{};
        const auto wall_start = std::chrono::steady_clock::now();

        auto write_snapshot = [&](double output_time) {
            if (timing_only) {
                return;
            }
            const auto output_start = std::chrono::steady_clock::now();
            const std::filesystem::path rank_file =
                output_root /
                (
                    "rank_" + std::to_string(mpi.rank) + "_" +
                    cfg.output_prefix + "_" + format_mpi_time_tag(output_time) + ".csv"
                );
            write_rank_csv<DIM>(
                rank_file,
                U,
                local_N,
                cfg.domain_min,
                dx,
                slab,
                mpi,
                decomp_axis,
                material_params
            );
            const auto output_end = std::chrono::steady_clock::now();
            timings.output_seconds +=
                std::chrono::duration<double>(output_end - output_start).count();
        };

        while (next_output_index < cfg.output_times.size() &&
               cfg.output_times[next_output_index] <= 1.0e-14)
        {
            ++next_output_index;
        }

        while (time < cfg.tfinal - 1.0e-14) {
            const double next_output_time =
                (next_output_index < cfg.output_times.size())
                    ? cfg.output_times[next_output_index]
                    : cfg.tfinal;
            const double dt_max = next_output_time - time;
            const auto cfl_start = std::chrono::steady_clock::now();
            const double local_dt = compute_local_owned_dt<DIM>(
                U,
                local_N,
                dx,
                material_params,
                cfg.dim_lambda_model,
                cfg.cfl,
                dt_max,
                mpi,
                decomp_axis
            );
            double dt = dt_max;
            MPI_Allreduce(&local_dt, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            const auto cfl_end = std::chrono::steady_clock::now();
            timings.cfl_seconds += std::chrono::duration<double>(cfl_end - cfl_start).count();

            std::vector<dim::State<DIM>> stage = U;
            const auto sweep_start = std::chrono::steady_clock::now();

            for (int dir = 0; dir < DIM; ++dir) {
                std::vector<dim::State<DIM>> next(stage.size(), dim::make_state<DIM>(material_params.nmat()));
                dim::sweep_direction_dispatch<DIM>(
                    dir,
                    stage,
                    local_N,
                    dx,
                    material_params,
                    dt,
                    cfg.dim_alpha_source_floor,
                    cfg.dim_lambda_model,
                    next
                );

                const auto boundary_start = std::chrono::steady_clock::now();
                refresh_boundaries<DIM>(
                    next,
                    local_N,
                    bc_lo,
                    bc_hi,
                    mpi,
                    decomp_axis,
                    material_params.nmat()
                );
                const auto boundary_end = std::chrono::steady_clock::now();
                timings.boundary_seconds +=
                    std::chrono::duration<double>(boundary_end - boundary_start).count();

                stage.swap(next);
            }

            const auto sweep_end = std::chrono::steady_clock::now();
            timings.sweep_seconds +=
                std::chrono::duration<double>(sweep_end - sweep_start).count();

            if (dt <= 0.0) {
                throw std::runtime_error("non-positive MPI timestep");
            }

            U.swap(stage);
            time += dt;
            ++step;

            while (next_output_index < cfg.output_times.size() &&
                   time >= cfg.output_times[next_output_index] - 1.0e-14)
            {
                write_snapshot(cfg.output_times[next_output_index]);
                ++next_output_index;
            }
        }

        const auto wall_end = std::chrono::steady_clock::now();
        const double local_wall =
            std::chrono::duration<double>(wall_end - wall_start).count();
        double max_wall = local_wall;
        MPI_Reduce(&local_wall, &max_wall, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (!timing_only) {
            const auto output_start = std::chrono::steady_clock::now();
            const std::filesystem::path rank_file =
                output_root /
                ("rank_" + std::to_string(mpi.rank) + "_" + cfg.output_prefix + ".csv");
            write_rank_csv<DIM>(
                rank_file,
                U,
                local_N,
                cfg.domain_min,
                dx,
                slab,
                mpi,
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

            std::cout << "MPI DIM run completed: dim=" << DIM
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
        const int result = run_mpi_dim<DIM_>(argc, argv, mpi);
        MPI_Finalize();
        return result;
    }
    catch (const std::exception& exc) {
        std::cerr << "[rank " << mpi.rank << "] ERROR: " << exc.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return 1;
}
