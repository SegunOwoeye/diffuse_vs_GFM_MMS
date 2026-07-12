#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/app/dimension.hpp"
#include "src/app/levelset/initial_levelset.hpp"
#include "src/app/material/material_builder.hpp"
#include "src/app/solver/solver_builder.hpp"
#include "src/core/phase_timings.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/state.hpp"
#include "src/euler/thermo_compute.hpp"
#include "src/io/config.hpp"
#include "src/io/config_loader.hpp"
#include "src/setup/initial_conditions.hpp"
#include "src/sim/solver/advance_step.hpp"

using EOS = MaterialEOS;

namespace {

constexpr int ghost_width = 4;

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

template<int DIM, typename T>
void slice_cell_field(
    const std::vector<T>& global,
    std::vector<T>& local,
    const std::array<int, DIM>& global_N,
    const std::array<int, DIM>& local_N,
    const Slab& slab,
    const MpiRuntime& mpi,
    int axis)
{
    local.assign(total_cells<DIM>(local_N), T{});
    const int lo = ghost_lo(mpi);
    for (int local_axis = 0; local_axis < local_N[axis]; ++local_axis) {
        const int global_axis = slab.start + local_axis - lo;
        if (global_axis < 0 || global_axis >= global_N[axis]) {
            throw std::runtime_error("slice_cell_field: local halo outside global grid");
        }
        for (int plane_linear = 0; plane_linear < plane_cell_count<DIM>(local_N, axis); ++plane_linear) {
            const auto local_idx = plane_index<DIM>(plane_linear, axis, local_axis, local_N);
            auto global_idx = local_idx;
            global_idx[axis] = global_axis;
            local[flat_id<DIM>(local_idx, local_N)] =
                global[flat_id<DIM>(global_idx, global_N)];
        }
    }
}

template<int DIM>
void pack_state_plane(
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
void unpack_state_plane(
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
void pack_int_plane(
    const std::vector<int>& field,
    const std::array<int, DIM>& N,
    int axis,
    int axis_value,
    std::vector<int>& buffer)
{
    const int cells = plane_cell_count<DIM>(N, axis);
    buffer.assign(cells, 0);
    for (int plane_linear = 0; plane_linear < cells; ++plane_linear) {
        const auto idx = plane_index<DIM>(plane_linear, axis, axis_value, N);
        buffer[plane_linear] = field[flat_id<DIM>(idx, N)];
    }
}

template<int DIM>
void unpack_int_plane(
    std::vector<int>& field,
    const std::array<int, DIM>& N,
    int axis,
    int axis_value,
    const std::vector<int>& buffer)
{
    const int cells = plane_cell_count<DIM>(N, axis);
    for (int plane_linear = 0; plane_linear < cells; ++plane_linear) {
        const auto idx = plane_index<DIM>(plane_linear, axis, axis_value, N);
        field[flat_id<DIM>(idx, N)] = buffer[plane_linear];
    }
}

template<int DIM>
void pack_scalar_plane(
    const std::vector<double>& field,
    const std::array<int, DIM>& N,
    int axis,
    int axis_value,
    std::vector<double>& buffer)
{
    const int cells = plane_cell_count<DIM>(N, axis);
    buffer.assign(cells, 0.0);
    for (int plane_linear = 0; plane_linear < cells; ++plane_linear) {
        const auto idx = plane_index<DIM>(plane_linear, axis, axis_value, N);
        buffer[plane_linear] = field[flat_id<DIM>(idx, N)];
    }
}

template<int DIM>
void unpack_scalar_plane(
    std::vector<double>& field,
    const std::array<int, DIM>& N,
    int axis,
    int axis_value,
    const std::vector<double>& buffer)
{
    const int cells = plane_cell_count<DIM>(N, axis);
    for (int plane_linear = 0; plane_linear < cells; ++plane_linear) {
        const auto idx = plane_index<DIM>(plane_linear, axis, axis_value, N);
        field[flat_id<DIM>(idx, N)] = buffer[plane_linear];
    }
}

template<int DIM>
void exchange_state_ghosts(
    std::vector<Conserved<DIM>>& U,
    const std::array<int, DIM>& N,
    const MpiRuntime& mpi,
    int axis)
{
    const int left = mpi.rank - 1;
    const int right = mpi.rank + 1;
    const int owned_lo = owned_lo_axis<DIM>(mpi);
    const int owned_hi = owned_hi_axis<DIM>(N, mpi, axis);
    const int plane_values = plane_cell_count<DIM>(N, axis) * (DIM + 2);
    std::vector<double> send_left;
    std::vector<double> recv_left(plane_values);
    std::vector<double> send_right;
    std::vector<double> recv_right(plane_values);

    for (int offset = 0; offset < ghost_width; ++offset) {
        if (left >= 0) {
            pack_state_plane<DIM>(U, N, axis, owned_lo + offset, send_left);
            MPI_Sendrecv(
                send_left.data(), plane_values, MPI_DOUBLE, left, 101 + offset,
                recv_left.data(), plane_values, MPI_DOUBLE, left, 201 + offset,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
            unpack_state_plane<DIM>(U, N, axis, owned_lo - 1 - offset, recv_left);
        }
        if (right < mpi.size) {
            pack_state_plane<DIM>(U, N, axis, owned_hi - offset, send_right);
            MPI_Sendrecv(
                send_right.data(), plane_values, MPI_DOUBLE, right, 201 + offset,
                recv_right.data(), plane_values, MPI_DOUBLE, right, 101 + offset,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
            unpack_state_plane<DIM>(U, N, axis, owned_hi + 1 + offset, recv_right);
        }
    }
}

template<int DIM>
void exchange_material_ghosts(
    std::vector<int>& material_id,
    const std::array<int, DIM>& N,
    const MpiRuntime& mpi,
    int axis)
{
    const int left = mpi.rank - 1;
    const int right = mpi.rank + 1;
    const int owned_lo = owned_lo_axis<DIM>(mpi);
    const int owned_hi = owned_hi_axis<DIM>(N, mpi, axis);
    const int plane_values = plane_cell_count<DIM>(N, axis);
    std::vector<int> send_left;
    std::vector<int> recv_left(plane_values);
    std::vector<int> send_right;
    std::vector<int> recv_right(plane_values);

    for (int offset = 0; offset < ghost_width; ++offset) {
        if (left >= 0) {
            pack_int_plane<DIM>(material_id, N, axis, owned_lo + offset, send_left);
            MPI_Sendrecv(
                send_left.data(), plane_values, MPI_INT, left, 301 + offset,
                recv_left.data(), plane_values, MPI_INT, left, 401 + offset,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
            unpack_int_plane<DIM>(material_id, N, axis, owned_lo - 1 - offset, recv_left);
        }
        if (right < mpi.size) {
            pack_int_plane<DIM>(material_id, N, axis, owned_hi - offset, send_right);
            MPI_Sendrecv(
                send_right.data(), plane_values, MPI_INT, right, 401 + offset,
                recv_right.data(), plane_values, MPI_INT, right, 301 + offset,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
            unpack_int_plane<DIM>(material_id, N, axis, owned_hi + 1 + offset, recv_right);
        }
    }
}

template<int DIM>
void exchange_phi_ghosts(
    std::vector<std::vector<double>>& phi_list,
    const std::array<int, DIM>& N,
    const MpiRuntime& mpi,
    int axis)
{
    const int left = mpi.rank - 1;
    const int right = mpi.rank + 1;
    const int owned_lo = owned_lo_axis<DIM>(mpi);
    const int owned_hi = owned_hi_axis<DIM>(N, mpi, axis);
    const int plane_values = plane_cell_count<DIM>(N, axis);

    for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
        std::vector<double> send_left;
        std::vector<double> recv_left(plane_values);
        std::vector<double> send_right;
        std::vector<double> recv_right(plane_values);

        for (int offset = 0; offset < ghost_width; ++offset) {
            const int tag_lo = 501 + 10 * k + offset;
            const int tag_hi = 601 + 10 * k + offset;
            if (left >= 0) {
                pack_scalar_plane<DIM>(phi_list[k], N, axis, owned_lo + offset, send_left);
                MPI_Sendrecv(
                    send_left.data(), plane_values, MPI_DOUBLE, left, tag_lo,
                    recv_left.data(), plane_values, MPI_DOUBLE, left, tag_hi,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE
                );
                unpack_scalar_plane<DIM>(phi_list[k], N, axis, owned_lo - 1 - offset, recv_left);
            }
            if (right < mpi.size) {
                pack_scalar_plane<DIM>(phi_list[k], N, axis, owned_hi - offset, send_right);
                MPI_Sendrecv(
                    send_right.data(), plane_values, MPI_DOUBLE, right, tag_hi,
                    recv_right.data(), plane_values, MPI_DOUBLE, right, tag_lo,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE
                );
                unpack_scalar_plane<DIM>(phi_list[k], N, axis, owned_hi + 1 + offset, recv_right);
            }
        }
    }
}

template<int DIM>
void exchange_scalar_ghosts(
    std::vector<double>& field,
    const std::array<int, DIM>& N,
    const MpiRuntime& mpi,
    int axis,
    int tag_base)
{
    const int left = mpi.rank - 1;
    const int right = mpi.rank + 1;
    const int owned_lo = owned_lo_axis<DIM>(mpi);
    const int owned_hi = owned_hi_axis<DIM>(N, mpi, axis);
    const int plane_values = plane_cell_count<DIM>(N, axis);
    std::vector<double> send_left;
    std::vector<double> recv_left(plane_values);
    std::vector<double> send_right;
    std::vector<double> recv_right(plane_values);

    for (int offset = 0; offset < ghost_width; ++offset) {
        if (left >= 0) {
            pack_scalar_plane<DIM>(field, N, axis, owned_lo + offset, send_left);
            MPI_Sendrecv(
                send_left.data(), plane_values, MPI_DOUBLE, left, tag_base + offset,
                recv_left.data(), plane_values, MPI_DOUBLE, left, tag_base + 100 + offset,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
            unpack_scalar_plane<DIM>(field, N, axis, owned_lo - 1 - offset, recv_left);
        }
        if (right < mpi.size) {
            pack_scalar_plane<DIM>(field, N, axis, owned_hi - offset, send_right);
            MPI_Sendrecv(
                send_right.data(), plane_values, MPI_DOUBLE, right, tag_base + 100 + offset,
                recv_right.data(), plane_values, MPI_DOUBLE, right, tag_base + offset,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
            unpack_scalar_plane<DIM>(field, N, axis, owned_hi + 1 + offset, recv_right);
        }
    }
}

template<int DIM>
void extend_normal_speed_mpi(
    std::vector<double>& speed,
    const std::vector<char>& fixed,
    const std::vector<double>& phi,
    const SolverContext<DIM>& ctx,
    const MpiRuntime& mpi,
    int axis,
    int tag_base,
    int iterations = 10)
{
    const int Ntot = static_cast<int>(speed.size());
    std::vector<double> work = speed;

    for (int iter = 0; iter < iterations; ++iter) {
        exchange_scalar_ghosts<DIM>(
            speed,
            ctx.N,
            mpi,
            axis,
            tag_base + 10 * iter
        );
        work = speed;

        #pragma omp parallel for if(Ntot > 512)
        for (int id = 0; id < Ntot; ++id) {
            if (fixed[id] != 0) {
                continue;
            }

            const std::array<int, DIM> idx = unflatten_index<DIM>(id, ctx.level_set_grid);
            const double phi_abs = std::abs(phi[id]);

            double sum = 0.0;
            int count = 0;
            for (int dir = 0; dir < DIM; ++dir) {
                for (const int step : {-1, 1}) {
                    std::array<int, DIM> nb_idx{};
                    if (!try_offset_index<DIM>(idx, dir, step, ctx.level_set_grid, nb_idx)) {
                        continue;
                    }
                    const int nb_id = flatten_index<DIM>(nb_idx, ctx.level_set_grid);
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

    exchange_scalar_ghosts<DIM>(
        speed,
        ctx.N,
        mpi,
        axis,
        tag_base + 10 * iterations
    );
}

template<int DIM>
void make_normal_speed_fields_mpi_consistent(
    std::vector<std::vector<double>>& normal_speed_fields,
    const std::vector<std::vector<double>>& phi_list,
    const SolverContext<DIM>& ctx,
    const MpiRuntime& mpi,
    int axis)
{
    double min_dx = ctx.dx[0];
    for (int d = 1; d < DIM; ++d) {
        min_dx = std::min(min_dx, ctx.dx[d]);
    }
    const double phi_tol = geometry_tolerance(min_dx, 1.0e-12, 1.0e-14);

    for (int k = 0; k < static_cast<int>(normal_speed_fields.size()); ++k) {
        std::vector<char> fixed(normal_speed_fields[k].size(), 0);
        for (int id = 0; id < static_cast<int>(fixed.size()); ++id) {
            fixed[id] = rgfm_is_bordering_cell<DIM>(
                id,
                phi_list[k],
                ctx.level_set_grid,
                phi_tol
            ) ? 1 : 0;
        }

        extend_normal_speed_mpi<DIM>(
            normal_speed_fields[k],
            fixed,
            phi_list[k],
            ctx,
            mpi,
            axis,
            800 + 200 * k
        );
    }
}

template<int DIM>
void apply_boundary_copy(
    std::vector<Conserved<DIM>>& U,
    std::vector<int>& material_id,
    std::vector<std::vector<double>>& phi_list,
    const std::array<int, DIM>& N,
    const std::array<int, DIM>& dst,
    const std::array<int, DIM>& src,
    int axis,
    BoundaryConditionType bc)
{
    const int dst_id = flat_id<DIM>(dst, N);
    const int src_id = flat_id<DIM>(src, N);
    U[dst_id] = U[src_id];
    if (bc == BoundaryConditionType::reflective) {
        U[dst_id].mom[axis] *= -1.0;
    }
    (void)material_id;
    (void)phi_list;
}

template<int DIM>
void apply_non_decomposed_boundaries(
    std::vector<Conserved<DIM>>& U,
    std::vector<int>& material_id,
    std::vector<std::vector<double>>& phi_list,
    const SolverContext<DIM>& ctx,
    int decomp_axis)
{
    const int cells = total_cells<DIM>(ctx.N);
    for (int axis = 0; axis < DIM; ++axis) {
        if (axis == decomp_axis) {
            continue;
        }
        for (int linear = 0; linear < cells; ++linear) {
            const auto idx = unflatten_id<DIM>(linear, ctx.N);
            if (idx[axis] == 0) {
                auto src = idx;
                src[axis] = 1;
                apply_boundary_copy<DIM>(U, material_id, phi_list, ctx.N, idx, src, axis, ctx.bc_lo[axis]);
            }
            else if (idx[axis] == ctx.N[axis] - 1) {
                auto src = idx;
                src[axis] = ctx.N[axis] - 2;
                apply_boundary_copy<DIM>(U, material_id, phi_list, ctx.N, idx, src, axis, ctx.bc_hi[axis]);
            }
        }
    }
}

template<int DIM>
void apply_decomposed_physical_boundaries(
    std::vector<Conserved<DIM>>& U,
    std::vector<int>& material_id,
    std::vector<std::vector<double>>& phi_list,
    const SolverContext<DIM>& ctx,
    const MpiRuntime& mpi,
    int axis)
{
    const int cells = plane_cell_count<DIM>(ctx.N, axis);
    const int owned_lo = owned_lo_axis<DIM>(mpi);
    const int owned_hi = owned_hi_axis<DIM>(ctx.N, mpi, axis);

    if (mpi.rank == 0) {
        for (int plane_linear = 0; plane_linear < cells; ++plane_linear) {
            auto dst = plane_index<DIM>(plane_linear, axis, owned_lo, ctx.N);
            auto src = dst;
            src[axis] = owned_lo + 1;
            apply_boundary_copy<DIM>(U, material_id, phi_list, ctx.N, dst, src, axis, ctx.bc_lo[axis]);
        }
    }
    if (mpi.rank == mpi.size - 1) {
        for (int plane_linear = 0; plane_linear < cells; ++plane_linear) {
            auto dst = plane_index<DIM>(plane_linear, axis, owned_hi, ctx.N);
            auto src = dst;
            src[axis] = owned_hi - 1;
            apply_boundary_copy<DIM>(U, material_id, phi_list, ctx.N, dst, src, axis, ctx.bc_hi[axis]);
        }
    }
}

template<int DIM>
void refresh_ghosts_and_boundaries(
    std::vector<Conserved<DIM>>& U,
    std::vector<int>& material_id,
    std::vector<std::vector<double>>& phi_list,
    SolverContext<DIM>& ctx,
    const MpiRuntime& mpi,
    int axis)
{
    apply_non_decomposed_boundaries<DIM>(U, material_id, phi_list, ctx, axis);
    exchange_state_ghosts<DIM>(U, ctx.N, mpi, axis);
    exchange_material_ghosts<DIM>(material_id, ctx.N, mpi, axis);
    exchange_phi_ghosts<DIM>(phi_list, ctx.N, mpi, axis);
    apply_decomposed_physical_boundaries<DIM>(U, material_id, phi_list, ctx, mpi, axis);
    ctx.material_id = material_id;
    ctx.phi_list = phi_list;
}

template<int DIM>
double compute_local_owned_dt(
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const SolverContext<DIM>& ctx,
    const MpiRuntime& mpi,
    double dt_max,
    int axis)
{
    double dt = dt_max;
    const int cells = total_cells<DIM>(ctx.N);
    const int owned_lo = owned_lo_axis<DIM>(mpi);
    const int owned_hi = owned_hi_axis<DIM>(ctx.N, mpi, axis);

    #pragma omp parallel for reduction(min:dt)
    for (int linear = 0; linear < cells; ++linear) {
        const auto idx = unflatten_id<DIM>(linear, ctx.N);
        if (idx[axis] < owned_lo || idx[axis] > owned_hi) {
            continue;
        }
        const int mat = material_id[linear];
        if (mat < 0 || mat >= static_cast<int>(ctx.material_params.size())) {
            continue;
        }
        const ThermoState<DIM> thermo =
            compute_thermo<DIM, EOS>(U[linear], ctx.material_params[mat]);
        for (int d = 0; d < DIM; ++d) {
            const double wave_speed = std::abs(thermo.vel[d]) + thermo.c;
            if (wave_speed > 1.0e-14) {
                dt = std::min(dt, ctx.cfl * ctx.dx[d] / wave_speed);
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
    const std::vector<std::vector<double>>& phi_list,
    const std::array<int, DIM>& local_N,
    const std::array<double, DIM>& domain_min,
    const std::array<double, DIM>& dx,
    const Slab& slab,
    const MpiRuntime& mpi,
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
    file << "p,e,mat";
    for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
        file << ",phi" << k;
    }
    file << "\n";
    file << std::setprecision(16);

    const int owned_lo = owned_lo_axis<DIM>(mpi);
    for (int local_axis = owned_lo; local_axis < owned_lo + slab.owned; ++local_axis) {
        const int global_axis = slab.start + local_axis - owned_lo;
        for (int plane_linear = 0; plane_linear < plane_cell_count<DIM>(local_N, axis); ++plane_linear) {
            const auto idx = plane_index<DIM>(plane_linear, axis, local_axis, local_N);
            const int id = flat_id<DIM>(idx, local_N);
            const int mat = material_id.empty() ? 0 : material_id[id];
            const Primitive<DIM> primitive =
                cons_to_prim<DIM, EOS>(U[id], material_params[mat]);
            const double e = EOS::internal_energy(
                primitive.rho,
                primitive.p,
                material_params[mat]
            );

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
            file << primitive.p << "," << e << "," << mat;
            for (const auto& phi : phi_list) {
                file << "," << phi[id];
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
    file << "interface_method = SIM_MPI\n";
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
    file << "phase_level_set_seconds = " << timings.level_set_seconds << "\n";
    file << "phase_material_transfer_seconds = " << timings.material_transfer_seconds << "\n";
    file << "phase_output_seconds = " << timings.output_seconds << "\n";
}

template<int DIM>
void initialise_local_from_global(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& global_N,
    const std::array<int, DIM>& local_N,
    const std::vector<EOSParams>& material_params,
    const Slab& slab,
    const MpiRuntime& mpi,
    int axis,
    std::vector<Conserved<DIM>>& U,
    std::vector<int>& material_id,
    std::vector<std::vector<double>>& phi_list,
    std::vector<TrackedInterface>& tracked_interfaces,
    int& background_material_id)
{
    std::vector<Conserved<DIM>> U_global;
    std::vector<int> material_global;
    initialise_from_config<DIM, EOS>(U_global, material_global, cfg, global_N);

    InitialLevelSetData<DIM> ls_data =
        initialise_phi_data_from_regions<DIM>(cfg, global_N, material_global);

    SolverContext<DIM> global_ctx = build_solver_context<DIM>(
        cfg,
        global_N,
        material_global,
        material_params,
        ls_data
    );

    if (global_ctx.reassign_material_from_phi && !global_ctx.phi_list.empty()) {
        assign_material_ids_from_phi<DIM>(
            global_ctx.phi_list,
            global_ctx.tracked_interfaces,
            global_ctx.background_material_id,
            global_ctx.material_id,
            global_ctx.level_set_grid,
            global_ctx.level_set_component_policy
        );
        fill_small_enclosed_background_cavities<DIM>(
            global_ctx.material_id,
            global_ctx.tracked_interfaces,
            global_ctx.background_material_id,
            global_ctx.level_set_grid
        );
        material_global = global_ctx.material_id;
    }

    slice_cell_field<DIM>(U_global, U, global_N, local_N, slab, mpi, axis);
    slice_cell_field<DIM>(material_global, material_id, global_N, local_N, slab, mpi, axis);
    phi_list.clear();
    for (const auto& phi_global : global_ctx.phi_list) {
        std::vector<double> phi_local;
        slice_cell_field<DIM>(phi_global, phi_local, global_N, local_N, slab, mpi, axis);
        phi_list.push_back(std::move(phi_local));
    }
    tracked_interfaces = global_ctx.tracked_interfaces;
    background_material_id = global_ctx.background_material_id;
}

template<int DIM>
SolverContext<DIM> build_local_context(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& local_N,
    const std::array<double, DIM>& dx,
    const std::vector<EOSParams>& material_params,
    const std::vector<int>& material_id,
    const std::vector<std::vector<double>>& phi_list,
    const std::vector<TrackedInterface>& tracked_interfaces,
    int background_material_id)
{
    SolverContext<DIM> ctx{};
    ctx.N = local_N;
    ctx.dx = dx;
    ctx.material_id = material_id;
    ctx.material_params = material_params;
    ctx.initialise_level_set_grid();
    for (int d = 0; d < DIM; ++d) {
        ctx.bc_lo[d] = boundary_condition_from_config(cfg.bc_lo[d]);
        ctx.bc_hi[d] = boundary_condition_from_config(cfg.bc_hi[d]);
    }
    ctx.phi_list = phi_list;
    ctx.tracked_interfaces = tracked_interfaces;
    ctx.background_material_id = background_material_id;
    ctx.cfl = cfg.cfl;
    ctx.dt_max = 1.0e-3;
    ctx.advect_level_set = (cfg.interface_method == "GFM" && cfg.use_level_set && !ctx.phi_list.empty());
    ctx.reassign_material_from_phi = ctx.advect_level_set;
    ctx.level_set_reinit_method = cfg.level_set_reinit_method;
    ctx.level_set_advection = cfg.level_set_advection;
    ctx.level_set_component_policy = cfg.level_set_component_policy;
    ctx.level_set_derivative_scheme =
        level_set_derivative_scheme_from_config(cfg.level_set_spatial_derivative);
    ctx.rgfm_star_velocity_mode = cfg.rgfm_star_velocity_mode;
    ctx.time_update = cfg.time_update;
    ctx.reinit_enabled = (
        cfg.interface_method == "GFM" &&
        cfg.use_level_set &&
        !ctx.phi_list.empty() &&
        cfg.reinit_interval > 0
    );
    ctx.reinit_frequency = ctx.reinit_enabled ? cfg.reinit_interval : 0;
    ctx.reinit_iterations = cfg.reinit_iterations;
    ctx.completed_steps = 0;
    return ctx;
}

template<int DIM>
void advance_one_mpi_step(
    std::vector<Conserved<DIM>>& U,
    std::vector<int>& material_id,
    std::vector<std::vector<double>>& phi_list,
    SolverContext<DIM>& ctx,
    const MpiRuntime& mpi,
    int axis,
    double dt,
    SolverPhaseTimings& timings)
{
    if (ctx.time_update != "split") {
        throw std::runtime_error("rgfm_mpi_main currently supports only split time update");
    }

    const int Ntot = static_cast<int>(U.size());
    std::vector<int> material_current = material_id;

    if (ctx.reassign_material_from_phi) {
        assign_material_ids_from_phi<DIM>(
            phi_list,
            ctx.tracked_interfaces,
            ctx.background_material_id,
            material_current,
            ctx.level_set_grid,
            ctx.level_set_component_policy
        );
        fill_small_enclosed_background_cavities<DIM>(
            material_current,
            ctx.tracked_interfaces,
            ctx.background_material_id,
            ctx.level_set_grid
        );
    }

    std::vector<std::vector<double>> phi_work = phi_list;
    std::vector<int> planar_phi_axis(phi_work.size(), -1);
    for (int k = 0; k < static_cast<int>(phi_work.size()); ++k) {
        planar_phi_axis[k] = detect_planar_level_set_axis<DIM>(phi_work[k], ctx);
    }
    const int planar_flow_axis = common_planar_level_set_axis<DIM>(planar_phi_axis);

    std::vector<std::vector<std::array<double, DIM>>> normals_current(phi_list.size());
    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(phi_list.size()); ++k) {
        normals_current[k] = compute_solver_normals<DIM>(phi_list[k], ctx);
    }

    RGFMInterfaceData<DIM> rgfm_current =
        build_rgfm_interface_data<DIM, EOS>(
            U,
            material_current,
            phi_list,
            ctx.tracked_interfaces,
            normals_current,
            ctx
        );

    std::vector<Conserved<DIM>> stage = U;
    zero_roundoff_planar_transverse_momentum<DIM>(stage, planar_flow_axis);

    const auto sweep_start = std::chrono::steady_clock::now();
    for (int dir = 0; dir < DIM; ++dir) {
        std::vector<Conserved<DIM>> next(stage.size());
        sweep_direction_dispatch<DIM, EOS>(
            dir,
            stage,
            material_current,
            phi_list,
            ctx.tracked_interfaces,
            normals_current,
            ctx,
            dt,
            next,
            nullptr
        );

        const auto boundary_start = std::chrono::steady_clock::now();
        refresh_ghosts_and_boundaries<DIM>(
            next,
            material_current,
            phi_list,
            ctx,
            mpi,
            axis
        );
        const auto boundary_end = std::chrono::steady_clock::now();
        timings.boundary_seconds +=
            std::chrono::duration<double>(boundary_end - boundary_start).count();
        zero_roundoff_planar_transverse_momentum<DIM>(next, planar_flow_axis);
        stage.swap(next);
    }
    const auto sweep_end = std::chrono::steady_clock::now();
    timings.sweep_seconds +=
        std::chrono::duration<double>(sweep_end - sweep_start).count();

    const auto level_set_start = std::chrono::steady_clock::now();
    if (ctx.advect_level_set && !phi_work.empty()) {
        std::vector<std::array<double, DIM>> level_set_velocity_field;
        const bool use_vector_level_set_transport =
            ctx.level_set_advection == "flow" ||
            ctx.level_set_advection == "physical_flow" ||
            (ctx.level_set_advection == "normal_speed" && phi_work.size() > 1);

        if (ctx.level_set_advection == "flow") {
            level_set_velocity_field =
                build_velocity_field<DIM, EOS>(
                    rgfm_current.U_real,
                    material_current,
                    ctx.material_params
                );
        }
        else if (ctx.level_set_advection == "physical_flow" ||
                 (ctx.level_set_advection == "normal_speed" && phi_work.size() > 1)) {
            level_set_velocity_field =
                build_velocity_field<DIM, EOS>(
                    U,
                    material_current,
                    ctx.material_params
                );
        }

        #pragma omp parallel for
        for (int k = 0; k < static_cast<int>(phi_work.size()); ++k) {
            if (use_vector_level_set_transport) {
                phi_work[k] = advect_phi<DIM>(
                    phi_list[k],
                    level_set_velocity_field,
                    ctx.level_set_grid,
                    dt,
                    ctx.level_set_derivative_scheme
                );
            }
            else {
                phi_work[k] = advect_phi_normal_speed<DIM>(
                    phi_list[k],
                    rgfm_current.normal_speed_fields[k],
                    ctx.level_set_grid,
                    dt,
                    ctx.level_set_derivative_scheme
                );
            }
            project_planar_level_set<DIM>(
                phi_work[k],
                ctx,
                planar_phi_axis[k]
            );
        }
    }

    if (ctx.reinit_enabled &&
        ((ctx.completed_steps + 1) % ctx.reinit_frequency == 0))
    {
        #pragma omp parallel for
        for (int k = 0; k < static_cast<int>(phi_work.size()); ++k) {
            phi_work[k] = reinitialise_solver_phi<DIM>(phi_work[k], ctx);
            project_planar_level_set<DIM>(
                phi_work[k],
                ctx,
                planar_phi_axis[k]
            );
        }
    }
    exchange_phi_ghosts<DIM>(phi_work, ctx.N, mpi, axis);
    const auto level_set_end = std::chrono::steady_clock::now();
    timings.level_set_seconds +=
        std::chrono::duration<double>(level_set_end - level_set_start).count();

    const auto transfer_start = std::chrono::steady_clock::now();
    std::vector<int> material_work = material_current;
    if (ctx.reassign_material_from_phi) {
        assign_material_ids_from_phi<DIM>(
            phi_work,
            ctx.tracked_interfaces,
            ctx.background_material_id,
            material_work,
            ctx.level_set_grid,
            ctx.level_set_component_policy
        );
        fill_small_enclosed_background_cavities<DIM>(
            material_work,
            ctx.tracked_interfaces,
            ctx.background_material_id,
            ctx.level_set_grid
        );
        enforce_level_set_sign_from_material_map<DIM>(
            phi_work,
            material_work,
            ctx
        );
    }

    std::vector<std::vector<Conserved<DIM>>> transfer_material_states;
    const std::vector<std::vector<Conserved<DIM>>>* transfer_material_states_ptr = nullptr;
    if (ctx.advect_level_set && !phi_list.empty()) {
        const RGFMInterfaceData<DIM> rgfm_transfer =
            build_rgfm_interface_data<DIM, EOS>(
                stage,
                material_current,
                phi_list,
                ctx.tracked_interfaces,
                normals_current,
                ctx
            );
        transfer_material_states = rgfm_transfer.material_states;
        transfer_material_states_ptr = &transfer_material_states;
    }

    std::vector<Conserved<DIM>> final_state =
        transfer_reassigned_material_states<DIM, EOS>(
            stage,
            material_current,
            material_work,
            ctx,
            transfer_material_states_ptr
        );

    zero_roundoff_planar_transverse_momentum<DIM>(final_state, planar_flow_axis);
    const auto transfer_end = std::chrono::steady_clock::now();
    timings.material_transfer_seconds +=
        std::chrono::duration<double>(transfer_end - transfer_start).count();

    refresh_ghosts_and_boundaries<DIM>(
        final_state,
        material_work,
        phi_work,
        ctx,
        mpi,
        axis
    );

    U.swap(final_state);
    material_id.swap(material_work);
    phi_list.swap(phi_work);
    ctx.material_id = material_id;
    ctx.phi_list = phi_list;
    ctx.completed_steps += 1;

    if (static_cast<int>(U.size()) != Ntot) {
        throw std::runtime_error("advance_one_mpi_step: state size changed unexpectedly");
    }
}

template<int DIM>
int run_mpi_gfm(int argc, char** argv, const MpiRuntime& mpi)
{
    if (argc < 2) {
        throw std::runtime_error("Usage: ./rgfm_mpi_main <config_file> [--resolution N[xN...]] [--output-dir path]");
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
    if (cfg.interface_method != "GFM") {
        throw std::runtime_error("rgfm_mpi_main supports only interface_method = GFM");
    }
    if (cfg.time_update != "split") {
        throw std::runtime_error("rgfm_mpi_main currently supports only split time update");
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
        local_N[decomp_axis] = slab.owned + ghost_lo(mpi) + ghost_hi(mpi);

        std::array<double, DIM> dx{};
        for (int d = 0; d < DIM; ++d) {
            dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / global_N[d];
        }

        std::vector<Conserved<DIM>> U;
        std::vector<int> material_id;
        std::vector<std::vector<double>> phi_list;
        std::vector<TrackedInterface> tracked_interfaces;
        int background_material_id = 0;

        initialise_local_from_global<DIM>(
            cfg,
            global_N,
            local_N,
            material_params,
            slab,
            mpi,
            decomp_axis,
            U,
            material_id,
            phi_list,
            tracked_interfaces,
            background_material_id
        );

        SolverContext<DIM> ctx = build_local_context<DIM>(
            cfg,
            local_N,
            dx,
            material_params,
            material_id,
            phi_list,
            tracked_interfaces,
            background_material_id
        );

        refresh_ghosts_and_boundaries<DIM>(U, material_id, phi_list, ctx, mpi, decomp_axis);

        const std::filesystem::path output_root =
            output_dir_override.empty()
                ? std::filesystem::path(cfg.output_dir) / (cfg.output_prefix + "_sim_mpi")
                : std::filesystem::path(output_dir_override);

        double time = 0.0;
        int step = 0;
        std::size_t next_output_index = 0;
        SolverPhaseTimings timings{};
        const auto wall_start = std::chrono::steady_clock::now();

        auto write_snapshot = [&](double output_time) {
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
                material_id,
                phi_list,
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
                material_id,
                ctx,
                mpi,
                dt_max,
                decomp_axis
            );
            double dt = dt_max;
            MPI_Allreduce(&local_dt, &dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
            const auto cfl_end = std::chrono::steady_clock::now();
            timings.cfl_seconds += std::chrono::duration<double>(cfl_end - cfl_start).count();

            if (dt <= 0.0) {
                throw std::runtime_error("non-positive MPI timestep");
            }

            advance_one_mpi_step<DIM>(
                U,
                material_id,
                phi_list,
                ctx,
                mpi,
                decomp_axis,
                dt,
                timings
            );

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

        const auto output_start = std::chrono::steady_clock::now();
        const std::filesystem::path rank_file =
            output_root /
            ("rank_" + std::to_string(mpi.rank) + "_" + cfg.output_prefix + ".csv");
        write_rank_csv<DIM>(
            rank_file,
            U,
            material_id,
            phi_list,
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

        SolverPhaseTimings reduced_timings{};
        MPI_Reduce(&timings.cfl_seconds, &reduced_timings.cfl_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&timings.sweep_seconds, &reduced_timings.sweep_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&timings.boundary_seconds, &reduced_timings.boundary_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&timings.level_set_seconds, &reduced_timings.level_set_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&timings.material_transfer_seconds, &reduced_timings.material_transfer_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
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

            std::cout << "MPI SIM run completed: dim=" << DIM
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
        const int result = run_mpi_gfm<DIM_>(argc, argv, mpi);
        MPI_Finalize();
        return result;
    }
    catch (const std::exception& exc) {
        std::cerr << "[rank " << mpi.rank << "] ERROR: " << exc.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return 1;
}
