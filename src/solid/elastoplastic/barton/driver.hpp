#pragma once

// Barton 1D and tensor driver loops.

#include "src/solid/elastoplastic/barton/material_points.hpp"

// 1D plate-impact driver.

// Driver loop for Barton 1D plate-impact validation cases.

namespace solid::barton {

inline int first_active_cell(const Config& cfg, double free_surface_position)
{
    const double dx = (cfg.domain_max - cfg.domain_min) / cfg.cells;
    const int index = static_cast<int>(
        std::floor((free_surface_position - cfg.domain_min) / dx));
    return std::clamp(index, 0, cfg.cells - 1);
}

inline int run_1d(const std::string& config_file)
{
    const Config cfg = load_config(config_file);
    std::vector<State> U = initialise(cfg);
    const double dx = (cfg.domain_max - cfg.domain_min) / cfg.cells;
    double free_surface_position =
        cfg.moving_free_surface ? cfg.free_surface_position : cfg.domain_min;

    std::string base = cfg.output_prefix;
    if (!cfg.output_dir.empty()) {
        base = cfg.output_dir + "/" + cfg.output_prefix + "/" + cfg.output_prefix;
    }
    const std::string nlabel = "_N" + std::to_string(cfg.cells);
    const std::filesystem::path output_path(base + nlabel + ".csv");
    ensure_directory(output_path.parent_path().string());

    double time = 0.0;
    int steps = 0;
    std::size_t next_output = 0;
    const auto wall_start = std::chrono::steady_clock::now();
    while (time < cfg.tfinal - 1.0e-15) {
        double dt = 0.0;
        if (cfg.moving_free_surface) {
            const int active_start = first_active_cell(cfg, free_surface_position);
            std::vector<State> active(U.begin() + active_start, U.end());
            BoundaryConditions active_bc = cfg.bc;
            active_bc.left = "free_surface";
            const Primitive surface_before = cons_to_prim(active.front(), cfg.material);
            dt = advance(active, cfg.material, dx, cfg.cfl, cfg.tfinal - time, active_bc);
            const Primitive surface_after = cons_to_prim(active.front(), cfg.material);
            const double surface_speed = std::max(0.0, 0.5 * (surface_before.u + surface_after.u));
            free_surface_position = std::min(cfg.domain_max, free_surface_position + surface_speed * dt);
            std::copy(active.begin(), active.end(), U.begin() + active_start);
            const int next_active_start = first_active_cell(cfg, free_surface_position);
            for (int i = 0; i < next_active_start; ++i) {
                U[i] = prim_to_cons(cfg.material.rho0, 0.0, cfg.material);
            }
        }
        else {
            dt = advance(U, cfg.material, dx, cfg.cfl, cfg.tfinal - time, cfg.bc);
        }
        time += dt;
        ++steps;
        while (next_output < cfg.output_times.size() &&
               time >= cfg.output_times[next_output] - 1.0e-15) {
            const double t_out = cfg.output_times[next_output];
            write_csv(base + time_suffix(t_out) + nlabel + ".csv", U, cfg, free_surface_position);
            ++next_output;
        }
    }
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();

    write_csv(output_path.string(), U, cfg, free_surface_position);
    std::ofstream runtime(base + nlabel + "_runtime.txt");
    runtime << "steps = " << steps << "\n";
    runtime << "final_time = " << time << "\n";
    runtime << "wall_time_seconds = " << wall_seconds << "\n";
    runtime << "cells = " << cfg.cells << "\n";
    runtime << "model = barton\n";

    std::cout << "Written: " << output_path.string() << "\n";
    std::cout << "steps = " << steps << ", final_time = " << time << "\n";
    return 0;
}


} // namespace solid::barton

// Tensor driver and public dispatch.

// Dispatches Barton elastoplastic configs to the Cartesian 1D or tensor multidimensional solver.

namespace solid::barton {

inline TensorState3D tensor_state_from_plate_state(
    const TensorSolverConfig::PlateState& state,
    const TensorMaterial& mat)
{
    const double temperature = state.has_pressure
        ? material_temperature_from_rho_p(state.rho, state.pressure, mat)
        : state.temperature;
    const std::array<double, 9> identity{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    return tensor_cons_from_F(
        state.rho,
        state.vel[0],
        state.vel[1],
        state.vel[2],
        temperature,
        identity,
        mat);
}

inline int tensor_bimaterial_transition_index(const std::vector<int>& material_id)
{
    for (int i = 0; i + 1 < static_cast<int>(material_id.size()); ++i) {
        if (material_id[i] != material_id[i + 1]) {
            return i;
        }
    }
    return -1;
}

inline double tensor_bimaterial_interface_speed(
    const std::vector<TensorState3D>& U,
    const std::vector<int>& material_id,
    const std::array<TensorMaterial, 2>& materials)
{
    const int i = tensor_bimaterial_transition_index(material_id);
    if (i < 0) {
        return 0.0;
    }
    TensorPrim3D PL = tensor_prim(U[i], materials[material_id[i]]);
    TensorPrim3D PR = tensor_prim(U[i + 1], materials[material_id[i + 1]]);
    tensor_solid_interface_star(PL, PR, materials[material_id[i]], materials[material_id[i + 1]]);
    return 0.5 * (PL.vel[0] + PR.vel[0]);
}

inline void remap_tensor_bimaterial_materials_1d(
    std::vector<TensorState3D>& U,
    std::vector<int>& material_id,
    const TensorSolverConfig& cfg,
    const std::array<TensorMaterial, 2>& materials,
    double interface_position)
{
    const int nx = cfg.cells[0];
    const double dx = (cfg.domain_max[0] - cfg.domain_min[0]) / nx;
    const TensorState3D fallback_left = tensor_state_from_plate_state(cfg.left_state, materials[0]);
    const TensorState3D fallback_right = tensor_state_from_plate_state(cfg.right_state, materials[1]);
    const std::vector<int> old_id = material_id;
    const std::vector<TensorState3D> old_U = U;

    for (int i = 0; i < nx; ++i) {
        const double x = cfg.domain_min[0] + (i + 0.5) * dx;
        const int desired = x < interface_position ? 0 : 1;
        if (desired == old_id[i]) {
            material_id[i] = desired;
            continue;
        }

        material_id[i] = desired;
        if (desired == 0) {
            if (i > 0 && material_id[i - 1] == 0) {
                U[i] = U[i - 1];
            }
            else if (i > 0 && old_id[i - 1] == 0) {
                U[i] = old_U[i - 1];
            }
            else {
                U[i] = fallback_left;
            }
        }
        else {
            if (i + 1 < nx && old_id[i + 1] == 1) {
                U[i] = old_U[i + 1];
            }
            else if (i + 1 < nx && material_id[i + 1] == 1) {
                U[i] = U[i + 1];
            }
            else {
                U[i] = fallback_right;
            }
        }
        enforce_tensor(U[i], materials[desired]);
    }
}

inline int run_tensor_solver(const std::string& config_file)
{
    const TensorSolverConfig cfg = load_tensor_solver_config(config_file);
    if (is_tensor_bimaterial_1d_test_case(cfg.test_case)) {
        std::vector<int> material_id;
        std::vector<TensorState3D> U = initialise_tensor_plate_impact_1d(cfg, material_id);
        const std::array<TensorMaterial, 2> materials{cfg.left_material, cfg.right_material};
        const int nx = cfg.cells[0];
        const double dx = (cfg.domain_max[0] - cfg.domain_min[0]) / nx;
        std::string base = cfg.output_prefix;
        if (!cfg.output_dir.empty()) {
            base = cfg.output_dir + "/" + cfg.output_prefix + "/" + cfg.output_prefix;
        }
        ensure_directory(std::filesystem::path(base).parent_path().string());

        std::vector<double> output_times = cfg.output_times.empty() ? std::vector<double>{cfg.tfinal} : cfg.output_times;
        if (cfg.output_times.empty() && cfg.tfinal >= 15.0e-6 - 1.0e-15) {
            output_times = {4.0e-6, 10.0e-6, 15.0e-6};
        }
        else if (cfg.output_times.empty() && cfg.tfinal >= 10.0e-6 - 1.0e-15) {
            output_times = {4.0e-6, 10.0e-6};
        }
        else if (cfg.output_times.empty() && cfg.tfinal >= 4.0e-6 - 1.0e-15) {
            output_times = {4.0e-6};
        }

        double time = 0.0;
        double interface_position = cfg.interface_position;
        int steps = 0;
        std::size_t next_output = 0;
        const auto wall_start = std::chrono::steady_clock::now();
        while (time < cfg.tfinal - 1.0e-15) {
            const double interface_speed = tensor_bimaterial_interface_speed(U, material_id, materials);
            const double dt = advance_tensor_plate_impact_1d(
                U, material_id, materials, cfg.bc, dx, cfg.cfl, cfg.tfinal - time);
            if (!std::isfinite(dt) || dt <= 0.0) {
                throw std::runtime_error("Barton tensor plate-impact timestep collapsed");
            }
            time += dt;
            interface_position = std::clamp(
                interface_position + interface_speed * dt,
                cfg.domain_min[0],
                cfg.domain_max[0]);
            remap_tensor_bimaterial_materials_1d(U, material_id, cfg, materials, interface_position);
            ++steps;
            while (next_output < output_times.size() &&
                   time >= output_times[next_output] - 1.0e-15) {
                const double t_out = output_times[next_output];
                write_tensor_plate_impact_1d(
                    base + time_suffix(t_out) + "_N" + std::to_string(nx) + ".csv",
                    U,
                    material_id,
                    cfg,
                    materials);
                ++next_output;
            }
            if (steps > 50000) {
                throw std::runtime_error("Barton tensor plate-impact solver exceeded 50000 steps");
            }
        }
        const auto wall_end = std::chrono::steady_clock::now();
        const double wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
        const std::filesystem::path output_path(base + "_N" + std::to_string(nx) + ".csv");
        write_tensor_plate_impact_1d(output_path.string(), U, material_id, cfg, materials);
        std::ofstream runtime(base + "_N" + std::to_string(nx) + "_runtime.txt");
        runtime << "steps = " << steps << "\n";
        runtime << "final_time = " << time << "\n";
        runtime << "wall_time_seconds = " << wall_seconds << "\n";
        runtime << "cells = " << nx << "\n";
        runtime << "model = barton_tensor_" << cfg.test_case << "\n";
        runtime << "initial_interface = " << cfg.interface_position << "\n";
        runtime << "final_interface = " << interface_position << "\n";
        std::cout << "Written: " << output_path.string() << "\n";
        std::cout << "steps = " << steps << ", final_time = " << time << "\n";
        return 0;
    }

    const TensorMaterial mat = cfg.material;
    if (cfg.dimension == 3) {
        std::vector<TensorState3D> U3 = initialise_tensor_cartesian_3d(cfg, mat);
        std::vector<TensorState3D> Ur = initialise_tensor_spherical_reference(cfg, mat);
        std::vector<MaterialPoint<3>> material_points =
            cfg.material_points ? seed_material_points<3>(cfg) : std::vector<MaterialPoint<3>>{};
        const int nx = cfg.cells[0];
        const int ny = cfg.cells[1];
        const int nz = cfg.cells[2];
        const double dx = (cfg.domain_max[0] - cfg.domain_min[0]) / nx;
        const double dy = (cfg.domain_max[1] - cfg.domain_min[1]) / ny;
        const double dz = (cfg.domain_max[2] - cfg.domain_min[2]) / nz;
        const double dr = (cfg.domain_max[0] - cfg.domain_min[0]) / cfg.radial_cells;
        std::string base = cfg.output_prefix;
        if (!cfg.output_dir.empty()) {
            base = cfg.output_dir + "/" + cfg.output_prefix + "/" + cfg.output_prefix;
        }
        ensure_directory(std::filesystem::path(base).parent_path().string());
        if (cfg.material_points) {
            write_material_points_vtp(material_point_vtp_path(base, "initial"), material_points);
        }
        double time = 0.0;
        int steps = 0;
        const auto wall_start = std::chrono::steady_clock::now();
        while (time < cfg.tfinal - 1.0e-15) {
            const double remaining = cfg.tfinal - time;
            if (remaining < 1.0e-14) {
                break;
            }
            const double dt3 = advance_tensor_3d(U3, mat, nx, ny, nz, dx, dy, dz, cfg.cfl, remaining);
            if (!std::isfinite(dt3) || dt3 < 1.0e-20) {
                std::ostringstream msg;
                msg << std::scientific
                    << "Barton tensor 3D solver timestep collapsed at t=" << time
                    << ", remaining=" << remaining
                    << ", dt=" << dt3;
                throw std::runtime_error(msg.str());
            }
            if (cfg.material_points) {
                advect_material_points(material_points, U3, mat, cfg, dt3);
                if (cfg.material_point_output_interval > 0 &&
                    (steps + 1) % cfg.material_point_output_interval == 0) {
                    std::ostringstream label;
                    label << "step" << std::setw(5) << std::setfill('0') << (steps + 1);
                    write_material_points_vtp(material_point_vtp_path(base, label.str()), material_points);
                }
            }
            double radial_elapsed = 0.0;
            while (radial_elapsed < dt3 - 1.0e-18) {
                const double dtr = advance_tensor_spherical(
                    Ur, mat, cfg.domain_min[0], dr, cfg.cfl, dt3 - radial_elapsed);
                if (!std::isfinite(dtr) || dtr <= 0.0) {
                    throw std::runtime_error("Barton tensor spherical reference timestep collapsed");
                }
                radial_elapsed += dtr;
            }
            time += dt3;
            ++steps;
            if (steps > 30000) {
                throw std::runtime_error("Barton tensor 3D solver exceeded 30000 steps at t=" + std::to_string(time));
            }
        }
        const auto wall_end = std::chrono::steady_clock::now();
        const double wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
        const std::filesystem::path output_path(
            base + "_N" + std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz) + ".csv");
        ensure_directory(output_path.parent_path().string());
        const std::string ref_path = base + "_spherical_N" + std::to_string(cfg.radial_cells) + ".csv";
        write_tensor_cartesian_3d(output_path.string(), U3, cfg, mat);
        write_tensor_spherical_reference(ref_path, Ur, cfg, mat);
        if (cfg.material_points) {
            write_material_points_vtp(material_point_vtp_path(base, "final"), material_points);
        }
        std::ofstream runtime(base + "_runtime.txt");
        runtime << "steps = " << steps << "\n";
        runtime << "final_time = " << time << "\n";
        runtime << "wall_time_seconds = " << wall_seconds << "\n";
        runtime << "cells = " << nx * ny * nz << "\n";
        runtime << "model = barton_tensor_3d\n";
        runtime << "material_points = " << material_points.size() << "\n";
        std::cout << "Written: " << output_path.string() << "\n";
        std::cout << "Written: " << ref_path << "\n";
        std::cout << "steps = " << steps << ", final_time = " << time << "\n";
        return 0;
    }

    std::vector<TensorState2D> U2 = initialise_tensor_cartesian_2d(cfg, mat);
    std::vector<TensorState2D> Ur = initialise_tensor_cylindrical_reference(cfg, mat);
    std::vector<MaterialPoint<2>> material_points =
        cfg.material_points ? seed_material_points<2>(cfg) : std::vector<MaterialPoint<2>>{};
    const int nx = cfg.cells[0];
    const int ny = cfg.cells[1];
    const double dx = (cfg.domain_max[0] - cfg.domain_min[0]) / nx;
    const double dy = (cfg.domain_max[1] - cfg.domain_min[1]) / ny;
    const double dr = (cfg.domain_max[0] - cfg.domain_min[0]) / cfg.radial_cells;
    std::string base = cfg.output_prefix;
    if (!cfg.output_dir.empty()) {
        base = cfg.output_dir + "/" + cfg.output_prefix + "/" + cfg.output_prefix;
    }
    ensure_directory(std::filesystem::path(base).parent_path().string());
    if (cfg.material_points) {
        write_material_points_vtp(material_point_vtp_path(base, "initial"), material_points);
    }
    double time = 0.0;
    int steps = 0;
    const auto wall_start = std::chrono::steady_clock::now();
    while (time < cfg.tfinal - 1.0e-15) {
        const double remaining = cfg.tfinal - time;
        if (remaining < 1.0e-14) {
            break;
        }
        const double dt2 = advance_tensor_2d(U2, mat, nx, ny, dx, dy, cfg.cfl, remaining);
        if (!std::isfinite(dt2) || dt2 < 1.0e-20) {
            std::ostringstream msg;
            msg << std::scientific
                << "Barton tensor solver timestep collapsed at t=" << time
                << ", remaining=" << remaining
                << ", dt=" << dt2;
            throw std::runtime_error(msg.str());
        }
        if (cfg.material_points) {
            advect_material_points(material_points, U2, mat, cfg, dt2);
            if (cfg.material_point_output_interval > 0 &&
                (steps + 1) % cfg.material_point_output_interval == 0) {
                std::ostringstream label;
                label << "step" << std::setw(5) << std::setfill('0') << (steps + 1);
                write_material_points_vtp(material_point_vtp_path(base, label.str()), material_points);
            }
        }
        double radial_elapsed = 0.0;
        while (radial_elapsed < dt2 - 1.0e-18) {
            const double dtr = advance_tensor_cyl(
                Ur, mat, cfg.domain_min[0], dr, cfg.cfl, dt2 - radial_elapsed);
            if (!std::isfinite(dtr) || dtr <= 0.0) {
                throw std::runtime_error("Barton tensor cylindrical reference timestep collapsed");
            }
            radial_elapsed += dtr;
        }
        time += dt2;
        ++steps;
        if (steps > 20000) {
            throw std::runtime_error("Barton tensor solver exceeded 20000 steps at t=" + std::to_string(time));
        }
    }
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
    const std::filesystem::path output_path(base + "_N" + std::to_string(nx) + "x" + std::to_string(ny) + ".csv");
    ensure_directory(output_path.parent_path().string());
    const std::string ref_path = base + "_cylindrical_N" + std::to_string(cfg.radial_cells) + ".csv";
    write_tensor_cartesian_2d(output_path.string(), U2, cfg, mat);
    write_tensor_cylindrical_reference(ref_path, Ur, cfg, mat);
    if (cfg.material_points) {
        write_material_points_vtp(material_point_vtp_path(base, "final"), material_points);
    }
    std::ofstream runtime(base + "_runtime.txt");
    runtime << "steps = " << steps << "\n";
    runtime << "final_time = " << time << "\n";
    runtime << "wall_time_seconds = " << wall_seconds << "\n";
    runtime << "cells = " << nx * ny << "\n";
    runtime << "model = barton_tensor\n";
    runtime << "material_points = " << material_points.size() << "\n";
    std::cout << "Written: " << output_path.string() << "\n";
    std::cout << "Written: " << ref_path << "\n";
    std::cout << "steps = " << steps << ", final_time = " << time << "\n";
    return 0;
}

inline int run(const std::string& config_file)
{
    if (is_tensor_solver_config(config_file)) {
        return run_tensor_solver(config_file);
    }
    return run_1d(config_file);
}


} // namespace solid::barton
