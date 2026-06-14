#pragma once

// Euler-fluid / Barton-solid rGFM driver.
// This owns the fluid-solid runner dispatch across supported dimensions. The
// dimension-specific sections below reuse the same interface-state contract:
// build rGFM states in the local interface normal direction, keep each side on
// its existing finite-volume machinery, and replace material-boundary states at
// the fluid-solid interface.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/euler/riemann/hll.hpp"
#include "src/sim/gfm/fluid_solid_rgfm.hpp"
#include "src/solid/elastoplastic/barton/advance.hpp"
#include "src/solid/elastoplastic/barton/plasticity.hpp"

namespace solid::barton {

struct FluidSolidStateConfig {
    double rho = 1.0;
    double u = 0.0;
    double p = 1.0;
    bool has_sigma11 = false;
    double sigma11 = -1.0;
};

struct FluidSolidConfig1D {
    double domain_min = 0.0;
    double domain_max = 10.0;
    double interface_x = 5.0;
    int cells = 2200;
    double tfinal = 4.45e-3;
    double cfl = 0.8;
    EOSParams fluid{};
    Material solid{};
    FluidSolidStateConfig fluid_initial{};
    FluidSolidStateConfig solid_initial{};
    BoundaryConditions fluid_bc{};
    BoundaryConditions solid_bc{};
    std::vector<double> output_times{};
    std::string output_prefix = "fluid_solid_rgfm";
    std::string output_dir = "data/csv/fluid_solid";
};

inline void parse_fluid_solid_fluid_value(
    EOSParams& params,
    const std::string& key,
    const std::string& value)
{
    if (key == "eos" || key == "type") {
        params.kind = eos_kind_from_string(value);
    }
    else if (key == "gamma") {
        params.gamma = std::stod(value);
    }
    else if (key == "p_inf" || key == "B") {
        params.p_inf = std::stod(value);
    }
    else if (key == "tait_B") {
        params.tait_B = std::stod(value);
    }
    else if (key == "rho0") {
        params.rho0 = std::stod(value);
    }
    else if (key == "p0") {
        params.p0 = std::stod(value);
    }
    else {
        throw std::runtime_error("Unknown fluid-solid fluid material key: " + key);
    }
}

inline FluidSolidStateConfig parse_fluid_solid_state(const std::string& value)
{
    FluidSolidStateConfig state{};
    for (const auto& part : solid::text::split_csv(value)) {
        const auto [key, raw] = solid::text::split_key_value(part);
        if (key == "rho") {
            state.rho = std::stod(raw);
        }
        else if (key == "u" || key == "vel") {
            state.u = std::stod(raw);
        }
        else if (key == "p" || key == "pressure") {
            state.p = std::stod(raw);
        }
        else if (key == "sigma11" || key == "sxx") {
            state.has_sigma11 = true;
            state.sigma11 = std::stod(raw);
        }
        else {
            throw std::runtime_error("Unknown fluid-solid initial state key: " + key);
        }
    }
    return state;
}

inline FluidSolidConfig1D load_fluid_solid_config(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open fluid-solid config: " + filename);
    }

    FluidSolidConfig1D cfg{};
    cfg.output_times = {cfg.tfinal};

    std::string line;
    while (std::getline(file, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line = line.substr(0, comment);
        }
        line = solid::text::trim(line);
        if (line.empty()) {
            continue;
        }

        const auto [key, value] = solid::text::split_key_value(line);
        if (key == "model" || key == "solid_model") {
            if (solid::text::trim(value) != "fluid_solid_rgfm") {
                throw std::runtime_error("Fluid-solid config must set model=fluid_solid_rgfm");
            }
        }
        else if (key == "model_type") {
            continue;
        }
        else if (key == "dimension") {
            if (static_cast<int>(solid::text::parse_single_bracket_value(value)) != 1) {
                throw std::runtime_error("Fluid-solid rGFM driver supports dimension=1");
            }
        }
        else if (key == "domain_min") {
            cfg.domain_min = solid::text::parse_single_bracket_value(value);
        }
        else if (key == "domain_max") {
            cfg.domain_max = solid::text::parse_single_bracket_value(value);
        }
        else if (key == "interface" || key == "interface_x") {
            cfg.interface_x = solid::text::parse_single_bracket_value(value);
        }
        else if (key == "N" || key == "cells") {
            cfg.cells = static_cast<int>(solid::text::parse_single_bracket_value(value));
        }
        else if (key == "tfinal") {
            cfg.tfinal = std::stod(value);
        }
        else if (key == "cfl") {
            cfg.cfl = std::stod(value);
        }
        else if (key == "fluid") {
            for (const auto& part : solid::text::split_csv(value)) {
                const auto [fluid_key, fluid_value] = solid::text::split_key_value(part);
                parse_fluid_solid_fluid_value(cfg.fluid, fluid_key, fluid_value);
            }
        }
        else if (key == "solid") {
            for (const auto& part : solid::text::split_csv(value)) {
                const auto [solid_key, solid_value] = solid::text::split_key_value(part);
                parse_material_value(cfg.solid, solid_key, solid_value);
            }
        }
        else if (key == "fluid_state") {
            cfg.fluid_initial = parse_fluid_solid_state(value);
        }
        else if (key == "solid_state") {
            cfg.solid_initial = parse_fluid_solid_state(value);
        }
        else if (key == "bc_lo") {
            cfg.fluid_bc.left = solid::text::trim(value);
        }
        else if (key == "bc_hi") {
            cfg.solid_bc.right = solid::text::trim(value);
        }
        else if (key == "output_times") {
            cfg.output_times.clear();
            for (const auto& part : solid::text::split_csv(value)) {
                cfg.output_times.push_back(solid::text::parse_single_bracket_value(part));
            }
            std::sort(cfg.output_times.begin(), cfg.output_times.end());
        }
        else if (key == "output_prefix") {
            cfg.output_prefix = value;
        }
        else if (key == "output_dir") {
            cfg.output_dir = value;
        }
        else {
            throw std::runtime_error("Unknown fluid-solid config key: " + key);
        }
    }

    if (cfg.cells <= 1 || cfg.domain_max <= cfg.domain_min ||
        cfg.interface_x <= cfg.domain_min || cfg.interface_x >= cfg.domain_max ||
        cfg.tfinal <= 0.0 || cfg.cfl <= 0.0) {
        throw std::runtime_error("Invalid fluid-solid mesh/time controls");
    }
    if (cfg.output_times.empty()) {
        cfg.output_times.push_back(cfg.tfinal);
    }
    return cfg;
}

inline Conserved<1> make_fluid_state_from_config(
    const FluidSolidStateConfig& state,
    const EOSParams& params)
{
    ::Primitive<1> P{};
    P.rho = state.rho;
    P.vel[0] = state.u;
    P.p = state.p;
    return ::prim_to_cons<1, MaterialEOS>(P, params);
}

inline State make_solid_state_from_config(
    const FluidSolidStateConfig& state,
    const Material& mat)
{
    const State base = prim_to_cons(state.rho, state.u, mat);
    const double sigma11 = state.has_sigma11 ? state.sigma11 : -state.p;
    return barton_state_with_sigma11(base, mat, sigma11, state.u);
}

inline void enforce_fluid_state(Conserved<1>& U, const EOSParams& params)
{
    U.rho = std::max(U.rho, 1.0e-12);
    if (!std::isfinite(U.mom[0])) {
        U.mom[0] = 0.0;
    }
    const ::Primitive<1> P = ::cons_to_prim<1, MaterialEOS>(U, params);
    if (!std::isfinite(P.p) || P.p <= 1.0e-12) {
        ::Primitive<1> fixed{};
        fixed.rho = U.rho;
        fixed.vel[0] = U.mom[0] / U.rho;
        fixed.p = 1.0e-12;
        U = ::prim_to_cons<1, MaterialEOS>(fixed, params);
    }
}

inline double max_fluid_signal_speed(
    const std::vector<Conserved<1>>& U,
    const EOSParams& params)
{
    double speed = 1.0;
    for (const auto& cell : U) {
        const ::Primitive<1> P = ::cons_to_prim<1, MaterialEOS>(cell, params);
        const double c = MaterialEOS::sound_speed<1>(cell, params);
        speed = std::max(speed, std::abs(P.vel[0]) + c);
    }
    return speed;
}

inline double max_coupled_signal_speed(
    const std::vector<Conserved<1>>& fluid,
    const std::vector<State>& solid_cells,
    const EOSParams& fluid_params,
    const Material& solid_mat)
{
    return std::max(
        max_fluid_signal_speed(fluid, fluid_params),
        max_signal_speed(solid_cells, solid_mat));
}

inline std::vector<Conserved<1>> advance_fluid_rgfm_1d(
    const std::vector<Conserved<1>>& U,
    const Conserved<1>& interface_state,
    const EOSParams& params,
    double dx,
    double dt,
    const BoundaryConditions& bc)
{
    const int n = static_cast<int>(U.size());
    std::array<double, 1> normal{1.0};
    std::vector<Conserved<1>> flux(n + 1);

    flux[0] = compute_flux_normal<1, MaterialEOS>(U.front(), normal, params);
    if (bc.left == "reflective") {
        Conserved<1> ghost = U.front();
        ghost.mom[0] = -ghost.mom[0];
        flux[0] = hll_flux_normal<1, MaterialEOS>(ghost, U.front(), normal, params, params);
    }
    for (int i = 1; i < n; ++i) {
        flux[i] = hll_flux_normal<1, MaterialEOS>(U[i - 1], U[i], normal, params, params);
    }
    flux[n] = compute_flux_normal<1, MaterialEOS>(interface_state, normal, params);

    std::vector<Conserved<1>> next = U;
    for (int i = 0; i < n; ++i) {
        next[i] = U[i] - (dt / dx) * (flux[i + 1] - flux[i]);
        enforce_fluid_state(next[i], params);
    }
    return next;
}

inline std::vector<State> advance_solid_rgfm_1d(
    const std::vector<State>& U,
    const State& interface_state,
    const Material& mat,
    double dx,
    double dt,
    const BoundaryConditions& bc)
{
    const int n = static_cast<int>(U.size());
    std::vector<State> flux(n + 1);
    std::vector<double> face_rhoF11(n + 1, 0.0);

    flux[0] = flux_x(interface_state, mat);
    face_rhoF11[0] = interface_state.rhoF11;
    for (int i = 1; i < n; ++i) {
        flux[i] = rusanov_flux(U[i - 1], U[i], mat);
        face_rhoF11[i] = 0.5 * (U[i - 1].rhoF11 + U[i].rhoF11);
    }

    State right = U.back();
    if (bc.right == "reflective") {
        right.mom = -right.mom;
    }
    flux[n] = (bc.right == "reflective")
        ? rusanov_flux(U.back(), right, mat)
        : flux_x(U.back(), mat);
    face_rhoF11[n] = U.back().rhoF11;

    std::vector<State> next = U;
    for (int i = 0; i < n; ++i) {
        next[i] = U[i] - (dt / dx) * (flux[i + 1] - flux[i]);
    }
    for (int i = 0; i < n; ++i) {
        const Primitive P = cons_to_prim(U[i], mat);
        const double beta = (face_rhoF11[i + 1] - face_rhoF11[i]) / dx;
        next[i].rhoF11 -= dt * P.u * beta;
        enforce(next[i], mat);
        apply_plastic_relaxation(next[i], mat, dt);
    }
    return next;
}

inline std::string fluid_solid_base_path(const FluidSolidConfig1D& cfg)
{
    if (cfg.output_dir.empty()) {
        return cfg.output_prefix;
    }
    return cfg.output_dir + "/" + cfg.output_prefix + "/" + cfg.output_prefix;
}

inline std::string fluid_solid_time_suffix(double t)
{
    std::ostringstream ss;
    ss << "_t" << std::scientific << std::setprecision(3) << t;
    std::string out = ss.str();
    for (char& c : out) {
        if (c == '+') c = 'p';
        if (c == '-') c = 'm';
    }
    return out;
}

inline void write_fluid_solid_csv(
    const std::string& filename,
    const std::vector<Conserved<1>>& fluid,
    const std::vector<State>& solid_cells,
    const FluidSolidConfig1D& cfg,
    double time)
{
    ensure_directory(std::filesystem::path(filename).parent_path().string());
    std::ofstream out(filename);
    out << "x,material,rho,u,p,sigma11,sigma22,sigma33,time\n";

    const double dx = (cfg.domain_max - cfg.domain_min) / cfg.cells;
    for (int i = 0; i < static_cast<int>(fluid.size()); ++i) {
        const double x = cfg.domain_min + (static_cast<double>(i) + 0.5) * dx;
        const ::Primitive<1> P = ::cons_to_prim<1, MaterialEOS>(fluid[i], cfg.fluid);
        out << x << ",fluid," << P.rho << "," << P.vel[0] << "," << P.p
            << ",0,0,0," << time << "\n";
    }
    for (int j = 0; j < static_cast<int>(solid_cells.size()); ++j) {
        const double x = cfg.interface_x + (static_cast<double>(j) + 0.5) * dx;
        const Primitive P = cons_to_prim(solid_cells[j], cfg.solid);
        out << x << ",solid," << P.rho << "," << P.u << ",0,"
            << P.sigma11 << "," << P.sigma22 << "," << P.sigma33
            << "," << time << "\n";
    }
}

inline int run_fluid_solid_rgfm_1d(const std::string& config_file)
{
    const FluidSolidConfig1D cfg = load_fluid_solid_config(config_file);
    const double dx = (cfg.domain_max - cfg.domain_min) / cfg.cells;
    const int fluid_cells = std::clamp(
        static_cast<int>(std::round((cfg.interface_x - cfg.domain_min) / dx)),
        1,
        cfg.cells - 1);
    const int solid_cells_n = cfg.cells - fluid_cells;

    std::vector<Conserved<1>> fluid(
        fluid_cells,
        make_fluid_state_from_config(cfg.fluid_initial, cfg.fluid));
    std::vector<State> solid_cells(
        solid_cells_n,
        make_solid_state_from_config(cfg.solid_initial, cfg.solid));

    const std::string base = fluid_solid_base_path(cfg);
    const std::string nlabel = "_N" + std::to_string(cfg.cells);

    double time = 0.0;
    int steps = 0;
    std::size_t next_output = 0;
    const auto wall_start = std::chrono::steady_clock::now();
    while (time < cfg.tfinal - 1.0e-15) {
        const double max_speed =
            max_coupled_signal_speed(fluid, solid_cells, cfg.fluid, cfg.solid);
        const double dt = std::min(cfg.tfinal - time, cfg.cfl * dx / max_speed);
        if (!std::isfinite(dt) || dt <= 0.0) {
            throw std::runtime_error("Fluid-solid rGFM timestep collapsed");
        }

        const FluidSolidRGFMState1D interface =
            fluid_solid_rgfm_interface_states_1d(
                fluid.back(),
                solid_cells.front(),
                cfg.fluid,
                cfg.solid,
                1);

        std::vector<Conserved<1>> rgfm_fluid = fluid;
        std::vector<State> rgfm_solid = solid_cells;
        rgfm_fluid.back() = interface.fluid_interface;
        rgfm_solid.front() = interface.solid_interface;

        fluid = advance_fluid_rgfm_1d(
            rgfm_fluid,
            interface.fluid_interface,
            cfg.fluid,
            dx,
            dt,
            cfg.fluid_bc);
        solid_cells = advance_solid_rgfm_1d(
            rgfm_solid,
            interface.solid_interface,
            cfg.solid,
            dx,
            dt,
            cfg.solid_bc);

        time += dt;
        ++steps;
        while (next_output < cfg.output_times.size() &&
               time >= cfg.output_times[next_output] - 1.0e-15) {
            const double t_out = cfg.output_times[next_output];
            write_fluid_solid_csv(
                base + fluid_solid_time_suffix(t_out) + nlabel + ".csv",
                fluid,
                solid_cells,
                cfg,
                time);
            ++next_output;
        }
    }
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_seconds =
        std::chrono::duration<double>(wall_end - wall_start).count();

    const std::string final_path = base + nlabel + ".csv";
    write_fluid_solid_csv(final_path, fluid, solid_cells, cfg, time);

    std::ofstream runtime(base + nlabel + "_runtime.txt");
    runtime << "steps = " << steps << "\n";
    runtime << "final_time = " << time << "\n";
    runtime << "wall_time_seconds = " << wall_seconds << "\n";
    runtime << "cells = " << cfg.cells << "\n";
    runtime << "model = fluid_solid_rgfm\n";
    runtime << "fluid_cells = " << fluid_cells << "\n";
    runtime << "solid_cells = " << solid_cells_n << "\n";

    std::cout << "Written: " << final_path << "\n";
    std::cout << "steps = " << steps << ", final_time = " << time << "\n";
    return 0;
}

// Two-dimensional planar fluid-solid rGFM driver.

struct FluidSolidStateConfig2D {
    double rho = 1.0;
    double un = 0.0;
    double us = 0.0;
    double p = 1.0;
    bool has_sigma_nn = false;
    double sigma_nn = -1.0;
};

struct FluidSolidConfig2D {
    std::array<double, 2> domain_min{0.0, 0.0};
    std::array<double, 2> domain_max{10.0, 10.0};
    std::array<int, 2> cells{251, 251};
    std::array<double, 2> interface_normal{0.5, 0.8660254037844386};
    double interface_offset = 5.0;
    double tfinal = 4.45e-3;
    double cfl = 0.4;
    bool normal_tangent_update = false;
    int normal_tangent_cells = 0;
    EOSParams fluid{};
    TensorMaterial solid{};
    FluidSolidStateConfig2D fluid_initial{};
    FluidSolidStateConfig2D solid_initial{};
    std::vector<double> output_times{};
    std::string output_prefix = "case4_3_rotated_water_solid_rgfm";
    std::string output_dir = "data/csv/fluid_solid";
};

struct FluidSolidInterfaceState2D {
    Conserved<2> fluid_interface{};
    TensorState2D solid_interface{};
    double normal_velocity = 0.0;
    double pressure = 0.0;
    double solid_normal_stress = 0.0;
};

inline std::array<double, 2> normalised_pair(std::array<double, 2> v)
{
    const double mag = std::hypot(v[0], v[1]);
    if (!std::isfinite(mag) || mag <= 1.0e-14) {
        throw std::runtime_error("fluid-solid 2D normal must be non-zero");
    }
    return {v[0] / mag, v[1] / mag};
}

inline std::array<double, 2> tangent_from_normal(const std::array<double, 2>& n)
{
    return {-n[1], n[0]};
}

inline double dot2(const std::array<double, 2>& a, const std::array<double, 2>& b)
{
    return a[0] * b[0] + a[1] * b[1];
}

inline double signed_phi(
    const FluidSolidConfig2D& cfg,
    double x,
    double y)
{
    return cfg.interface_normal[0] * x + cfg.interface_normal[1] * y - cfg.interface_offset;
}

inline bool is_fluid_phi(double phi)
{
    return phi < 0.0;
}

inline void parse_fluid_solid_fluid_value_2d(
    EOSParams& params,
    const std::string& key,
    const std::string& value)
{
    parse_fluid_solid_fluid_value(params, key, value);
}

inline FluidSolidStateConfig2D parse_fluid_solid_state_2d(const std::string& value)
{
    FluidSolidStateConfig2D state{};
    for (const auto& part : solid::text::split_csv(value)) {
        const auto [key, raw] = solid::text::split_key_value(part);
        if (key == "rho") state.rho = std::stod(raw);
        else if (key == "un" || key == "u" || key == "normal_velocity") state.un = std::stod(raw);
        else if (key == "us" || key == "v" || key == "tangent_velocity") state.us = std::stod(raw);
        else if (key == "p" || key == "pressure") state.p = std::stod(raw);
        else if (key == "sigma_nn" || key == "snn") {
            state.has_sigma_nn = true;
            state.sigma_nn = std::stod(raw);
        }
        else {
            throw std::runtime_error("Unknown 2D fluid-solid initial state key: " + key);
        }
    }
    return state;
}

inline std::array<double, 2> parse_pair2d_or_scalar(const std::string& value)
{
    const auto values = solid::text::parse_numeric_list(value);
    if (values.size() == 1) return {values[0], values[0]};
    if (values.size() == 2) return {values[0], values[1]};
    throw std::runtime_error("Expected one or two numeric values in: " + value);
}

inline std::array<int, 2> parse_cells2d_or_scalar(const std::string& value)
{
    const auto values = solid::text::parse_numeric_list(value);
    if (values.size() == 1) {
        const int n = static_cast<int>(values[0]);
        return {n, n};
    }
    if (values.size() == 2) {
        return {static_cast<int>(values[0]), static_cast<int>(values[1])};
    }
    throw std::runtime_error("Expected one or two cell counts in: " + value);
}

inline FluidSolidConfig2D load_fluid_solid_config_2d(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open fluid-solid 2D config: " + filename);
    }

    FluidSolidConfig2D cfg{};
    cfg.output_times = {cfg.tfinal};

    std::string line;
    while (std::getline(file, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);
        line = solid::text::trim(line);
        if (line.empty()) continue;

        const auto [key, value] = solid::text::split_key_value(line);
        if (key == "model" || key == "solid_model") {
            if (solid::text::trim(value) != "fluid_solid_rgfm") {
                throw std::runtime_error("Fluid-solid config must set model=fluid_solid_rgfm");
            }
        }
        else if (key == "model_type") continue;
        else if (key == "dimension") {
            if (static_cast<int>(solid::text::parse_single_bracket_value(value)) != 2) {
                throw std::runtime_error("Fluid-solid 2D driver supports dimension=2");
            }
        }
        else if (key == "domain_min") cfg.domain_min = parse_pair2d_or_scalar(value);
        else if (key == "domain_max") cfg.domain_max = parse_pair2d_or_scalar(value);
        else if (key == "N" || key == "cells") cfg.cells = parse_cells2d_or_scalar(value);
        else if (key == "interface_normal") cfg.interface_normal = normalised_pair(parse_pair2d_or_scalar(value));
        else if (key == "interface_offset") cfg.interface_offset = std::stod(value);
        else if (key == "rotation_degrees") {
            const double theta = std::stod(value) * std::acos(-1.0) / 180.0;
            cfg.interface_normal = {std::cos(theta), std::sin(theta)};
        }
        else if (key == "tfinal") cfg.tfinal = std::stod(value);
        else if (key == "cfl") cfg.cfl = std::stod(value);
        else if (key == "normal_tangent_update") {
            const std::string flag = solid::text::trim(value);
            cfg.normal_tangent_update = flag == "true" || flag == "1" || flag == "yes";
        }
        else if (key == "normal_tangent_cells") {
            cfg.normal_tangent_cells = static_cast<int>(solid::text::parse_single_bracket_value(value));
        }
        else if (key == "fluid") {
            for (const auto& part : solid::text::split_csv(value)) {
                const auto [fluid_key, fluid_value] = solid::text::split_key_value(part);
                parse_fluid_solid_fluid_value_2d(cfg.fluid, fluid_key, fluid_value);
            }
        }
        else if (key == "solid") {
            for (const auto& part : solid::text::split_csv(value)) {
                const auto [solid_key, solid_value] = solid::text::split_key_value(part);
                parse_tensor_material_value(cfg.solid, solid_key, solid_value);
            }
        }
        else if (key == "fluid_state") cfg.fluid_initial = parse_fluid_solid_state_2d(value);
        else if (key == "solid_state") cfg.solid_initial = parse_fluid_solid_state_2d(value);
        else if (key == "output_times") {
            cfg.output_times.clear();
            for (const auto& part : solid::text::split_csv(value)) {
                cfg.output_times.push_back(solid::text::parse_single_bracket_value(part));
            }
            std::sort(cfg.output_times.begin(), cfg.output_times.end());
        }
        else if (key == "output_prefix") cfg.output_prefix = value;
        else if (key == "output_dir") cfg.output_dir = value;
        else if (key == "bc_lo" || key == "bc_hi") continue;
        else {
            throw std::runtime_error("Unknown fluid-solid 2D config key: " + key);
        }
    }

    cfg.interface_normal = normalised_pair(cfg.interface_normal);
    if (cfg.cells[0] <= 1 || cfg.cells[1] <= 1 ||
        cfg.domain_max[0] <= cfg.domain_min[0] ||
        cfg.domain_max[1] <= cfg.domain_min[1] ||
        cfg.tfinal <= 0.0 || cfg.cfl <= 0.0) {
        throw std::runtime_error("Invalid fluid-solid 2D mesh/time controls");
    }
    if (cfg.output_times.empty()) cfg.output_times.push_back(cfg.tfinal);
    return cfg;
}

inline Conserved<2> make_fluid_state_from_config_2d(
    const FluidSolidStateConfig2D& state,
    const EOSParams& params,
    const std::array<double, 2>& n)
{
    const std::array<double, 2> s = tangent_from_normal(n);
    ::Primitive<2> P{};
    P.rho = state.rho;
    P.vel[0] = state.un * n[0] + state.us * s[0];
    P.vel[1] = state.un * n[1] + state.us * s[1];
    P.p = state.p;
    return ::prim_to_cons<2, MaterialEOS>(P, params);
}

inline std::array<double, 9> local_stretch_F(
    const std::array<double, 2>& n,
    double normal_stretch,
    double tangent_stretch)
{
    const std::array<double, 2> s = tangent_from_normal(n);
    std::array<double, 9> F{};
    F[0] = normal_stretch * n[0] * n[0] + tangent_stretch * s[0] * s[0];
    F[1] = normal_stretch * n[0] * n[1] + tangent_stretch * s[0] * s[1];
    F[3] = normal_stretch * n[1] * n[0] + tangent_stretch * s[1] * s[0];
    F[4] = normal_stretch * n[1] * n[1] + tangent_stretch * s[1] * s[1];
    F[8] = tangent_stretch;
    return F;
}

inline std::array<double, 9> local_stretch_tensor(
    const std::array<double, 2>& n,
    double normal_stretch,
    double tangent_stretch)
{
    return local_stretch_F(n, normal_stretch, tangent_stretch);
}

inline std::array<double, 9> local_normal_shear_stretch_tensor(
    const std::array<double, 2>& n,
    double normal_log_stretch,
    double shear)
{
    const std::array<double, 2> s = tangent_from_normal(n);
    const double a = std::exp(normal_log_stretch);
    const double b = std::exp(-0.5 * normal_log_stretch);
    std::array<double, 9> F{};
    for (int row = 0; row < 2; ++row) {
        const double er_n = row == 0 ? n[0] : n[1];
        const double er_s = row == 0 ? s[0] : s[1];
        for (int col = 0; col < 2; ++col) {
            const double ec_n = col == 0 ? n[0] : n[1];
            const double ec_s = col == 0 ? s[0] : s[1];
            F[3 * row + col] =
                a * er_n * ec_n +
                b * er_s * ec_s +
                shear * er_s * ec_n;
        }
    }
    F[8] = b;
    return F;
}

inline double normal_stress_from_tensor(
    const TensorPrim2D& P,
    const std::array<double, 2>& n)
{
    return n[0] * (P.sigma[0] * n[0] + P.sigma[1] * n[1]) +
           n[1] * (P.sigma[3] * n[0] + P.sigma[4] * n[1]);
}

inline double tangential_stress_from_tensor(
    const TensorPrim2D& P,
    const std::array<double, 2>& n)
{
    const std::array<double, 2> s = tangent_from_normal(n);
    return s[0] * (P.sigma[0] * s[0] + P.sigma[1] * s[1]) +
           s[1] * (P.sigma[3] * s[0] + P.sigma[4] * s[1]);
}

inline double shear_stress_from_tensor(
    const TensorPrim2D& P,
    const std::array<double, 2>& n)
{
    const std::array<double, 2> s = tangent_from_normal(n);
    return s[0] * (P.sigma[0] * n[0] + P.sigma[1] * n[1]) +
           s[1] * (P.sigma[3] * n[0] + P.sigma[4] * n[1]);
}

inline TensorState2D tensor_state_with_normal_stress(
    const TensorState2D& state,
    const TensorMaterial& mat,
    const std::array<double, 2>& n,
    double target_sigma_nn,
    double target_un)
{
    const TensorPrim2D P = tensor_prim(state, mat);
    const std::array<double, 2> s = tangent_from_normal(n);
    const double us = P.vel[0] * s[0] + P.vel[1] * s[1];

    auto candidate = [&](double q) {
        const auto stretch = local_stretch_tensor(n, std::exp(q), std::exp(-0.5 * q));
        const std::array<double, 9> F = matmul3(stretch, P.F);
        TensorState2D U = tensor_cons_from_F(
            P.rho,
            target_un * n[0] + us * s[0],
            target_un * n[1] + us * s[1],
            P.T,
            F,
            mat);
        U.rhoEqps = state.rhoEqps;
        U.rhoDamage = state.rhoDamage;
        enforce_tensor(U, mat);
        return U;
    };

    auto residual = [&](double q) {
        return normal_stress_from_tensor(tensor_prim(candidate(q), mat), n) - target_sigma_nn;
    };

    double lo = -0.75;
    double hi = 0.75;
    double rlo = residual(lo);
    double rhi = residual(hi);
    for (int expand = 0; rlo * rhi > 0.0 && expand < 4; ++expand) {
        lo *= 1.25;
        hi *= 1.25;
        rlo = residual(lo);
        rhi = residual(hi);
    }
    if (rlo * rhi > 0.0 || !std::isfinite(rlo) || !std::isfinite(rhi)) {
        return candidate(0.0);
    }
    for (int iter = 0; iter < 36; ++iter) {
        const double mid = 0.5 * (lo + hi);
        const double rm = residual(mid);
        if (rlo * rm <= 0.0) {
            hi = mid;
            rhi = rm;
        }
        else {
            lo = mid;
            rlo = rm;
        }
    }
    (void)rhi;
    return candidate(0.5 * (lo + hi));
}

inline TensorState2D tensor_state_with_fluid_interface_traction(
    const TensorState2D& state,
    const TensorMaterial& mat,
    const std::array<double, 2>& n,
    double target_sigma_nn,
    double target_sigma_sn,
    double target_un)
{
    const TensorPrim2D P = tensor_prim(state, mat);
    const std::array<double, 2> s = tangent_from_normal(n);
    const double us = P.vel[0] * s[0] + P.vel[1] * s[1];

    auto candidate = [&](double q, double g) {
        const auto stretch = local_normal_shear_stretch_tensor(n, q, g);
        const std::array<double, 9> F = matmul3(stretch, P.F);
        TensorState2D U = tensor_cons_from_F(
            P.rho,
            target_un * n[0] + us * s[0],
            target_un * n[1] + us * s[1],
            P.T,
            F,
            mat);
        U.rhoEqps = state.rhoEqps;
        U.rhoDamage = state.rhoDamage;
        enforce_tensor(U, mat);
        return U;
    };

    auto residual = [&](double q, double g) {
        const TensorPrim2D Pc = tensor_prim(candidate(q, g), mat);
        return std::array<double, 2>{
            normal_stress_from_tensor(Pc, n) - target_sigma_nn,
            shear_stress_from_tensor(Pc, n) - target_sigma_sn,
        };
    };

    double q = 0.0;
    double g = 0.0;
    for (int iter = 0; iter < 16; ++iter) {
        const auto r = residual(q, g);
        const double norm = std::max(std::abs(r[0]), std::abs(r[1]));
        if (!std::isfinite(norm)) break;
        if (norm < 1.0e-7 * std::max(1.0, std::abs(target_sigma_nn))) {
            return candidate(q, g);
        }

        const double hq = 1.0e-4 * std::max(1.0, std::abs(q));
        const double hg = 1.0e-4 * std::max(1.0, std::abs(g));
        const auto rq = residual(q + hq, g);
        const auto rg = residual(q, g + hg);
        const double a00 = (rq[0] - r[0]) / hq;
        const double a10 = (rq[1] - r[1]) / hq;
        const double a01 = (rg[0] - r[0]) / hg;
        const double a11 = (rg[1] - r[1]) / hg;
        const double det = a00 * a11 - a01 * a10;
        if (!std::isfinite(det) || std::abs(det) < 1.0e-12) break;

        const double dq = (-r[0] * a11 + a01 * r[1]) / det;
        const double dg = (a10 * r[0] - a00 * r[1]) / det;
        q = std::clamp(q + std::clamp(dq, -0.25, 0.25), -1.5, 1.5);
        g = std::clamp(g + std::clamp(dg, -0.25, 0.25), -1.5, 1.5);
    }

    TensorState2D fallback = tensor_state_with_normal_stress(
        state, mat, n, target_sigma_nn, target_un);
    return fallback;
}

inline TensorState2D make_solid_state_from_config_2d(
    const FluidSolidStateConfig2D& state,
    const TensorMaterial& mat,
    const std::array<double, 2>& n)
{
    TensorState2D base = tensor_cons_from_F(
        state.rho,
        state.un * n[0],
        state.un * n[1],
        mat.T0,
        local_stretch_F(n, 1.0, 1.0),
        mat);
    enforce_tensor(base, mat);
    if (state.has_sigma_nn) {
        base = tensor_state_with_normal_stress(base, mat, n, state.sigma_nn, state.un);
    }
    return base;
}

inline Conserved<2> fluid_state_from_pressure_normal_velocity_2d(
    const Conserved<2>& state,
    const EOSParams& params,
    const std::array<double, 2>& n,
    double pressure,
    double target_un)
{
    const ::Primitive<2> P = ::cons_to_prim<2, MaterialEOS>(state, params);
    const std::array<double, 2> s = tangent_from_normal(n);
    const double us = P.vel[0] * s[0] + P.vel[1] * s[1];
    const double invariant = MaterialEOS::entropy_invariant(P.rho, P.p, params);
    const double bounded_pressure = std::max(pressure, 1.0e-12);
    ::Primitive<2> out{};
    out.rho = MaterialEOS::density_from_p_invariant(bounded_pressure, invariant, params);
    out.vel[0] = target_un * n[0] + us * s[0];
    out.vel[1] = target_un * n[1] + us * s[1];
    out.p = bounded_pressure;
    return ::prim_to_cons<2, MaterialEOS>(out, params);
}

inline FluidSolidInterfaceState2D fluid_solid_rgfm_interface_states_2d(
    const Conserved<2>& fluid_state,
    const TensorState2D& solid_state,
    const EOSParams& fluid_params,
    const TensorMaterial& solid_mat,
    const std::array<double, 2>& normal_fluid_to_solid)
{
    const std::array<double, 2> n = normalised_pair(normal_fluid_to_solid);
    const ::Primitive<2> Pf = ::cons_to_prim<2, MaterialEOS>(fluid_state, fluid_params);
    const TensorPrim2D Ps = tensor_prim(solid_state, solid_mat);
    const double cf = MaterialEOS::template sound_speed<2>(fluid_state, fluid_params);
    const double Zf = std::max(Pf.rho * cf, 1.0e-12);
    const double Zs = std::max(Ps.rho * Ps.wave_speed, 1.0e-12);
    const double uf_n = Pf.vel[0] * n[0] + Pf.vel[1] * n[1];
    const double us_n = Ps.vel[0] * n[0] + Ps.vel[1] * n[1];
    const double sigma_nn = normal_stress_from_tensor(Ps, n);
    const double p0 = std::max(Pf.p, 1.0e-12);

    const double un_star = (p0 + sigma_nn + Zf * uf_n + Zs * us_n) / (Zf + Zs);
    const double p_star = std::max(p0 + Zf * (uf_n - un_star), 1.0e-12);
    const double sigma_star = -p_star;

    FluidSolidInterfaceState2D out{};
    out.normal_velocity = un_star;
    out.pressure = p_star;
    out.solid_normal_stress = sigma_star;
    out.fluid_interface = fluid_state_from_pressure_normal_velocity_2d(
        fluid_state, fluid_params, n, p_star, un_star);
    out.solid_interface = tensor_state_with_fluid_interface_traction(
        solid_state, solid_mat, n, sigma_star, 0.0, un_star);
    return out;
}

inline void enforce_fluid_state_2d(Conserved<2>& U, const EOSParams& params)
{
    if (!std::isfinite(U.rho) || U.rho <= 1.0e-12) {
        U.rho = 1.0e-12;
    }
    for (double& m : U.mom) {
        if (!std::isfinite(m)) m = 0.0;
    }
    if (!std::isfinite(U.E)) {
        U.E = 1.0e-12;
    }
    const ::Primitive<2> P = ::cons_to_prim<2, MaterialEOS>(U, params);
    if (!std::isfinite(P.p) || P.p <= 1.0e-12) {
        ::Primitive<2> fixed{};
        fixed.rho = U.rho;
        fixed.vel[0] = U.mom[0] / U.rho;
        fixed.vel[1] = U.mom[1] / U.rho;
        fixed.p = 1.0e-12;
        U = ::prim_to_cons<2, MaterialEOS>(fixed, params);
    }
}

inline double max_fluid_signal_speed_2d(
    const std::vector<Conserved<2>>& U,
    const std::vector<char>& fluid_mask,
    const EOSParams& params)
{
    double speed = 1.0;
    for (std::size_t id = 0; id < U.size(); ++id) {
        if (!fluid_mask[id]) continue;
        const ::Primitive<2> P = ::cons_to_prim<2, MaterialEOS>(U[id], params);
        const double c = MaterialEOS::template sound_speed<2>(U[id], params);
        speed = std::max(speed, std::hypot(P.vel[0], P.vel[1]) + c);
    }
    return speed;
}

inline double max_solid_signal_speed_2d(
    const std::vector<TensorState2D>& U,
    const std::vector<char>& solid_mask,
    const TensorMaterial& mat)
{
    double speed = 1.0;
    for (std::size_t id = 0; id < U.size(); ++id) {
        if (!solid_mask[id]) continue;
        const TensorPrim2D P = tensor_prim(U[id], mat);
        speed = std::max(speed, std::hypot(P.vel[0], P.vel[1]) + P.wave_speed);
    }
    return speed;
}

inline std::string fluid_solid_base_path(const FluidSolidConfig2D& cfg)
{
    if (cfg.output_dir.empty()) return cfg.output_prefix;
    return cfg.output_dir + "/" + cfg.output_prefix + "/" + cfg.output_prefix;
}

inline void initialise_solid_material_coordinates(
    std::vector<double>& x_material,
    std::vector<double>& y_material,
    const std::vector<char>& solid_mask,
    const FluidSolidConfig2D& cfg)
{
    const int nx = cfg.cells[0];
    const int ny = cfg.cells[1];
    const double dx = (cfg.domain_max[0] - cfg.domain_min[0]) / nx;
    const double dy = (cfg.domain_max[1] - cfg.domain_min[1]) / ny;
    const std::array<double, 2> n = cfg.interface_normal;
    const std::array<double, 2> s = tangent_from_normal(n);

    x_material.assign(nx * ny, std::numeric_limits<double>::quiet_NaN());
    y_material.assign(nx * ny, std::numeric_limits<double>::quiet_NaN());
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = hidx(i, j, nx);
            if (!solid_mask[id]) continue;
            const double x = cfg.domain_min[0] + (static_cast<double>(i) + 0.5) * dx;
            const double y = cfg.domain_min[1] + (static_cast<double>(j) + 0.5) * dy;
            x_material[id] = n[0] * x + n[1] * y;
            y_material[id] = s[0] * x + s[1] * y;
        }
    }
}

inline void advect_solid_material_coordinates(
    std::vector<double>& x_material,
    std::vector<double>& y_material,
    const std::vector<TensorState2D>& solid_cells,
    const std::vector<char>& solid_mask,
    const TensorMaterial& mat,
    int nx,
    int ny,
    double dx,
    double dy,
    double dt)
{
    auto sample = [&](const std::vector<double>& values, int i, int j, int fallback_id) {
        i = std::clamp(i, 0, nx - 1);
        j = std::clamp(j, 0, ny - 1);
        const int id = hidx(i, j, nx);
        if (!solid_mask[id] || !std::isfinite(values[id])) {
            return values[fallback_id];
        }
        return values[id];
    };

    std::vector<double> next_x = x_material;
    std::vector<double> next_y = y_material;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = hidx(i, j, nx);
            if (!solid_mask[id]) continue;

            const TensorPrim2D P = tensor_prim(solid_cells[id], mat);
            auto update = [&](const std::vector<double>& values) {
                const double here = values[id];
                if (!std::isfinite(here)) return here;

                const double left = sample(values, i - 1, j, id);
                const double right = sample(values, i + 1, j, id);
                const double down = sample(values, i, j - 1, id);
                const double up = sample(values, i, j + 1, id);
                const double dqx = P.vel[0] >= 0.0 ? (here - left) / dx : (right - here) / dx;
                const double dqy = P.vel[1] >= 0.0 ? (here - down) / dy : (up - here) / dy;
                return here - dt * (P.vel[0] * dqx + P.vel[1] * dqy);
            };

            next_x[id] = update(x_material);
            next_y[id] = update(y_material);
        }
    }
    x_material.swap(next_x);
    y_material.swap(next_y);
}

inline void apply_masked_tensor_compatibility_and_plasticity(
    std::vector<TensorState2D>& next,
    const std::vector<TensorState2D>& old,
    const std::vector<char>& solid_mask,
    const TensorMaterial& mat,
    int nx,
    int ny,
    double dx,
    double dy,
    double dt)
{
    auto sample = [&](int i, int j, int fallback_id) {
        i = std::clamp(i, 0, nx - 1);
        j = std::clamp(j, 0, ny - 1);
        const int id = hidx(i, j, nx);
        return solid_mask[id] ? old[id] : old[fallback_id];
    };

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = hidx(i, j, nx);
            if (!solid_mask[id]) continue;

            TensorState2D& U = next[id];
            const TensorPrim2D Pold = tensor_prim(old[id], mat);
            std::array<double, 3> beta{};
            for (int col = 0; col < 3; ++col) {
                beta[col] =
                    (sample(i + 1, j, id).rhoF[3 * 0 + col] -
                     sample(i - 1, j, id).rhoF[3 * 0 + col]) / (2.0 * dx) +
                    (sample(i, j + 1, id).rhoF[3 * 1 + col] -
                     sample(i, j - 1, id).rhoF[3 * 1 + col]) / (2.0 * dy);
            }

            const std::array<double, 3> vel{Pold.vel[0], Pold.vel[1], 0.0};
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    U.rhoF[3 * row + col] -= dt * vel[row] * beta[col];
                }
            }
            enforce_tensor(U, mat);
            apply_tensor_plastic_relaxation(U, mat, dt);
        }
    }
}

inline void advance_masked_rgfm_tensor_sweep(
    std::vector<TensorState2D>& U,
    const std::vector<Conserved<2>>& fluid,
    const std::vector<char>& fluid_mask,
    const std::vector<char>& solid_mask,
    const FluidSolidConfig2D& cfg,
    double h,
    double dt,
    int dir)
{
    const int nx = cfg.cells[0];
    const int ny = cfg.cells[1];
    const int lines = dir == 0 ? ny : nx;
    const int n = dir == 0 ? nx : ny;
    auto index = [&](int a, int line) {
        return dir == 0 ? hidx(a, line, nx) : hidx(line, a, nx);
    };
    auto interface_state = [&](int fluid_id, int solid_id) {
        return fluid_solid_rgfm_interface_states_2d(
            fluid[fluid_id],
            U[solid_id],
            cfg.fluid,
            cfg.solid,
            cfg.interface_normal).solid_interface;
    };

    for (int line = 0; line < lines; ++line) {
        int start = 0;
        while (start < n) {
            while (start < n && !solid_mask[index(start, line)]) {
                ++start;
            }
            if (start >= n) break;

            int end = start;
            while (end + 1 < n && solid_mask[index(end + 1, line)]) {
                ++end;
            }

            const int segment_cells = end - start + 1;
            std::vector<TensorState2D> ext(segment_cells + 2);
            for (int a = 0; a < segment_cells; ++a) {
                ext[a + 1] = U[index(start + a, line)];
            }

            const int left_solid = index(start, line);
            const int right_solid = index(end, line);
            const int left_neighbour = start > 0 ? index(start - 1, line) : -1;
            const int right_neighbour = end + 1 < n ? index(end + 1, line) : -1;
            const bool left_fluid = left_neighbour >= 0 && fluid_mask[left_neighbour];
            const bool right_fluid = right_neighbour >= 0 && fluid_mask[right_neighbour];
            ext[0] = left_fluid ? interface_state(left_neighbour, left_solid) : ext[1];
            ext[segment_cells + 1] =
                right_fluid ? interface_state(right_neighbour, right_solid) : ext[segment_cells];

            std::vector<TensorState2D> left_cell(segment_cells + 2);
            std::vector<TensorState2D> right_cell(segment_cells + 2);
            std::vector<TensorState2D> half_cell(segment_cells + 2);
            half_cell[0] = ext[0];
            half_cell[segment_cells + 1] = ext[segment_cells + 1];
            for (int a = 1; a <= segment_cells; ++a) {
                const TensorSlope2D slope = scale_tensor_slope(
                    limited_tensor_slope(ext[a - 1], ext[a], ext[a + 1]),
                    tensor_flattening_scale(ext[a - 1], ext[a], ext[a + 1], cfg.solid));
                left_cell[a] = add_tensor_slope(ext[a], slope, -0.5);
                right_cell[a] = add_tensor_slope(ext[a], slope, 0.5);
                enforce_tensor(left_cell[a], cfg.solid);
                enforce_tensor(right_cell[a], cfg.solid);
                half_cell[a] = hancock_predict_tensor_cell(
                    ext[a], left_cell[a], right_cell[a], cfg.solid, dt, h, dir);
            }
            half_cell[0] = ext[0];
            half_cell[segment_cells + 1] = ext[segment_cells + 1];

            std::vector<TensorState2D> left_half(segment_cells + 2);
            std::vector<TensorState2D> right_half(segment_cells + 2);
            for (int a = 1; a <= segment_cells; ++a) {
                const TensorSlope2D slope = scale_tensor_slope(
                    limited_tensor_slope(half_cell[a - 1], half_cell[a], half_cell[a + 1]),
                    tensor_flattening_scale(half_cell[a - 1], half_cell[a], half_cell[a + 1], cfg.solid));
                left_half[a] = add_tensor_slope(half_cell[a], slope, -0.5);
                right_half[a] = add_tensor_slope(half_cell[a], slope, 0.5);
                enforce_tensor(left_half[a], cfg.solid);
                enforce_tensor(right_half[a], cfg.solid);
            }

            std::vector<TensorState2D> flux(segment_cells + 1);
            std::vector<std::array<double, 3>> faceF(segment_cells + 1);
            if (left_fluid) {
                flux[0] = tensor_flux(ext[0], cfg.solid, dir);
                for (int col = 0; col < 3; ++col) {
                    faceF[0][col] = ext[0].rhoF[3 * dir + col];
                }
            }
            else {
                flux[0] = tensor_rusanov(ext[0], left_half[1], cfg.solid, dir);
                for (int col = 0; col < 3; ++col) {
                    faceF[0][col] = 0.5 * (
                        ext[0].rhoF[3 * dir + col] +
                        left_half[1].rhoF[3 * dir + col]);
                }
            }

            for (int a = 1; a < segment_cells; ++a) {
                flux[a] = tensor_rusanov(right_half[a], left_half[a + 1], cfg.solid, dir);
                for (int col = 0; col < 3; ++col) {
                    faceF[a][col] = 0.5 * (
                        right_half[a].rhoF[3 * dir + col] +
                        left_half[a + 1].rhoF[3 * dir + col]);
                }
            }

            if (right_fluid) {
                flux[segment_cells] = tensor_flux(ext[segment_cells + 1], cfg.solid, dir);
                for (int col = 0; col < 3; ++col) {
                    faceF[segment_cells][col] = ext[segment_cells + 1].rhoF[3 * dir + col];
                }
            }
            else {
                flux[segment_cells] =
                    tensor_rusanov(right_half[segment_cells], ext[segment_cells + 1], cfg.solid, dir);
                for (int col = 0; col < 3; ++col) {
                    faceF[segment_cells][col] = 0.5 * (
                        right_half[segment_cells].rhoF[3 * dir + col] +
                        ext[segment_cells + 1].rhoF[3 * dir + col]);
                }
            }

            for (int a = 0; a < segment_cells; ++a) {
                const int cell = index(start + a, line);
                TensorState2D next = U[cell] - (dt / h) * (flux[a + 1] - flux[a]);
                const TensorPrim2D Pold = tensor_prim(U[cell], cfg.solid);
                const std::array<double, 3> vel{Pold.vel[0], Pold.vel[1], 0.0};
                for (int col = 0; col < 3; ++col) {
                    const double beta = (faceF[a + 1][col] - faceF[a][col]) / h;
                    for (int row = 0; row < 3; ++row) {
                        next.rhoF[3 * row + col] -= dt * vel[row] * beta;
                    }
                }
                enforce_tensor(next, cfg.solid);
                U[cell] = next;
            }

            start = end + 1;
        }
    }
}

inline void write_fluid_solid_csv_2d(
    const std::string& filename,
    const std::vector<Conserved<2>>& fluid,
    const std::vector<TensorState2D>& solid_cells,
    const std::vector<double>& phi,
    const std::vector<double>& x_material,
    const std::vector<double>& y_material,
    const FluidSolidConfig2D& cfg,
    double time)
{
    ensure_directory(std::filesystem::path(filename).parent_path().string());
    std::ofstream out(filename);
    out << std::setprecision(17);
    out << "x,y,phi,x_lag,y_lag,x_mat,y_mat,material,rho,u,v,p,sigma_xx,sigma_xy,sigma_yy,"
        << "sigma_nn,sigma_ss,sigma_sn,vn,time\n";

    const int nx = cfg.cells[0];
    const int ny = cfg.cells[1];
    const double dx = (cfg.domain_max[0] - cfg.domain_min[0]) / nx;
    const double dy = (cfg.domain_max[1] - cfg.domain_min[1]) / ny;
    const std::array<double, 2> n = cfg.interface_normal;
    const std::array<double, 2> s = tangent_from_normal(n);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = hidx(i, j, nx);
            const double x = cfg.domain_min[0] + (static_cast<double>(i) + 0.5) * dx;
            const double y = cfg.domain_min[1] + (static_cast<double>(j) + 0.5) * dy;
            const double x_lag = n[0] * x + n[1] * y;
            const double y_lag = s[0] * x + s[1] * y;
            if (is_fluid_phi(phi[id])) {
                const ::Primitive<2> P = ::cons_to_prim<2, MaterialEOS>(fluid[id], cfg.fluid);
                const double vn = P.vel[0] * n[0] + P.vel[1] * n[1];
                out << x << "," << y << "," << phi[id] << ","
                    << x_lag << "," << y_lag << ",nan,nan,fluid,"
                    << P.rho << "," << P.vel[0] << "," << P.vel[1] << "," << P.p
                    << ",0,0,0,0,0,0," << vn << "," << time << "\n";
            }
            else {
                const TensorPrim2D P = tensor_prim(solid_cells[id], cfg.solid);
                const double sigma_nn = normal_stress_from_tensor(P, n);
                const double sigma_ss = tangential_stress_from_tensor(P, n);
                const double sigma_sn = shear_stress_from_tensor(P, n);
                const double vn = P.vel[0] * n[0] + P.vel[1] * n[1];
                out << x << "," << y << "," << phi[id] << ","
                    << x_lag << "," << y_lag << ","
                    << x_material[id] << "," << y_material[id] << ",solid,"
                    << P.rho << "," << P.vel[0] << "," << P.vel[1] << ",0,"
                    << P.sigma[0] << "," << 0.5 * (P.sigma[1] + P.sigma[3]) << ","
                    << P.sigma[4] << "," << sigma_nn << "," << sigma_ss << ","
                    << sigma_sn << "," << vn << "," << time << "\n";
            }
        }
    }
}

inline int normal_tangent_sample_index(
    double xi,
    double xi_min,
    double dxi,
    int cells)
{
    const int idx = static_cast<int>(std::floor((xi - xi_min) / dxi));
    return std::clamp(idx, 0, cells - 1);
}

inline void write_fluid_solid_normal_tangent_csv_2d(
    const std::string& filename,
    const std::vector<Conserved<2>>& fluid_line,
    const std::vector<TensorState2D>& solid_line,
    const FluidSolidConfig2D& cfg,
    double xi_min,
    double dxi,
    double time)
{
    ensure_directory(std::filesystem::path(filename).parent_path().string());
    std::ofstream out(filename);
    out << std::setprecision(17);
    out << "x,y,phi,x_lag,y_lag,x_mat,y_mat,material,rho,u,v,p,sigma_xx,sigma_xy,sigma_yy,"
        << "sigma_nn,sigma_ss,sigma_sn,vn,time\n";

    const int nx = cfg.cells[0];
    const int ny = cfg.cells[1];
    const int normal_cells = static_cast<int>(fluid_line.size());
    const double dx = (cfg.domain_max[0] - cfg.domain_min[0]) / nx;
    const double dy = (cfg.domain_max[1] - cfg.domain_min[1]) / ny;
    const std::array<double, 2> n = cfg.interface_normal;
    const std::array<double, 2> s = tangent_from_normal(n);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x = cfg.domain_min[0] + (static_cast<double>(i) + 0.5) * dx;
            const double y = cfg.domain_min[1] + (static_cast<double>(j) + 0.5) * dy;
            const double x_lag = n[0] * x + n[1] * y;
            const double y_lag = s[0] * x + s[1] * y;
            const double phi = x_lag - cfg.interface_offset;
            const int k = normal_tangent_sample_index(x_lag, xi_min, dxi, normal_cells);

            if (is_fluid_phi(phi)) {
                const ::Primitive<2> P = ::cons_to_prim<2, MaterialEOS>(fluid_line[k], cfg.fluid);
                const double u = P.vel[0] * n[0] + P.vel[1] * s[0];
                const double v = P.vel[0] * n[1] + P.vel[1] * s[1];
                out << x << "," << y << "," << phi << ","
                    << x_lag << "," << y_lag << ",nan,nan,fluid,"
                    << P.rho << "," << u << "," << v << "," << P.p
                    << ",0,0,0,0,0,0," << P.vel[0] << "," << time << "\n";
            }
            else {
                const TensorPrim2D P = tensor_prim(solid_line[k], cfg.solid);
                const double un = P.vel[0];
                const double us = P.vel[1];
                const double u = un * n[0] + us * s[0];
                const double v = un * n[1] + us * s[1];
                const double snn = P.sigma[0];
                const double sss = P.sigma[4];
                const double ssn = 0.5 * (P.sigma[1] + P.sigma[3]);
                const double sigma_xx =
                    snn * n[0] * n[0] + sss * s[0] * s[0] +
                    ssn * (n[0] * s[0] + s[0] * n[0]);
                const double sigma_xy =
                    snn * n[0] * n[1] + sss * s[0] * s[1] +
                    ssn * (n[0] * s[1] + s[0] * n[1]);
                const double sigma_yy =
                    snn * n[1] * n[1] + sss * s[1] * s[1] +
                    ssn * (n[1] * s[1] + s[1] * n[1]);
                out << x << "," << y << "," << phi << ","
                    << x_lag << "," << y_lag << ","
                    << x_lag << "," << y_lag << ",solid,"
                    << P.rho << "," << u << "," << v << ",0,"
                    << sigma_xx << "," << sigma_xy << "," << sigma_yy << ","
                    << snn << "," << sss << "," << ssn << "," << un << "," << time << "\n";
            }
        }
    }
}

inline int run_fluid_solid_rgfm_2d_normal_tangent(const FluidSolidConfig2D& cfg)
{
    const int cells = cfg.normal_tangent_cells > 1 ? cfg.normal_tangent_cells : cfg.cells[0];
    const double xi_min = cfg.domain_min[0];
    const double xi_max = cfg.domain_max[0];
    const double dxi = (xi_max - xi_min) / cells;
    const int count = cells;
    FluidSolidConfig2D line_cfg = cfg;
    line_cfg.cells = {cells, 1};
    line_cfg.domain_min = {xi_min, 0.0};
    line_cfg.domain_max = {xi_max, 1.0};
    line_cfg.interface_normal = {1.0, 0.0};

    std::vector<char> fluid_mask(count, 0);
    std::vector<char> solid_mask(count, 0);
    std::vector<Conserved<2>> fluid(
        count, make_fluid_state_from_config_2d(cfg.fluid_initial, cfg.fluid, line_cfg.interface_normal));
    std::vector<TensorState2D> solid_line(
        count, make_solid_state_from_config_2d(cfg.solid_initial, cfg.solid, line_cfg.interface_normal));

    for (int i = 0; i < cells; ++i) {
        const double xi = xi_min + (static_cast<double>(i) + 0.5) * dxi;
        const bool is_fluid = xi - cfg.interface_offset < 0.0;
        fluid_mask[i] = is_fluid ? 1 : 0;
        solid_mask[i] = is_fluid ? 0 : 1;
    }

    const std::string base = fluid_solid_base_path(cfg);
    const std::string nlabel = "_N" + std::to_string(cfg.cells[0]) + "x" + std::to_string(cfg.cells[1]);
    double time = 0.0;
    int steps = 0;
    std::size_t next_output = 0;
    const auto wall_start = std::chrono::steady_clock::now();
    const std::array<double, 2> ex{1.0, 0.0};

    while (time < cfg.tfinal - 1.0e-15) {
        const double sf = max_fluid_signal_speed_2d(fluid, fluid_mask, cfg.fluid);
        const double ss = max_solid_signal_speed_2d(solid_line, solid_mask, cfg.solid);
        const double dt = std::min(
            cfg.tfinal - time,
            cfg.cfl * dxi / std::max(std::max(sf, ss), 1.0e-30));
        if (!std::isfinite(dt) || dt <= 0.0) {
            throw std::runtime_error("Normal-tangent fluid-solid 2D timestep collapsed");
        }

        std::vector<Conserved<2>> rgfm_fluid = fluid;
        std::vector<TensorState2D> rgfm_solid = solid_line;
        std::vector<Conserved<2>> flux_f(cells + 1);
        auto face_interface = [&](int fluid_id, int solid_id) {
            const auto state = fluid_solid_rgfm_interface_states_2d(
                fluid[fluid_id],
                solid_line[solid_id],
                cfg.fluid,
                cfg.solid,
                line_cfg.interface_normal);
            rgfm_fluid[fluid_id] = state.fluid_interface;
            rgfm_solid[solid_id] = state.solid_interface;
            return state;
        };

        for (int f = 0; f <= cells; ++f) {
            if (f == 0) {
                flux_f[f] = compute_flux_normal<2, MaterialEOS>(rgfm_fluid[0], ex, cfg.fluid);
                continue;
            }
            if (f == cells) {
                flux_f[f] = compute_flux_normal<2, MaterialEOS>(rgfm_fluid[cells - 1], ex, cfg.fluid);
                continue;
            }
            const int left = f - 1;
            const int right = f;
            if (fluid_mask[left] && fluid_mask[right]) {
                flux_f[f] = hll_flux_normal<2, MaterialEOS>(
                    rgfm_fluid[left], rgfm_fluid[right], ex, cfg.fluid, cfg.fluid);
            }
            else if (fluid_mask[left] && solid_mask[right]) {
                const auto state = face_interface(left, right);
                flux_f[f] = compute_flux_normal<2, MaterialEOS>(state.fluid_interface, ex, cfg.fluid);
            }
            else if (solid_mask[left] && fluid_mask[right]) {
                const auto state = face_interface(right, left);
                flux_f[f] = compute_flux_normal<2, MaterialEOS>(state.fluid_interface, ex, cfg.fluid);
            }
        }

        std::vector<Conserved<2>> next_fluid = fluid;
        std::vector<TensorState2D> next_solid = rgfm_solid;
        advance_masked_rgfm_tensor_sweep(
            next_solid, fluid, fluid_mask, solid_mask, line_cfg, dxi, dt, 0);
        for (int i = 0; i < cells; ++i) {
            if (fluid_mask[i]) {
                next_fluid[i] = rgfm_fluid[i] - (dt / dxi) * (flux_f[i + 1] - flux_f[i]);
                enforce_fluid_state_2d(next_fluid[i], cfg.fluid);
            }
            if (solid_mask[i]) {
                apply_tensor_plastic_relaxation(next_solid[i], cfg.solid, dt);
            }
        }

        fluid.swap(next_fluid);
        solid_line.swap(next_solid);
        time += dt;
        ++steps;
        while (next_output < cfg.output_times.size() &&
               time >= cfg.output_times[next_output] - 1.0e-15) {
            const double t_out = cfg.output_times[next_output];
            write_fluid_solid_normal_tangent_csv_2d(
                base + fluid_solid_time_suffix(t_out) + nlabel + ".csv",
                fluid, solid_line, cfg, xi_min, dxi, time);
            ++next_output;
        }
        if (steps > 200000) {
            throw std::runtime_error("Normal-tangent fluid-solid 2D exceeded 200000 steps");
        }
    }

    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_seconds =
        std::chrono::duration<double>(wall_end - wall_start).count();
    const std::string final_path = base + nlabel + ".csv";
    write_fluid_solid_normal_tangent_csv_2d(
        final_path, fluid, solid_line, cfg, xi_min, dxi, time);

    std::ofstream runtime(base + nlabel + "_runtime.txt");
    runtime << "steps = " << steps << "\n";
    runtime << "final_time = " << time << "\n";
    runtime << "wall_time_seconds = " << wall_seconds << "\n";
    runtime << "cells = " << cfg.cells[0] * cfg.cells[1] << "\n";
    runtime << "model = fluid_solid_rgfm_2d_normal_tangent\n";
    runtime << "normal_cells = " << cells << "\n";
    runtime << "interface_normal = " << cfg.interface_normal[0] << ","
            << cfg.interface_normal[1] << "\n";
    runtime << "interface_offset = " << cfg.interface_offset << "\n";

    std::cout << "Written: " << final_path << "\n";
    std::cout << "steps = " << steps << ", final_time = " << time << "\n";
    return 0;
}

inline int run_fluid_solid_rgfm_2d(const std::string& config_file)
{
    const FluidSolidConfig2D cfg = load_fluid_solid_config_2d(config_file);
    if (cfg.normal_tangent_update) {
        return run_fluid_solid_rgfm_2d_normal_tangent(cfg);
    }
    const int nx = cfg.cells[0];
    const int ny = cfg.cells[1];
    const double dx = (cfg.domain_max[0] - cfg.domain_min[0]) / nx;
    const double dy = (cfg.domain_max[1] - cfg.domain_min[1]) / ny;
    const int count = nx * ny;

    std::vector<double> phi(count, 0.0);
    std::vector<char> fluid_mask(count, 0);
    std::vector<char> solid_mask(count, 0);
    std::vector<Conserved<2>> fluid(
        count, make_fluid_state_from_config_2d(cfg.fluid_initial, cfg.fluid, cfg.interface_normal));
    std::vector<TensorState2D> solid_cells(
        count, make_solid_state_from_config_2d(cfg.solid_initial, cfg.solid, cfg.interface_normal));
    std::vector<double> solid_x_material;
    std::vector<double> solid_y_material;

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int id = hidx(i, j, nx);
            const double x = cfg.domain_min[0] + (static_cast<double>(i) + 0.5) * dx;
            const double y = cfg.domain_min[1] + (static_cast<double>(j) + 0.5) * dy;
            phi[id] = signed_phi(cfg, x, y);
            fluid_mask[id] = is_fluid_phi(phi[id]) ? 1 : 0;
            solid_mask[id] = fluid_mask[id] ? 0 : 1;
        }
    }
    initialise_solid_material_coordinates(
        solid_x_material, solid_y_material, solid_mask, cfg);

    const std::string base = fluid_solid_base_path(cfg);
    const std::string nlabel = "_N" + std::to_string(nx) + "x" + std::to_string(ny);
    double time = 0.0;
    int steps = 0;
    std::size_t next_output = 0;
    const auto wall_start = std::chrono::steady_clock::now();

    while (time < cfg.tfinal - 1.0e-15) {
        const double sf = max_fluid_signal_speed_2d(fluid, fluid_mask, cfg.fluid);
        const double ss = max_solid_signal_speed_2d(solid_cells, solid_mask, cfg.solid);
        const double denom = std::max(sf, ss) * (1.0 / dx + 1.0 / dy);
        const double dt = std::min(cfg.tfinal - time, cfg.cfl / std::max(denom, 1.0e-30));
        if (!std::isfinite(dt) || dt <= 0.0) {
            throw std::runtime_error("Fluid-solid 2D rGFM timestep collapsed");
        }

        std::vector<Conserved<2>> rgfm_fluid = fluid;
        std::vector<TensorState2D> rgfm_solid = solid_cells;
        std::vector<Conserved<2>> flux_fx((nx + 1) * ny);
        std::vector<Conserved<2>> flux_fy(nx * (ny + 1));
        const std::array<double, 2> ex{1.0, 0.0};
        const std::array<double, 2> ey{0.0, 1.0};

        auto face_interface = [&](int fluid_id, int solid_id) {
            const auto state = fluid_solid_rgfm_interface_states_2d(
                fluid[fluid_id],
                solid_cells[solid_id],
                cfg.fluid,
                cfg.solid,
                cfg.interface_normal);
            rgfm_fluid[fluid_id] = state.fluid_interface;
            rgfm_solid[solid_id] = state.solid_interface;
            return state;
        };

        for (int j = 0; j < ny; ++j) {
            for (int f = 0; f <= nx; ++f) {
                const int fid = j * (nx + 1) + f;
                if (f == 0) {
                    const int id = hidx(0, j, nx);
                    flux_fx[fid] = compute_flux_normal<2, MaterialEOS>(rgfm_fluid[id], ex, cfg.fluid);
                    continue;
                }
                if (f == nx) {
                    const int id = hidx(nx - 1, j, nx);
                    flux_fx[fid] = compute_flux_normal<2, MaterialEOS>(rgfm_fluid[id], ex, cfg.fluid);
                    continue;
                }
                const int left = hidx(f - 1, j, nx);
                const int right = hidx(f, j, nx);
                if (fluid_mask[left] && fluid_mask[right]) {
                    flux_fx[fid] = hll_flux_normal<2, MaterialEOS>(
                        rgfm_fluid[left], rgfm_fluid[right], ex, cfg.fluid, cfg.fluid);
                }
                else if (solid_mask[left] && solid_mask[right]) {
                    continue;
                }
                else if (fluid_mask[left] && solid_mask[right]) {
                    const auto state = face_interface(left, right);
                    flux_fx[fid] = compute_flux_normal<2, MaterialEOS>(state.fluid_interface, ex, cfg.fluid);
                }
                else {
                    const auto state = face_interface(right, left);
                    flux_fx[fid] = compute_flux_normal<2, MaterialEOS>(state.fluid_interface, ex, cfg.fluid);
                }
            }
        }

        for (int f = 0; f < nx; ++f) {
            for (int j = 0; j <= ny; ++j) {
                const int fid = j * nx + f;
                if (j == 0) {
                    const int id = hidx(f, 0, nx);
                    flux_fy[fid] = compute_flux_normal<2, MaterialEOS>(rgfm_fluid[id], ey, cfg.fluid);
                    continue;
                }
                if (j == ny) {
                    const int id = hidx(f, ny - 1, nx);
                    flux_fy[fid] = compute_flux_normal<2, MaterialEOS>(rgfm_fluid[id], ey, cfg.fluid);
                    continue;
                }
                const int down = hidx(f, j - 1, nx);
                const int up = hidx(f, j, nx);
                if (fluid_mask[down] && fluid_mask[up]) {
                    flux_fy[fid] = hll_flux_normal<2, MaterialEOS>(
                        rgfm_fluid[down], rgfm_fluid[up], ey, cfg.fluid, cfg.fluid);
                }
                else if (solid_mask[down] && solid_mask[up]) {
                    continue;
                }
                else if (fluid_mask[down] && solid_mask[up]) {
                    const auto state = face_interface(down, up);
                    flux_fy[fid] = compute_flux_normal<2, MaterialEOS>(state.fluid_interface, ey, cfg.fluid);
                }
                else {
                    const auto state = face_interface(up, down);
                    flux_fy[fid] = compute_flux_normal<2, MaterialEOS>(state.fluid_interface, ey, cfg.fluid);
                }
            }
        }

        std::vector<Conserved<2>> next_fluid = fluid;
        std::vector<TensorState2D> next_solid = rgfm_solid;
        advance_masked_rgfm_tensor_sweep(
            next_solid, fluid, fluid_mask, solid_mask, cfg, dx, dt, 0);
        advance_masked_rgfm_tensor_sweep(
            next_solid, fluid, fluid_mask, solid_mask, cfg, dy, dt, 1);
        for (int id = 0; id < count; ++id) {
            if (solid_mask[id]) {
                apply_tensor_plastic_relaxation(next_solid[id], cfg.solid, dt);
            }
        }
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int id = hidx(i, j, nx);
                if (fluid_mask[id]) {
                    next_fluid[id] = rgfm_fluid[id]
                        - (dt / dx) * (flux_fx[j * (nx + 1) + i + 1] - flux_fx[j * (nx + 1) + i])
                        - (dt / dy) * (flux_fy[(j + 1) * nx + i] - flux_fy[j * nx + i]);
                    enforce_fluid_state_2d(next_fluid[id], cfg.fluid);
                }
            }
        }

        advect_solid_material_coordinates(
            solid_x_material,
            solid_y_material,
            solid_cells,
            solid_mask,
            cfg.solid,
            nx,
            ny,
            dx,
            dy,
            dt);
        fluid.swap(next_fluid);
        solid_cells.swap(next_solid);
        time += dt;
        ++steps;
        while (next_output < cfg.output_times.size() &&
               time >= cfg.output_times[next_output] - 1.0e-15) {
            const double t_out = cfg.output_times[next_output];
            write_fluid_solid_csv_2d(
                base + fluid_solid_time_suffix(t_out) + nlabel + ".csv",
                fluid, solid_cells, phi, solid_x_material, solid_y_material, cfg, time);
            ++next_output;
        }
        if (steps > 200000) {
            throw std::runtime_error("Fluid-solid 2D rGFM exceeded 200000 steps");
        }
    }

    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_seconds =
        std::chrono::duration<double>(wall_end - wall_start).count();
    const std::string final_path = base + nlabel + ".csv";
    write_fluid_solid_csv_2d(
        final_path, fluid, solid_cells, phi, solid_x_material, solid_y_material, cfg, time);

    std::ofstream runtime(base + nlabel + "_runtime.txt");
    runtime << "steps = " << steps << "\n";
    runtime << "final_time = " << time << "\n";
    runtime << "wall_time_seconds = " << wall_seconds << "\n";
    runtime << "cells = " << nx * ny << "\n";
    runtime << "model = fluid_solid_rgfm_2d\n";
    runtime << "interface_normal = " << cfg.interface_normal[0] << ","
            << cfg.interface_normal[1] << "\n";
    runtime << "interface_offset = " << cfg.interface_offset << "\n";

    std::cout << "Written: " << final_path << "\n";
    std::cout << "steps = " << steps << ", final_time = " << time << "\n";
    return 0;
}

inline int run_fluid_solid_rgfm(const std::string& config_file)
{
    std::ifstream file(config_file);
    if (!file) {
        throw std::runtime_error("Cannot open fluid-solid config: " + config_file);
    }
    std::string line;
    while (std::getline(file, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);
        line = solid::text::trim(line);
        if (line.empty()) continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = solid::text::trim(line.substr(0, eq));
        if (key == "dimension") {
            const int dim = static_cast<int>(
                solid::text::parse_single_bracket_value(line.substr(eq + 1)));
            if (dim == 1) return run_fluid_solid_rgfm_1d(config_file);
            if (dim == 2) return run_fluid_solid_rgfm_2d(config_file);
            throw std::runtime_error("fluid_solid_rgfm supports dimension=1 or dimension=2");
        }
    }
    return run_fluid_solid_rgfm_1d(config_file);
}


} // namespace solid::barton
