#pragma once

// Barton CSV output helpers for 1D and tensor validation data.

#include "src/solid/elastoplastic/barton/advance.hpp"
#include "src/solid/elastoplastic/barton/initial_conditions.hpp"

// 1D CSV output.

// CSV output helpers for Barton 1D plate-impact validation data.

namespace solid::barton {

inline std::vector<State> initialise(const Config& cfg)
{
    std::vector<State> U(cfg.cells);
    const double dx = (cfg.domain_max - cfg.domain_min) / cfg.cells;
    for (int i = 0; i < cfg.cells; ++i) {
        const double x = cfg.domain_min + (i + 0.5) * dx;
        bool found = false;
        for (const auto& region : cfg.regions) {
            if (x >= region.x_min && x < region.x_max) {
                U[i] = prim_to_cons(region.rho, region.u, cfg.material);
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error("Barton initialisation: uncovered cell at x=" + std::to_string(x));
        }
    }
    return U;
}

inline std::string time_suffix(double time)
{
    std::ostringstream s;
    s << std::fixed << std::setprecision(3) << time * 1.0e6;
    std::string text = s.str();
    for (char& c : text) {
        if (c == '.') {
            c = 'p';
        }
    }
    return "_t" + text + "us";
}

inline void ensure_directory(const std::string& dir)
{
    if (!dir.empty()) {
        std::filesystem::create_directories(dir);
    }
}

inline void write_csv(const std::string& filename, const std::vector<State>& U, const Config& cfg)
{
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Cannot write Barton solid CSV: " + filename);
    }
    const double dx = (cfg.domain_max - cfg.domain_min) / cfg.cells;
    out << "x,rho,u,p,e,hydro_e,elastic_e,gxx,gyy,gzz,Fp_xx,Fp_yy,Fp_zz,"
        << "plastic_xx,sxx,syy,szz,normal_stress,equivalent_stress,plastic_strain,wave_speed\n";
    for (int i = 0; i < cfg.cells; ++i) {
        const double x = cfg.domain_min + (i + 0.5) * dx;
        const Primitive P = cons_to_prim(U[i], cfg.material);
        out << x << ","
            << P.rho << ","
            << P.u << ","
            << P.pressure << ","
            << P.internal_e << ","
            << cold_energy(P.rho, cfg.material) << ","
            << P.shear_e << ","
            << P.F11 << ","
            << P.F22 << ","
            << P.F33 << ","
            << 1.0 << ","
            << 1.0 << ","
            << 1.0 << ","
            << 0.0 << ","
            << P.s11 << ","
            << P.s22 << ","
            << P.s33 << ","
            << P.sigma11 << ","
            << P.sigmaI << ","
            << P.plastic_strain << ","
            << P.wave_speed << "\n";
    }
}

inline void write_csv(const std::string& filename, const std::vector<State>& U, const Config& cfg,
                      double free_surface_position)
{
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Cannot write Barton solid CSV: " + filename);
    }
    const double dx = (cfg.domain_max - cfg.domain_min) / cfg.cells;
    out << "x,rho,u,p,e,hydro_e,elastic_e,gxx,gyy,gzz,Fp_xx,Fp_yy,Fp_zz,"
        << "plastic_xx,sxx,syy,szz,normal_stress,equivalent_stress,plastic_strain,wave_speed\n";
    for (int i = 0; i < cfg.cells; ++i) {
        const double x = cfg.domain_min + (i + 0.5) * dx;
        Primitive P = cons_to_prim(U[i], cfg.material);
        if (cfg.moving_free_surface && x < free_surface_position) {
            P = cons_to_prim(prim_to_cons(cfg.material.rho0, 0.0, cfg.material), cfg.material);
        }
        out << x << ","
            << P.rho << ","
            << P.u << ","
            << P.pressure << ","
            << P.internal_e << ","
            << cold_energy(P.rho, cfg.material) << ","
            << P.shear_e << ","
            << P.F11 << ","
            << P.F22 << ","
            << P.F33 << ","
            << 1.0 << ","
            << 1.0 << ","
            << 1.0 << ","
            << 0.0 << ","
            << P.s11 << ","
            << P.s22 << ","
            << P.s33 << ","
            << P.sigma11 << ","
            << P.sigmaI << ","
            << P.plastic_strain << ","
            << P.wave_speed << "\n";
    }
}

} // namespace solid::barton

// Tensor CSV output.

// CSV writers for the active tensor Barton tensor multidimensional validation outputs.

namespace solid::barton {

inline void write_tensor_cartesian_2d(
    const std::string& filename,
    const std::vector<TensorState2D>& U,
    const TensorSolverConfig& cfg,
    const TensorMaterial& mat)
{
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot write Barton tensor radial pressure CSV: " + filename);
    const int nx = cfg.cells[0];
    const int ny = cfg.cells[1];
    const double dx = (cfg.domain_max[0] - cfg.domain_min[0]) / nx;
    const double dy = (cfg.domain_max[1] - cfg.domain_min[1]) / ny;
    out << "x,y,r,rho,ur,srr,T,p,ux,uy\n";
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double x = cfg.domain_min[0] + (i + 0.5) * dx;
            const double y = cfg.domain_min[1] + (j + 0.5) * dy;
            const double r = std::sqrt(x * x + y * y);
            const TensorPrim2D P = tensor_prim(U[hidx(i, j, nx)], mat);
            const double ur = r > 1.0e-14 ? (P.vel[0] * x + P.vel[1] * y) / r : 0.0;
            const double nxr = r > 1.0e-14 ? x / r : 1.0;
            const double nyr = r > 1.0e-14 ? y / r : 0.0;
            const double srr = nxr * nxr * P.sigma[0] +
                2.0 * nxr * nyr * P.sigma[1] +
                nyr * nyr * P.sigma[4];
            out << x * 100.0 << "," << y * 100.0 << "," << r * 100.0 << ","
                << P.rho / 1000.0 << "," << ur / 1000.0 << ","
                << srr / 1.0e9 << "," << P.T << "," << P.p << ","
                << P.vel[0] / 1000.0 << "," << P.vel[1] / 1000.0 << "\n";
        }
    }
}

inline void write_tensor_cylindrical_reference(
    const std::string& filename,
    const std::vector<TensorState2D>& U,
    const TensorSolverConfig& cfg,
    const TensorMaterial& mat)
{
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot write Barton tensor radial pressure reference CSV: " + filename);
    const double dr = (cfg.domain_max[0] - cfg.domain_min[0]) / cfg.radial_cells;
    out << "r,rho,ur,srr,T,p,stt\n";
    for (int i = 0; i < cfg.radial_cells; ++i) {
        const double r = cfg.domain_min[0] + (i + 0.5) * dr;
        const TensorPrim2D P = tensor_prim(U[i], mat);
        out << r * 100.0 << "," << P.rho / 1000.0 << "," << P.vel[0] / 1000.0 << ","
            << P.sigma[0] / 1.0e9 << "," << P.T << "," << P.p << ","
            << P.sigma[4] / 1.0e9 << "\n";
    }
}

} // namespace solid::barton
