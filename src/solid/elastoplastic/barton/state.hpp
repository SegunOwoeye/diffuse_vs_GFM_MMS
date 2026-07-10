#pragma once

// Barton elastoplastic state definitions and shared numerical helpers.

// Shared 1D state types and numerical helpers.

// Shared types and numerical safety helpers for the Barton 1D plate-impact model.
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

#ifdef _OPENMP
#include <omp.h>
#endif

#include "src/core/model_type.hpp"
#include "src/core/fv/limiters.hpp"
#include "src/solid/common/config_text.hpp"

namespace solid::barton {

struct Material {
    double rho0 = 2700.0;
    double shear_modulus = 24.8e9;
    double sigma0 = 0.2976e9;
    double tau0 = 1.0e-5;
    double relaxation_n = 100.0;
    double p01 = 72.0e9;
    double p02 = 172.0e9;
    double p03 = 40.0e9;
};

struct Region {
    double x_min = 0.0;
    double x_max = 0.0;
    double rho = 2700.0;
    double u = 0.0;
};

struct Config {
    double domain_min = 0.0;
    double domain_max = 1.0;
    int cells = 100;
    double tfinal = 0.0;
    double cfl = 0.6;
    bool moving_free_surface = false;
    double free_surface_position = 0.0;
    std::vector<double> output_times{};
    Material material{};
    BoundaryConditions bc{};
    std::vector<Region> regions{};
    std::string output_prefix = "barton_solid";
    std::string output_dir = "data/csv/solid";
};

struct State {
    double rho = 0.0;
    double mom = 0.0;
    double E = 0.0;
    double rhoF11 = 0.0;
    double rhoF22 = 0.0;
    double rhoF33 = 0.0;
    double rhoPlastic = 0.0;
};

struct StateSlope {
    double rho = 0.0;
    double mom = 0.0;
    double E = 0.0;
    double rhoF11 = 0.0;
    double rhoF22 = 0.0;
    double rhoF33 = 0.0;
    double rhoPlastic = 0.0;
};

struct Primitive {
    double rho = 0.0;
    double u = 0.0;
    double F11 = 1.0;
    double F22 = 1.0;
    double F33 = 1.0;
    double pressure = 0.0;
    double sigma11 = 0.0;
    double sigma22 = 0.0;
    double sigma33 = 0.0;
    double s11 = 0.0;
    double s22 = 0.0;
    double s33 = 0.0;
    double sigmaI = 0.0;
    double internal_e = 0.0;
    double shear_e = 0.0;
    double wave_speed = 1.0;
    double plastic_strain = 0.0;
};

inline double finite_or(double value, double fallback)
{
    return std::isfinite(value) ? value : fallback;
}

inline double clamp_min(double value, double floor)
{
    return std::max(finite_or(value, floor), floor);
}

inline double limited_component_slope(double a, double b)
{
    return core::fv::minmod(a, b);
}

} // namespace solid::barton

// Tensor state types, stress law, and primitive conversion.

// Tensor state, stress law, and primitive conversion for the active Barton tensor multidimensional solver.

namespace solid::barton {

inline int hidx(int i, int j, int nx) { return j * nx + i; }
inline int hidx3(int i, int j, int k, int nx, int ny) { return (k * ny + j) * nx + i; }

struct TensorMaterial {
    std::string eos = "eq51";
    double rho0 = 8900.0;
    double c0 = 4651.0;
    double b0 = 2141.0;
    double cv = 390.0;
    double T0 = 300.0;
    double alpha = 1.0;
    double beta = 3.0;
    double gamma = 2.0;
    double sigma0 = 0.045e9;
    double tau0 = 0.92;
    double relaxation_n = 10.1;
    bool damage_enabled = false;
    double jc_D1 = 0.54;
    double jc_D2 = 4.89;
    double jc_D3 = -3.03;
    double jc_D4 = 0.014;
    double jc_D5 = 1.12;
    double reference_plastic_strain_rate = 1.0;
    double melt_temperature = 1356.0;
    double failed_damage = 1.0;
    double residual_strength_fraction = 0.0;

    double K0() const { return c0 * c0 - (4.0 / 3.0) * b0 * b0; }
};

inline double material_cold_energy(double rho, const TensorMaterial& mat)
{
    const double r = clamp_min(rho, 1.0e-12) / mat.rho0;
    const double a = mat.alpha;
    const double term = std::pow(r, a) - 1.0;
    return mat.K0() / (2.0 * a * a) * term * term;
}

inline double material_pressure_from_rho_T(double rho, double T, const TensorMaterial& mat)
{
    rho = clamp_min(rho, 1.0e-12);
    const double r = rho / mat.rho0;
    const double a = mat.alpha;
    const double cold = mat.rho0 * mat.K0() / a *
        std::pow(r, a + 1.0) * (std::pow(r, a) - 1.0);
    const double thermal = rho * mat.gamma * mat.cv *
        (T - mat.T0 * std::pow(r, mat.gamma));
    return cold + thermal;
}

inline double material_temperature_from_rho_p(double rho, double p, const TensorMaterial& mat)
{
    rho = clamp_min(rho, 1.0e-12);
    const double r = rho / mat.rho0;
    const double a = mat.alpha;
    const double cold = mat.rho0 * mat.K0() / a *
        std::pow(r, a + 1.0) * (std::pow(r, a) - 1.0);
    return mat.T0 * std::pow(r, mat.gamma) +
        (p - cold) / std::max(rho * mat.gamma * mat.cv, 1.0e-30);
}

inline double material_temperature_from_rho_e(double rho, double e, const TensorMaterial& mat)
{
    const double r = clamp_min(rho, 1.0e-12) / mat.rho0;
    return mat.T0 * std::pow(r, mat.gamma) +
        (e - material_cold_energy(rho, mat)) / mat.cv;
}

inline double material_internal_energy_from_rho_T(double rho, double T, const TensorMaterial& mat)
{
    const double r = clamp_min(rho, 1.0e-12) / mat.rho0;
    return material_cold_energy(rho, mat) +
        mat.cv * (T - mat.T0 * std::pow(r, mat.gamma));
}

inline double material_bulk_modulus(double rho, double T, const TensorMaterial& mat)
{
    const double dr = 1.0e-4 * mat.rho0;
    const double p_lo = material_pressure_from_rho_T(std::max(rho - dr, 0.5 * mat.rho0), T, mat);
    const double p_hi = material_pressure_from_rho_T(rho + dr, T, mat);
    return std::max(rho * (p_hi - p_lo) / (2.0 * dr), 1.0e6);
}

inline double material_density_from_p_T(double p, double T, const TensorMaterial& mat)
{
    double lo = 0.5 * mat.rho0;
    double hi = 1.5 * mat.rho0;
    for (int iter = 0; iter < 100; ++iter) {
        const double mid = 0.5 * (lo + hi);
        if (material_pressure_from_rho_T(mid, T, mat) < p) {
            lo = mid;
        }
        else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

template<int DIM>
struct TensorState {
    double rho = 0.0;
    std::array<double, DIM> mom{};
    double E = 0.0;
    std::array<double, 9> rhoF{};
    double rhoEqps = 0.0;
    double rhoDamage = 0.0;
};

template<int DIM>
struct TensorPrim {
    double rho = 0.0;
    std::array<double, DIM> vel{};
    double e = 0.0;
    double T = 300.0;
    std::array<double, 9> F{};
    std::array<double, 9> sigma{};
    double p = 0.0;
    double wave_speed = 1.0;
    double eqps = 0.0;
    double damage = 0.0;
    bool failed = false;
};

template<int DIM>
struct TensorSlope {
    double rho = 0.0;
    std::array<double, DIM> mom{};
    double E = 0.0;
    std::array<double, 9> rhoF{};
    double rhoEqps = 0.0;
    double rhoDamage = 0.0;
};

using TensorState2D = TensorState<2>;
using TensorPrim2D = TensorPrim<2>;
using TensorSlope2D = TensorSlope<2>;
using TensorState3D = TensorState<3>;
using TensorPrim3D = TensorPrim<3>;
using TensorSlope3D = TensorSlope<3>;

inline double det3(const std::array<double, 9>& A)
{
    return A[0] * (A[4] * A[8] - A[5] * A[7])
        - A[1] * (A[3] * A[8] - A[5] * A[6])
        + A[2] * (A[3] * A[7] - A[4] * A[6]);
}

inline std::array<double, 9> inv3(const std::array<double, 9>& A)
{
    const double d = det3(A);
    const double inv_d = 1.0 / (std::abs(d) > 1.0e-14 ? d : (d < 0.0 ? -1.0e-14 : 1.0e-14));
    return {
        (A[4] * A[8] - A[5] * A[7]) * inv_d,
        (A[2] * A[7] - A[1] * A[8]) * inv_d,
        (A[1] * A[5] - A[2] * A[4]) * inv_d,
        (A[5] * A[6] - A[3] * A[8]) * inv_d,
        (A[0] * A[8] - A[2] * A[6]) * inv_d,
        (A[2] * A[3] - A[0] * A[5]) * inv_d,
        (A[3] * A[7] - A[4] * A[6]) * inv_d,
        (A[1] * A[6] - A[0] * A[7]) * inv_d,
        (A[0] * A[4] - A[1] * A[3]) * inv_d
    };
}

inline std::array<double, 9> matmul3(const std::array<double, 9>& A, const std::array<double, 9>& B)
{
    std::array<double, 9> C{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                C[3 * i + j] += A[3 * i + k] * B[3 * k + j];
            }
        }
    }
    return C;
}

inline std::array<double, 9> transpose3(const std::array<double, 9>& A)
{
    return {A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]};
}

inline double tensor_energy_from_F_entropy(
    const std::array<double, 9>& F,
    double entropy_factor,
    const TensorMaterial& mat)
{
    const auto Finv = inv3(F);
    const auto G = matmul3(transpose3(Finv), Finv);
    const double I1 = G[0] + G[4] + G[8];
    const auto G2 = matmul3(G, G);
    const double trG2 = G2[0] + G2[4] + G2[8];
    const double I2 = 0.5 * (I1 * I1 - trG2);
    const double I3 = std::max(det3(G), 1.0e-18);
    const double cold_term = mat.K0() / (2.0 * mat.alpha * mat.alpha) *
        std::pow(std::pow(I3, 0.5 * mat.alpha) - 1.0, 2.0);
    const double thermal_term = mat.cv * mat.T0 * std::pow(I3, 0.5 * mat.gamma) *
        (entropy_factor - 1.0);
    const double shear_term = 0.5 * mat.b0 * mat.b0 * std::pow(I3, 0.5 * mat.beta) *
        (I1 * I1 / 3.0 - I2);
    return cold_term + thermal_term + shear_term;
}

inline double tensor_energy_from_F_T(
    const std::array<double, 9>& F,
    double T,
    const TensorMaterial& mat)
{
    const auto Finv = inv3(F);
    const auto G = matmul3(transpose3(Finv), Finv);
    const double I3 = std::max(det3(G), 1.0e-18);
    const double entropy_factor = T / (mat.T0 * std::pow(I3, 0.5 * mat.gamma));
    return tensor_energy_from_F_entropy(F, entropy_factor, mat);
}

inline std::array<double, 9> tensor_stress_from_F_T(
    const std::array<double, 9>& F,
    double rho,
    double T,
    const TensorMaterial& mat)
{
    const auto Finv = inv3(F);
    const auto G = matmul3(transpose3(Finv), Finv);
    const auto Ginv = inv3(G);
    const auto G2 = matmul3(G, G);
    const double I1 = G[0] + G[4] + G[8];
    const double trG2 = G2[0] + G2[4] + G2[8];
    const double I2 = 0.5 * (I1 * I1 - trG2);
    const double I3 = std::max(det3(G), 1.0e-18);
    const double entropy_factor = T / (mat.T0 * std::pow(I3, 0.5 * mat.gamma));
    const double Q = I1 * I1 / 3.0 - I2;
    const double I3a = std::pow(I3, 0.5 * mat.alpha);
    const double I3b = std::pow(I3, 0.5 * mat.beta);
    const double E1 = 0.5 * mat.b0 * mat.b0 * I3b * (2.0 * I1 / 3.0);
    const double E2 = -0.5 * mat.b0 * mat.b0 * I3b;
    const double E3 =
        mat.K0() / (2.0 * mat.alpha) * (I3a - 1.0) * std::pow(I3, 0.5 * mat.alpha - 1.0) +
        mat.cv * mat.T0 * (entropy_factor - 1.0) * (0.5 * mat.gamma) *
            std::pow(I3, 0.5 * mat.gamma - 1.0) +
        0.5 * mat.b0 * mat.b0 * (0.5 * mat.beta) *
            std::pow(I3, 0.5 * mat.beta - 1.0) * Q;

    std::array<double, 9> dEdG{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double identity = (i == j) ? 1.0 : 0.0;
            dEdG[3 * i + j] =
                E1 * identity +
                E2 * (I1 * identity - G[3 * i + j]) +
                E3 * I3 * Ginv[3 * j + i];
        }
    }

    const auto GE = matmul3(G, dEdG);
    std::array<double, 9> sigma{};
    for (int q = 0; q < 9; ++q) {
        sigma[q] = -2.0 * rho * GE[q];
    }
    return sigma;
}

template<int DIM>
inline TensorPrim<DIM> tensor_prim(const TensorState<DIM>& U, const TensorMaterial& mat)
{
    TensorPrim<DIM> P{};
    P.rho = clamp_min(U.rho, 1.0e-12);
    for (int d = 0; d < DIM; ++d) {
        P.vel[d] = U.mom[d] / P.rho;
    }
    for (int q = 0; q < 9; ++q) {
        P.F[q] = U.rhoF[q] / P.rho;
    }
    P.F[0] = clamp_min(P.F[0], 1.0e-10);
    P.F[4] = clamp_min(P.F[4], 1.0e-10);
    P.F[8] = clamp_min(P.F[8], 1.0e-10);
    double kinetic = 0.0;
    for (int d = 0; d < DIM; ++d) {
        kinetic += P.vel[d] * P.vel[d];
    }
    P.e = U.E / P.rho - 0.5 * kinetic;

    const auto Finv = inv3(P.F);
    const auto G = matmul3(transpose3(Finv), Finv);
    const double I3 = std::max(det3(G), 1.0e-18);
    const double entropy_reference = mat.T0 * std::pow(I3, 0.5 * mat.gamma);
    const double no_thermal = tensor_energy_from_F_entropy(P.F, 1.0, mat);
    P.T = entropy_reference + (P.e - no_thermal) / mat.cv;
    P.T = std::clamp(finite_or(P.T, mat.T0), 50.0, 5000.0);
    P.sigma = tensor_stress_from_F_T(P.F, P.rho, P.T, mat);
    P.eqps = std::max(finite_or(U.rhoEqps / P.rho, 0.0), 0.0);
    P.damage = std::clamp(finite_or(U.rhoDamage / P.rho, 0.0), 0.0, mat.failed_damage);
    P.failed = mat.damage_enabled && P.damage >= mat.failed_damage;
    if (mat.damage_enabled && P.damage > 0.0) {
        const double mean = (P.sigma[0] + P.sigma[4] + P.sigma[8]) / 3.0;
        const double d = std::clamp(P.damage / std::max(mat.failed_damage, 1.0e-12), 0.0, 1.0);
        const double strength = std::max(mat.residual_strength_fraction, 1.0 - d);
        P.sigma[0] = mean + strength * (P.sigma[0] - mean);
        P.sigma[4] = mean + strength * (P.sigma[4] - mean);
        P.sigma[8] = mean + strength * (P.sigma[8] - mean);
        P.sigma[1] *= strength;
        P.sigma[2] *= strength;
        P.sigma[3] *= strength;
        P.sigma[5] *= strength;
        P.sigma[6] *= strength;
        P.sigma[7] *= strength;
    }
    P.p = -(P.sigma[0] + P.sigma[4] + P.sigma[8]) / 3.0;
    const double K = material_bulk_modulus(P.rho, P.T, mat);
    P.wave_speed = std::sqrt(std::max((K + 4.0 * mat.b0 * mat.b0 * P.rho / 3.0) / P.rho, 1.0));
    return P;
}

inline TensorState2D tensor_cons_from_F(
    double rho,
    double ux,
    double uy,
    double T,
    const std::array<double, 9>& F,
    const TensorMaterial& mat)
{
    TensorState2D U{};
    U.rho = rho;
    U.mom[0] = rho * ux;
    U.mom[1] = rho * uy;
    for (int q = 0; q < 9; ++q) {
        U.rhoF[q] = rho * F[q];
    }
    const double e = tensor_energy_from_F_T(F, T, mat);
    U.E = rho * (e + 0.5 * (ux * ux + uy * uy));
    U.rhoEqps = 0.0;
    U.rhoDamage = 0.0;
    return U;
}

inline TensorState3D tensor_cons_from_F(
    double rho,
    double ux,
    double uy,
    double uz,
    double T,
    const std::array<double, 9>& F,
    const TensorMaterial& mat)
{
    TensorState3D U{};
    U.rho = rho;
    U.mom[0] = rho * ux;
    U.mom[1] = rho * uy;
    U.mom[2] = rho * uz;
    for (int q = 0; q < 9; ++q) {
        U.rhoF[q] = rho * F[q];
    }
    const double e = tensor_energy_from_F_T(F, T, mat);
    U.E = rho * (e + 0.5 * (ux * ux + uy * uy + uz * uz));
    U.rhoEqps = 0.0;
    U.rhoDamage = 0.0;
    return U;
}

template<int DIM>
inline TensorState<DIM> operator+(const TensorState<DIM>& a, const TensorState<DIM>& b)
{
    TensorState<DIM> c{};
    c.rho = a.rho + b.rho;
    for (int d = 0; d < DIM; ++d) c.mom[d] = a.mom[d] + b.mom[d];
    c.E = a.E + b.E;
    for (int q = 0; q < 9; ++q) c.rhoF[q] = a.rhoF[q] + b.rhoF[q];
    c.rhoEqps = a.rhoEqps + b.rhoEqps;
    c.rhoDamage = a.rhoDamage + b.rhoDamage;
    return c;
}

template<int DIM>
inline TensorState<DIM> operator-(const TensorState<DIM>& a, const TensorState<DIM>& b)
{
    TensorState<DIM> c{};
    c.rho = a.rho - b.rho;
    for (int d = 0; d < DIM; ++d) c.mom[d] = a.mom[d] - b.mom[d];
    c.E = a.E - b.E;
    for (int q = 0; q < 9; ++q) c.rhoF[q] = a.rhoF[q] - b.rhoF[q];
    c.rhoEqps = a.rhoEqps - b.rhoEqps;
    c.rhoDamage = a.rhoDamage - b.rhoDamage;
    return c;
}

template<int DIM>
inline TensorState<DIM> operator*(double a, const TensorState<DIM>& u)
{
    TensorState<DIM> c{};
    c.rho = a * u.rho;
    for (int d = 0; d < DIM; ++d) c.mom[d] = a * u.mom[d];
    c.E = a * u.E;
    for (int q = 0; q < 9; ++q) c.rhoF[q] = a * u.rhoF[q];
    c.rhoEqps = a * u.rhoEqps;
    c.rhoDamage = a * u.rhoDamage;
    return c;
}

inline TensorState2D operator+(const TensorState2D& a, const TensorState2D& b)
{
    TensorState2D c{};
    c.rho = a.rho + b.rho;
    c.mom[0] = a.mom[0] + b.mom[0];
    c.mom[1] = a.mom[1] + b.mom[1];
    c.E = a.E + b.E;
    for (int q = 0; q < 9; ++q) c.rhoF[q] = a.rhoF[q] + b.rhoF[q];
    c.rhoEqps = a.rhoEqps + b.rhoEqps;
    c.rhoDamage = a.rhoDamage + b.rhoDamage;
    return c;
}

inline TensorState2D operator-(const TensorState2D& a, const TensorState2D& b)
{
    TensorState2D c{};
    c.rho = a.rho - b.rho;
    c.mom[0] = a.mom[0] - b.mom[0];
    c.mom[1] = a.mom[1] - b.mom[1];
    c.E = a.E - b.E;
    for (int q = 0; q < 9; ++q) c.rhoF[q] = a.rhoF[q] - b.rhoF[q];
    c.rhoEqps = a.rhoEqps - b.rhoEqps;
    c.rhoDamage = a.rhoDamage - b.rhoDamage;
    return c;
}

inline TensorState2D operator*(double a, const TensorState2D& u)
{
    TensorState2D c{};
    c.rho = a * u.rho;
    c.mom[0] = a * u.mom[0];
    c.mom[1] = a * u.mom[1];
    c.E = a * u.E;
    for (int q = 0; q < 9; ++q) c.rhoF[q] = a * u.rhoF[q];
    c.rhoEqps = a * u.rhoEqps;
    c.rhoDamage = a * u.rhoDamage;
    return c;
}

template<int DIM>
inline TensorSlope<DIM> limited_tensor_slope_dim(
    const TensorState<DIM>& left,
    const TensorState<DIM>& centre,
    const TensorState<DIM>& right)
{
    TensorSlope<DIM> s{};
    s.rho = limited_component_slope(centre.rho - left.rho, right.rho - centre.rho);
    for (int d = 0; d < DIM; ++d) {
        s.mom[d] = limited_component_slope(centre.mom[d] - left.mom[d], right.mom[d] - centre.mom[d]);
    }
    s.E = limited_component_slope(centre.E - left.E, right.E - centre.E);
    for (int q = 0; q < 9; ++q) {
        s.rhoF[q] = limited_component_slope(centre.rhoF[q] - left.rhoF[q], right.rhoF[q] - centre.rhoF[q]);
    }
    s.rhoEqps = limited_component_slope(centre.rhoEqps - left.rhoEqps, right.rhoEqps - centre.rhoEqps);
    s.rhoDamage = limited_component_slope(centre.rhoDamage - left.rhoDamage, right.rhoDamage - centre.rhoDamage);
    return s;
}

inline TensorSlope2D limited_tensor_slope(
    const TensorState2D& left,
    const TensorState2D& centre,
    const TensorState2D& right)
{
    TensorSlope2D s{};
    s.rho = limited_component_slope(centre.rho - left.rho, right.rho - centre.rho);
    s.mom[0] = limited_component_slope(centre.mom[0] - left.mom[0], right.mom[0] - centre.mom[0]);
    s.mom[1] = limited_component_slope(centre.mom[1] - left.mom[1], right.mom[1] - centre.mom[1]);
    s.E = limited_component_slope(centre.E - left.E, right.E - centre.E);
    for (int q = 0; q < 9; ++q) {
        s.rhoF[q] = limited_component_slope(centre.rhoF[q] - left.rhoF[q], right.rhoF[q] - centre.rhoF[q]);
    }
    s.rhoEqps = limited_component_slope(centre.rhoEqps - left.rhoEqps, right.rhoEqps - centre.rhoEqps);
    s.rhoDamage = limited_component_slope(centre.rhoDamage - left.rhoDamage, right.rhoDamage - centre.rhoDamage);
    return s;
}

inline TensorSlope3D limited_tensor_slope(
    const TensorState3D& left,
    const TensorState3D& centre,
    const TensorState3D& right)
{
    return limited_tensor_slope_dim(left, centre, right);
}

template<int DIM>
inline TensorState<DIM> add_tensor_slope_dim(const TensorState<DIM>& u, const TensorSlope<DIM>& s, double scale)
{
    TensorState<DIM> out = u;
    out.rho += scale * s.rho;
    for (int d = 0; d < DIM; ++d) out.mom[d] += scale * s.mom[d];
    out.E += scale * s.E;
    for (int q = 0; q < 9; ++q) {
        out.rhoF[q] += scale * s.rhoF[q];
    }
    out.rhoEqps += scale * s.rhoEqps;
    out.rhoDamage += scale * s.rhoDamage;
    return out;
}

inline TensorState2D add_tensor_slope(const TensorState2D& u, const TensorSlope2D& s, double scale)
{
    TensorState2D out = u;
    out.rho += scale * s.rho;
    out.mom[0] += scale * s.mom[0];
    out.mom[1] += scale * s.mom[1];
    out.E += scale * s.E;
    for (int q = 0; q < 9; ++q) {
        out.rhoF[q] += scale * s.rhoF[q];
    }
    out.rhoEqps += scale * s.rhoEqps;
    out.rhoDamage += scale * s.rhoDamage;
    return out;
}

inline TensorState3D add_tensor_slope(const TensorState3D& u, const TensorSlope3D& s, double scale)
{
    return add_tensor_slope_dim(u, s, scale);
}

template<int DIM>
inline TensorSlope<DIM> scale_tensor_slope_dim(const TensorSlope<DIM>& s, double scale)
{
    TensorSlope<DIM> out{};
    out.rho = scale * s.rho;
    for (int d = 0; d < DIM; ++d) out.mom[d] = scale * s.mom[d];
    out.E = scale * s.E;
    for (int q = 0; q < 9; ++q) {
        out.rhoF[q] = scale * s.rhoF[q];
    }
    out.rhoEqps = scale * s.rhoEqps;
    out.rhoDamage = scale * s.rhoDamage;
    return out;
}

inline TensorSlope2D scale_tensor_slope(const TensorSlope2D& s, double scale)
{
    TensorSlope2D out{};
    out.rho = scale * s.rho;
    out.mom[0] = scale * s.mom[0];
    out.mom[1] = scale * s.mom[1];
    out.E = scale * s.E;
    for (int q = 0; q < 9; ++q) {
        out.rhoF[q] = scale * s.rhoF[q];
    }
    out.rhoEqps = scale * s.rhoEqps;
    out.rhoDamage = scale * s.rhoDamage;
    return out;
}

inline TensorSlope3D scale_tensor_slope(const TensorSlope3D& s, double scale)
{
    return scale_tensor_slope_dim(s, scale);
}

} // namespace solid::barton
