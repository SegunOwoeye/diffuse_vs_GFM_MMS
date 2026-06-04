#pragma once

// Barton material models, EOS helpers, and conservative/primitive conversion.

#include "src/solid/elastoplastic/barton/state.hpp"

// 1D EOS and conservative/primitive conversion.

// Equation of state, elastic energy, and conservative/primitive conversion for the 1D Barton model.

namespace solid::barton {

inline double cold_pressure(double rho, const Material& mat)
{
    const double eta = clamp_min(rho, 1.0e-12) / mat.rho0 - 1.0;
    const double p = mat.p01 * eta + mat.p02 * eta * eta + mat.p03 * eta * eta * eta;
    return std::max(p, -0.95 * mat.p01);
}

inline double cold_energy(double rho, const Material& mat)
{
    const double y = clamp_min(rho, 1.0e-12) / mat.rho0;
    const double A = mat.p01;
    const double B = mat.p02;
    const double C = mat.p03;
    const double K = -A + B - C;

    auto antiderivative = [&](double yy) {
        return 0.5 * C * yy * yy +
               (B - 3.0 * C) * yy +
               (A - 2.0 * B + 3.0 * C) * std::log(std::max(yy, 1.0e-12)) -
               K / std::max(yy, 1.0e-12);
    };

    return (antiderivative(y) - antiderivative(1.0)) / mat.rho0;
}

inline double shear_energy(double F11, double F22, double F33, const Material& mat)
{
    const double J = clamp_min(F11 * F22 * F33, 1.0e-18);
    const double I1 = F11 * F11 + F22 * F22 + F33 * F33;
    return 0.5 * mat.shear_modulus / mat.rho0 *
           (I1 - 3.0 * std::pow(J, 2.0 / 3.0));
}

inline double bulk_modulus(double rho, const Material& mat)
{
    const double eta = clamp_min(rho, 1.0e-12) / mat.rho0 - 1.0;
    const double dp_drho =
        (mat.p01 + 2.0 * mat.p02 * eta + 3.0 * mat.p03 * eta * eta) / mat.rho0;
    return std::max(rho * dp_drho, 1.0);
}

inline Primitive cons_to_prim(const State& U, const Material& mat)
{
    Primitive P{};
    P.rho = clamp_min(U.rho, 1.0e-12);
    P.u = U.mom / P.rho;
    P.F11 = finite_or(U.rhoF11 / P.rho, 1.0);
    P.F22 = finite_or(U.rhoF22 / P.rho, 1.0);
    P.F33 = finite_or(U.rhoF33 / P.rho, 1.0);
    P.F11 = clamp_min(P.F11, 1.0e-10);
    P.F22 = clamp_min(P.F22, 1.0e-10);
    P.F33 = clamp_min(P.F33, 1.0e-10);

    const double J = clamp_min(P.F11 * P.F22 * P.F33, 1.0e-18);
    const double J23 = std::pow(J, 2.0 / 3.0);
    P.pressure = cold_pressure(P.rho, mat);

    const double scale = mat.shear_modulus / J;
    const double tau11 = scale * (P.F11 * P.F11 - J23);
    const double tau22 = scale * (P.F22 * P.F22 - J23);
    const double tau33 = scale * (P.F33 * P.F33 - J23);
    const double mean_tau = (tau11 + tau22 + tau33) / 3.0;
    P.s11 = tau11 - mean_tau;
    P.s22 = tau22 - mean_tau;
    P.s33 = tau33 - mean_tau;

    P.sigma11 = -P.pressure + P.s11;
    P.sigma22 = -P.pressure + P.s22;
    P.sigma33 = -P.pressure + P.s33;
    P.sigmaI = std::sqrt(std::max(
        0.5 * ((P.sigma11 - P.sigma22) * (P.sigma11 - P.sigma22) +
               (P.sigma22 - P.sigma33) * (P.sigma22 - P.sigma33) +
               (P.sigma33 - P.sigma11) * (P.sigma33 - P.sigma11)),
        0.0));
    P.shear_e = shear_energy(P.F11, P.F22, P.F33, mat);
    P.internal_e = cold_energy(P.rho, mat) + P.shear_e;
    P.wave_speed = std::sqrt(
        std::max((bulk_modulus(P.rho, mat) + 4.0 * mat.shear_modulus / 3.0) / P.rho, 1.0));
    P.plastic_strain = U.rhoPlastic / P.rho;
    return P;
}

inline State prim_to_cons(double rho, double u, const Material& mat)
{
    State U{};
    U.rho = rho;
    U.mom = rho * u;
    U.rhoF11 = rho;
    U.rhoF22 = rho;
    U.rhoF33 = rho;
    const Primitive P = cons_to_prim(U, mat);
    U.E = rho * (P.internal_e + 0.5 * u * u);
    return U;
}

inline State operator+(const State& a, const State& b)
{
    return {
        a.rho + b.rho,
        a.mom + b.mom,
        a.E + b.E,
        a.rhoF11 + b.rhoF11,
        a.rhoF22 + b.rhoF22,
        a.rhoF33 + b.rhoF33,
        a.rhoPlastic + b.rhoPlastic
    };
}

inline State operator-(const State& a, const State& b)
{
    return {
        a.rho - b.rho,
        a.mom - b.mom,
        a.E - b.E,
        a.rhoF11 - b.rhoF11,
        a.rhoF22 - b.rhoF22,
        a.rhoF33 - b.rhoF33,
        a.rhoPlastic - b.rhoPlastic
    };
}

inline State operator*(double a, const State& u)
{
    return {
        a * u.rho,
        a * u.mom,
        a * u.E,
        a * u.rhoF11,
        a * u.rhoF22,
        a * u.rhoF33,
        a * u.rhoPlastic
    };
}

} // namespace solid::barton
