#pragma once

// Barton 1D and tensor fluxes, boundary states, and invariant enforcement.

#include "src/solid/elastoplastic/barton/eos.hpp"

// 1D fluxes, MUSCL-Hancock face prediction, and free-surface ghost states.

// Fluxes, MUSCL-Hancock face prediction, and free-surface ghost states for the 1D Barton model.

namespace solid::barton {

inline double limit_slope(double a, double b)
{
    if (a * b <= 0.0) {
        return 0.0;
    }
    return std::abs(a) < std::abs(b) ? a : b;
}

inline StateSlope limited_slope(const State& left, const State& centre, const State& right)
{
    StateSlope s{};
    s.rho = limit_slope(centre.rho - left.rho, right.rho - centre.rho);
    s.mom = limit_slope(centre.mom - left.mom, right.mom - centre.mom);
    s.E = limit_slope(centre.E - left.E, right.E - centre.E);
    s.rhoF11 = limit_slope(centre.rhoF11 - left.rhoF11, right.rhoF11 - centre.rhoF11);
    s.rhoF22 = limit_slope(centre.rhoF22 - left.rhoF22, right.rhoF22 - centre.rhoF22);
    s.rhoF33 = limit_slope(centre.rhoF33 - left.rhoF33, right.rhoF33 - centre.rhoF33);
    s.rhoPlastic = limit_slope(
        centre.rhoPlastic - left.rhoPlastic,
        right.rhoPlastic - centre.rhoPlastic);
    return s;
}

inline State add_slope(const State& u, const StateSlope& s, double scale)
{
    State out = u;
    out.rho += scale * s.rho;
    out.mom += scale * s.mom;
    out.E += scale * s.E;
    out.rhoF11 += scale * s.rhoF11;
    out.rhoF22 += scale * s.rhoF22;
    out.rhoF33 += scale * s.rhoF33;
    out.rhoPlastic += scale * s.rhoPlastic;
    return out;
}

inline State flux_x(const State& U, const Material& mat)
{
    const Primitive P = cons_to_prim(U, mat);
    State F{};
    F.rho = U.mom;
    F.mom = U.mom * P.u - P.sigma11;
    F.E = P.u * (U.E - P.sigma11);
    F.rhoF11 = 0.0;
    F.rhoF22 = P.u * U.rhoF22;
    F.rhoF33 = P.u * U.rhoF33;
    F.rhoPlastic = P.u * U.rhoPlastic;
    return F;
}

inline void enforce(State& U, const Material& mat);

inline State hancock_predict_cell(const State& centre, const State& left_face,
                                  const State& right_face, const Material& mat,
                                  double dt, double dx)
{
    const State FL = flux_x(left_face, mat);
    const State FR = flux_x(right_face, mat);
    State predicted = centre - (0.5 * dt / dx) * (FR - FL);
    enforce(predicted, mat);
    return predicted;
}

inline State rusanov_flux(const State& left, const State& right, const Material& mat)
{
    const Primitive L = cons_to_prim(left, mat);
    const Primitive R = cons_to_prim(right, mat);
    const State FL = flux_x(left, mat);
    const State FR = flux_x(right, mat);
    const double a = std::max(std::abs(L.u) + L.wave_speed, std::abs(R.u) + R.wave_speed);
    return 0.5 * (FL + FR) - 0.5 * a * (right - left);
}

inline State make_state_from_deformation(double rho, double u, double F11, double F22, double F33,
                                         double rhoPlastic, const Material& mat)
{
    State U{};
    U.rho = rho;
    U.mom = rho * u;
    U.rhoF11 = rho * clamp_min(F11, 1.0e-10);
    U.rhoF22 = rho * clamp_min(F22, 1.0e-10);
    U.rhoF33 = rho * clamp_min(F33, 1.0e-10);
    U.rhoPlastic = std::max(rhoPlastic, 0.0);
    const Primitive P = cons_to_prim(U, mat);
    U.E = rho * (P.internal_e + 0.5 * u * u);
    return U;
}

inline State traction_free_ghost(const State& U, const Material& mat)
{
    const Primitive P = cons_to_prim(U, mat);
    const double target = -P.sigma11;
    const double J = clamp_min(P.F11 * P.F22 * P.F33, 1.0e-18);
    const double J13 = std::pow(J, 1.0 / 3.0);

    auto candidate = [&](double q) {
        return make_state_from_deformation(
            P.rho,
            P.u,
            J13 * std::exp(q),
            J13 * std::exp(-0.5 * q),
            J13 * std::exp(-0.5 * q),
            U.rhoPlastic,
            mat);
    };
    auto residual = [&](double q) {
        return cons_to_prim(candidate(q), mat).sigma11 - target;
    };

    double lo = -3.0;
    double hi = 3.0;
    double rlo = residual(lo);
    double rhi = residual(hi);
    if (rlo * rhi > 0.0) {
        State G = U;
        G.mom = -G.mom;
        return G;
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

inline State ghost(const State& U, const BoundaryConditions& bc, bool left, const Material& mat)
{
    State G = U;
    const std::string& kind = left ? bc.left : bc.right;
    if (kind == "free_surface") {
        G = traction_free_ghost(U, mat);
    }
    else if (kind == "reflective") {
        G.mom = -G.mom;
    }
    return G;
}

inline void enforce(State& U, const Material& mat)
{
    U.rho = clamp_min(U.rho, 1.0e-12);
    if (!std::isfinite(U.mom)) {
        U.mom = 0.0;
    }
    U.rhoF11 = clamp_min(U.rhoF11, 1.0e-12);
    U.rhoF22 = clamp_min(U.rhoF22, 1.0e-12);
    U.rhoF33 = clamp_min(U.rhoF33, 1.0e-12);
    U.rhoPlastic = std::max(finite_or(U.rhoPlastic, 0.0), 0.0);
    const Primitive P = cons_to_prim(U, mat);
    const double kinetic = 0.5 * U.rho * P.u * P.u;
    const double internal = U.rho * P.internal_e;
    if (!std::isfinite(U.E) || U.E < kinetic + internal) {
        U.E = kinetic + internal;
    }
}

} // namespace solid::barton

// Tensor fluxes and boundary handling.

// Tensor finite-volume fluxes and invariant enforcement for the Barton tensor multidimensional solver.

namespace solid::barton {

inline TensorState2D tensor_flux_from_prim(const TensorState2D& U, const TensorPrim2D& P, int dir);

inline TensorState2D tensor_flux(const TensorState2D& U, const TensorMaterial& mat, int dir)
{
    const TensorPrim2D P = tensor_prim(U, mat);
    (void)mat;
    return tensor_flux_from_prim(U, P, dir);
}

inline TensorState2D tensor_flux_from_prim(const TensorState2D& U, const TensorPrim2D& P, int dir)
{
    const double uk = dir == 0 ? P.vel[0] : P.vel[1];
    TensorState2D F{};
    F.rho = U.rho * uk;
    F.mom[0] = U.mom[0] * uk - P.sigma[0 * 3 + dir];
    F.mom[1] = U.mom[1] * uk - P.sigma[1 * 3 + dir];
    F.E = uk * U.E - (P.vel[0] * P.sigma[0 * 3 + dir] + P.vel[1] * P.sigma[1 * 3 + dir]);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double ui = (i == 0) ? P.vel[0] : ((i == 1) ? P.vel[1] : 0.0);
            F.rhoF[3 * i + j] = U.rhoF[3 * i + j] * uk - U.rhoF[3 * dir + j] * ui;
        }
    }
    return F;
}

inline TensorState2D tensor_rusanov(const TensorState2D& L, const TensorState2D& R,
                                    const TensorMaterial& mat, int dir)
{
    const TensorPrim2D PL = tensor_prim(L, mat);
    const TensorPrim2D PR = tensor_prim(R, mat);
    const double ul = dir == 0 ? PL.vel[0] : PL.vel[1];
    const double ur = dir == 0 ? PR.vel[0] : PR.vel[1];
    const double a = std::max(std::abs(ul) + PL.wave_speed, std::abs(ur) + PR.wave_speed);
    return 0.5 * (tensor_flux(L, mat, dir) + tensor_flux(R, mat, dir)) - 0.5 * a * (R - L);
}

inline void enforce_tensor(TensorState2D& U, const TensorMaterial& mat)
{
    U.rho = std::clamp(finite_or(U.rho, mat.rho0), 0.2 * mat.rho0, 2.0 * mat.rho0);
    if (!std::isfinite(U.mom[0])) U.mom[0] = 0.0;
    if (!std::isfinite(U.mom[1])) U.mom[1] = 0.0;
    for (int q = 0; q < 9; ++q) {
        if (!std::isfinite(U.rhoF[q])) U.rhoF[q] = 0.0;
    }
    if (U.rhoF[0] <= 0.0) U.rhoF[0] = U.rho;
    if (U.rhoF[4] <= 0.0) U.rhoF[4] = U.rho;
    if (U.rhoF[8] <= 0.0) U.rhoF[8] = U.rho;
    std::array<double, 9> F{};
    for (int q = 0; q < 9; ++q) {
        F[q] = U.rhoF[q] / U.rho;
    }
    const double detF = det3(F);
    const double target_det = mat.rho0 / U.rho;
    if (std::isfinite(detF) && std::abs(detF) > 1.0e-14 && target_det > 0.0) {
        const double scale = std::cbrt(std::abs(target_det / detF));
        for (int q = 0; q < 9; ++q) {
            U.rhoF[q] *= scale;
        }
    }
    TensorPrim2D P = tensor_prim(U, mat);
    const double min_e = tensor_energy_from_F_T(P.F, 50.0, mat);
    const double kinetic = 0.5 * U.rho * (P.vel[0] * P.vel[0] + P.vel[1] * P.vel[1]);
    if (!std::isfinite(U.E) || U.E < U.rho * min_e + kinetic) {
        U.E = U.rho * min_e + kinetic;
    }
}

} // namespace solid::barton
