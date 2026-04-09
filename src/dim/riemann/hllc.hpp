#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "src/dim/state.hpp"
#include "src/dim/conservative.hpp"
#include "src/dim/eos.hpp"
#include "src/dim/eos_params.hpp"
#include "src/dim/flux.hpp"

#include "src/math/vector_ops.hpp"
#include "src/math/numerical_safety.hpp"


// [0] Compute total mixture density
template<int DIM, int NMAT>
inline double compute_total_density(
    const Conserved<DIM, NMAT>& U
)
{
    double rho = 0.0;

    for (int k = 0; k < NMAT; ++k) {
        rho += U.arho[k];
    }

    return rho;
}


// [1] Compute unit normal
template<int DIM>
inline std::array<double, DIM> compute_unit_normal(
    const std::array<double, DIM>& normal,
    double tol = 1e-10
)
{
    const double mag = norm<DIM>(normal);

    std::array<double, DIM> n{};

    if (mag < tol) {
        n[0] = 1.0;
    } else {
        n = scale<DIM>(1.0 / mag, normal);
    }

    return n;
}


// [2] Compute mixture sound speed from conserved state
template<int DIM, int NMAT, typename EOS>
inline double compute_mixture_sound_speed(
    const Conserved<DIM, NMAT>& U,
    const EOSParams<NMAT>& params,
    double rho_floor = 1e-10,
    double p_floor = 1e-10,
    double alpha_floor = 1e-12
)
{
    const Primitive<DIM, NMAT> P = cons_to_prim<DIM, NMAT, EOS>(
        U,
        params,
        rho_floor,
        p_floor,
        alpha_floor
    );

    return EOS::template mixture_sound_speed<NMAT>(
        P.alpha,
        P.rho,
        P.p,
        params
    );
}


// [3] Compute HLLC star total density
inline double compute_star_density(
    double rho,
    double sK,
    double unK,
    double s_star,
    double tol = 1e-10
)
{
    return rho * safe_div(sK - unK, sK - s_star, tol);
}


// [4] Compute HLLC star total energy
inline double compute_star_energy(
    double rho_star,
    double E,
    double rho,
    double p,
    double sK,
    double unK,
    double s_star,
    double tol = 1e-10
)
{
    return rho_star * (
        safe_div(E, rho, tol) +
        (s_star - unK) * (
            s_star + safe_div(p, rho * (sK - unK), tol)
        )
    );
}


// [5] HLLC Flux using arbitrary interface normal for the reduced 5-equation model
template<int DIM, int NMAT, typename EOS>
inline Conserved<DIM, NMAT> hllc_flux_normal(
    const Conserved<DIM, NMAT>& UL,
    const Conserved<DIM, NMAT>& UR,
    const std::array<double, DIM>& normal,
    const EOSParams<NMAT>& params,
    double tol = 1e-10
)
{
    // [5.1] Unit normal
    const std::array<double, DIM> n = compute_unit_normal<DIM>(
        normal,
        tol
    );

    // [5.2] Primitive states
    const Primitive<DIM, NMAT> PL = cons_to_prim<DIM, NMAT, EOS>(
        UL,
        params
    );

    const Primitive<DIM, NMAT> PR = cons_to_prim<DIM, NMAT, EOS>(
        UR,
        params
    );

    // [5.3] Mixture densities
    const double rhoL = require_positive(
        compute_total_density<DIM, NMAT>(UL),
        "dim hllc: invalid left density"
    );

    const double rhoR = require_positive(
        compute_total_density<DIM, NMAT>(UR),
        "dim hllc: invalid right density"
    );

    // [5.4] Normal velocities
    const double unL = dot<DIM>(PL.vel, n);
    const double unR = dot<DIM>(PR.vel, n);

    // [5.5] Pressures
    const double pL = require_positive(
        PL.p,
        "dim hllc: invalid left pressure"
    );

    const double pR = require_positive(
        PR.p,
        "dim hllc: invalid right pressure"
    );

    // [5.6] Mixture sound speeds
    const double cL = compute_mixture_sound_speed<DIM, NMAT, EOS>(
        UL,
        params
    );

    const double cR = compute_mixture_sound_speed<DIM, NMAT, EOS>(
        UR,
        params
    );

    // [5.7] Wave speed estimates
    const double sL = std::min(unL - cL, unR - cR);
    const double sR = std::max(unL + cL, unR + cR);

    // [5.8] Contact wave speed
    const double numerator =
        (pR - pL)
        + rhoL * unL * (sL - unL)
        - rhoR * unR * (sR - unR);

    const double denom =
        rhoL * (sL - unL)
        - rhoR * (sR - unR);

    double s_star = safe_div(numerator, denom, tol);
    
    // [5.8.1] Contact consistency correction
    if (std::abs(pL - pR) < tol && std::abs(unL - unR) < tol) {
        s_star = unL;
    }

    // [5.9] Physical fluxes
    const Conserved<DIM, NMAT> FL = compute_flux_normal<DIM, NMAT, EOS>(
        UL,
        n,
        params
    );

    const Conserved<DIM, NMAT> FR = compute_flux_normal<DIM, NMAT, EOS>(
        UR,
        n,
        params
    );

    // [5.10] Left supersonic region
    if (0.0 <= sL) {
        Conserved<DIM, NMAT> F = FL;

        for (int k = 0; k < NMAT; ++k) {
            F.alpha[k] = PL.alpha[k] * s_star;
        }

        return F;
    }

    // [5.11] Right supersonic region
    if (sR <= 0.0) {
        Conserved<DIM, NMAT> F = FR;

        for (int k = 0; k < NMAT; ++k) {
            F.alpha[k] = PR.alpha[k] * s_star;
        }

        return F;
    }

    // [5.12] Left star region
    if (sL <= 0.0 && 0.0 <= s_star) {

        Conserved<DIM, NMAT> UL_star{};

        const double rhoL_star = compute_star_density(
            rhoL,
            sL,
            unL,
            s_star,
            tol
        );

        const double EL_star = compute_star_energy(
            rhoL_star,
            UL.E,
            rhoL,
            pL,
            sL,
            unL,
            s_star,
            tol
        );

        if (!std::isfinite(rhoL_star) ||
            !std::isfinite(EL_star) ||
            rhoL_star <= 0.0 ||
            EL_star <= 0.0) {

            throw std::runtime_error(
                "dim hllc left star invalid: "
                "rhoL=" + std::to_string(rhoL) +
                " unL=" + std::to_string(unL) +
                " pL=" + std::to_string(pL) +
                " cL=" + std::to_string(cL) +
                " sL=" + std::to_string(sL) +
                " sR=" + std::to_string(sR) +
                " s_star=" + std::to_string(s_star) +
                " rhoL_star=" + std::to_string(rhoL_star) +
                " EL=" + std::to_string(UL.E) +
                " EL_star=" + std::to_string(EL_star)
            );
        }

        // [5.12.1] Partial masses
        const double mass_scale = safe_div(rhoL_star, rhoL, tol);

        for (int k = 0; k < NMAT; ++k) {
            UL_star.arho[k] = UL.arho[k] * mass_scale;
        }

        // [5.12.2] Momentum
        std::array<double, DIM> u_star = PL.vel;
        const double delta = s_star - unL;

        for (int d = 0; d < DIM; ++d) {
            u_star[d] += delta * n[d];
            UL_star.mom[d] = rhoL_star * u_star[d];
        }

        // [5.12.3] Energy
        UL_star.E = EL_star;

        Conserved<DIM, NMAT> F = FL + sL * (UL_star - UL);

        // alpha flux advection
        for (int k = 0; k < NMAT; ++k) {
            F.alpha[k] = PL.alpha[k] * s_star;
        }

        return F;


    }

    // [5.13] Right star region
    Conserved<DIM, NMAT> UR_star{};

    const double rhoR_star = compute_star_density(
        rhoR,
        sR,
        unR,
        s_star,
        tol
    );

    const double ER_star = compute_star_energy(
        rhoR_star,
        UR.E,
        rhoR,
        pR,
        sR,
        unR,
        s_star,
        tol
    );

    if (!std::isfinite(rhoR_star) ||
        !std::isfinite(ER_star) ||
        rhoR_star <= 0.0 ||
        ER_star <= 0.0) {

        throw std::runtime_error(
            "dim hllc right star invalid: "
            "rhoR=" + std::to_string(rhoR) +
            " unR=" + std::to_string(unR) +
            " pR=" + std::to_string(pR) +
            " cR=" + std::to_string(cR) +
            " sL=" + std::to_string(sL) +
            " sR=" + std::to_string(sR) +
            " s_star=" + std::to_string(s_star) +
            " rhoR_star=" + std::to_string(rhoR_star) +
            " ER=" + std::to_string(UR.E) +
            " ER_star=" + std::to_string(ER_star)
        );
    }

    // [5.13.1] Partial masses
    const double mass_scale = safe_div(rhoR_star, rhoR, tol);

    for (int k = 0; k < NMAT; ++k) {
        UR_star.arho[k] = UR.arho[k] * mass_scale;
    }

    // [5.13.2] Momentum
    std::array<double, DIM> u_star = PR.vel;
    const double delta = s_star - unR;

    for (int d = 0; d < DIM; ++d) {
        u_star[d] += delta * n[d];
        UR_star.mom[d] = rhoR_star * u_star[d];
    }

    // [5.13.3] Energy
    UR_star.E = ER_star;

    Conserved<DIM, NMAT> F = FR + sR * (UR_star - UR);

    for (int k = 0; k < NMAT; ++k) {
        F.alpha[k] = PR.alpha[k] * s_star;
    }

    return F;
}

