#pragma once

#include <algorithm>
#include <array>

#include "src/dim/flux.hpp"
#include "src/math/numerical_safety.hpp"
#include "src/math/vector_ops.hpp"

namespace dim {

    // [1] Computes the HLLC star-region density
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

    // [2] Computes the star-region total energy
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
            safe_div(E, rho, tol) + (s_star - unK) * (s_star + safe_div(p, rho * (sK - unK), tol))
        );
    }

    /*
    [3] Fallback HLL flux used when the HLLC star state is invalid 
        or when a more diffusive safe flux is needed
    */
    template<int DIM>
    inline Flux<DIM> hll_flux(
        const State<DIM>& UL,
        const State<DIM>& UR,
        const Flux<DIM>& FL,
        const Flux<DIM>& FR,
        double sL,
        double sR
    )
    {
        const Flux<DIM> delta = state_difference(UR, UL);
        return safe_div((sR * FL - sL * FR + (sL * sR) * delta), (sR - sL));
    }

    // [4] HLLC Riemann Solver
    template<int DIM>
    inline RiemannResult<DIM> hllc_flux_normal(
        const State<DIM>& UL,
        const State<DIM>& UR,
        const EOSParams& params,
        const std::array<double, DIM>& normal,
        double tol = 1e-10
    )
    {
        const int nmat = params.nmat();
        require_compatible_state(UL, nmat, "dim::hllc_flux_normal left");
        require_compatible_state(UR, nmat, "dim::hllc_flux_normal right");

        const double mag = norm<DIM>(normal);
        std::array<double, DIM> n{};
        if (mag < tol) {
            n[0] = 1.0;
        }
        else {
            n = scale<DIM>(1.0 / mag, normal);
        }

        const Primitive<DIM> PL = cons_to_prim(UL, params);
        const Primitive<DIM> PR = cons_to_prim(UR, params);

        const double rhoL = require_positive(total_density(UL), "dim::hllc_flux_normal: invalid left density");
        const double rhoR = require_positive(total_density(UR), "dim::hllc_flux_normal: invalid right density");
        const double pL = require_positive(PL.p, "dim::hllc_flux_normal: invalid left pressure");
        const double pR = require_positive(PR.p, "dim::hllc_flux_normal: invalid right pressure");

        const double unL = dot<DIM>(PL.vel, n);
        const double unR = dot<DIM>(PR.vel, n);

        const double cL = IdealGasEOS::mixture_sound_speed(rhoL, pL, PL.alpha, params);
        const double cR = IdealGasEOS::mixture_sound_speed(rhoR, pR, PR.alpha, params);

        const double sL = std::min(unL - cL, unR - cR);
        const double sR = std::max(unL + cL, unR + cR);

        const double numerator = (pR - pL) + rhoL * unL * (sL - unL)- rhoR * unR * (sR - unR);
        const double denom = rhoL * (sL - unL) - rhoR * (sR - unR);

        double s_star = safe_div(numerator, denom, tol);

        if (std::abs(pL - pR) < tol && std::abs(unL - unR) < tol) {
            s_star = unL;
        }

        const Flux<DIM> FL = compute_flux_normal<DIM>(UL, PL, n);
        const Flux<DIM> FR = compute_flux_normal<DIM>(UR, PR, n);

        if (0.0 <= sL) {
            return {FL, unL};
        }

        if (sR <= 0.0) {
            return {FR, unR};
        }

        if (!(sL < sR)) {
            return {0.5 * (FL + FR), 0.5 * (unL + unR)};
        }

        if (sL <= 0.0 && 0.0 <= s_star) {
            State<DIM> UL_star = make_state<DIM>(nmat);

            const double rhoL_star = compute_star_density(rhoL, sL, unL, s_star, tol);
            const double EL_star = compute_star_energy(rhoL_star, UL.E, rhoL, pL, sL, unL, s_star, tol);

            if (rhoL_star <= 0.0 || EL_star <= 0.0 || !std::isfinite(rhoL_star) || !std::isfinite(EL_star)) {
                return {hll_flux(UL, UR, FL, FR, sL, sR), s_star};
            }

            const double mass_scale = safe_div(rhoL_star, rhoL, tol);
            for (int k = 0; k < nmat; ++k) {
                UL_star.partial_mass[k] = UL.partial_mass[k] * mass_scale;
            }

            std::array<double, DIM> u_star = PL.vel;
            const double delta = s_star - unL;
            for (int d = 0; d < DIM; ++d) {
                u_star[d] += delta * n[d];
                UL_star.mom[d] = rhoL_star * u_star[d];
            }

            UL_star.E = EL_star;
            return {FL + sL * state_difference(UL_star, UL), s_star};
        }

        State<DIM> UR_star = make_state<DIM>(nmat);

        const double rhoR_star = compute_star_density(rhoR, sR, unR, s_star, tol);
        const double ER_star = compute_star_energy(rhoR_star, UR.E, rhoR, pR, sR, unR, s_star, tol);

        if (rhoR_star <= 0.0 || ER_star <= 0.0 || !std::isfinite(rhoR_star) || !std::isfinite(ER_star)) {
            return {hll_flux(UL, UR, FL, FR, sL, sR), s_star};
        }

        const double mass_scale = safe_div(rhoR_star, rhoR, tol);
        for (int k = 0; k < nmat; ++k) {
            UR_star.partial_mass[k] = UR.partial_mass[k] * mass_scale;
        }

        std::array<double, DIM> u_star = PR.vel;
        const double delta = s_star - unR;
        for (int d = 0; d < DIM; ++d) {
            u_star[d] += delta * n[d];
            UR_star.mom[d] = rhoR_star * u_star[d];
        }

        UR_star.E = ER_star;
        return {FR + sR * state_difference(UR_star, UR), s_star};
    }

} 



