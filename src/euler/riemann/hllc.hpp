#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "src/euler/state.hpp"
#include "src/euler/thermo_state.hpp"
#include "src/euler/thermo_compute.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/flux.hpp"

#include "src/math/vector_ops.hpp"
#include "src/math/numerical_safety.hpp"


// [0] HLLC Flux using arbitrary interface normal
template<int DIM, typename EOS>
inline Conserved<DIM> hllc_flux_normal(
    const Conserved<DIM>& UL,
    const Conserved<DIM>& UR,
    const std::array<double, DIM>& normal,
    const EOSParams& paramsL,
    const EOSParams& paramsR,
    double tol = 1e-10
)
{
    // [0.1] Thermodynamic states
    const ThermoState<DIM> TL = compute_thermo<DIM, EOS>(UL, paramsL);
    const ThermoState<DIM> TR = compute_thermo<DIM, EOS>(UR, paramsR);

    // [0.2]
    const double mag = norm<DIM>(normal);
    const double inv_mag = (mag < tol) ? 1.0 : 1.0 / mag;

    const double rhoL = TL.rho;
    const double rhoR = TR.rho;

    const double unL = dot<DIM>(TL.vel, normal) * inv_mag;
    const double unR = dot<DIM>(TR.vel, normal) * inv_mag;

    const double pL = TL.p;
    const double pR = TR.p;

    const double cL = TL.c;
    const double cR = TR.c;

    // [0.3] Wave speed estimates
    const double sL = std::min(unL - cL, unR - cR);
    const double sR = std::max(unL + cL, unR + cR);

    // [0.4] Contact wave speed
    const double numerator = (pR - pL)
        + rhoL * unL * (sL - unL)
        - rhoR * unR * (sR - unR);

    const double denom = rhoL * (sL - unL) - rhoR * (sR - unR);

    const double s_star = safe_div(numerator, denom, tol);

    // [0.5] Star region densities
    const double rhoL_star = rhoL * safe_div((sL - unL), (sL - s_star), tol);
    const double rhoR_star = rhoR * safe_div((sR - unR), (sR - s_star), tol);

    // [0.6] Star region energies
    const double EL = UL.E;
    const double ER = UR.E;

    const double EL_star = rhoL_star * (
            safe_div(EL, rhoL, tol) + (s_star - unL) * (
            s_star + safe_div(pL, rhoL * (sL - unL), tol)
        )
    );

    const double ER_star = rhoR_star * (
            safe_div(ER, rhoR, tol) + (s_star - unR) * (
            s_star + safe_div(pR, rhoR * (sR - unR), tol)
        )
    );

    // To Delete
    if (!std::isfinite(rhoL_star) || !std::isfinite(EL_star) ||
        rhoL_star <= 0.0 || EL_star <= 0.0) {
        throw std::runtime_error(
            "hllc left star invalid: "
            "rhoL=" + std::to_string(rhoL) +
            " unL=" + std::to_string(unL) +
            " pL=" + std::to_string(pL) +
            " cL=" + std::to_string(cL) +
            " sL=" + std::to_string(sL) +
            " sR=" + std::to_string(sR) +
            " s_star=" + std::to_string(s_star) +
            " rhoL_star=" + std::to_string(rhoL_star) +
            " EL=" + std::to_string(EL) +
            " EL_star=" + std::to_string(EL_star)
        );
    }

    if (!std::isfinite(rhoR_star) || !std::isfinite(ER_star) ||
        rhoR_star <= 0.0 || ER_star <= 0.0) {
        throw std::runtime_error("hllc_flux_normal: invalid right star state");
    }


    // [0.7] Physical fluxes
    const Conserved<DIM> FL = compute_flux_normal<DIM, EOS>(UL, normal, paramsL);
    const Conserved<DIM> FR = compute_flux_normal<DIM, EOS>(UR, normal, paramsR);

    // [0.8] HLLC region selection
    if (0.0 <= sL) {
        return FL;
    }

    if (sL <= 0.0 && 0.0 <= s_star) {
        Conserved<DIM> UL_star{};

        UL_star.rho = rhoL_star;

        std::array<double, DIM> u_star = TL.vel;
        const double delta = s_star - unL;

        for (int d = 0; d < DIM; ++d) {
            u_star[d] += delta * normal[d];
            UL_star.mom[d] = rhoL_star * u_star[d];
        }

        UL_star.E = EL_star;

        return FL + sL * (UL_star - UL);
    }

    if (s_star <= 0.0 && 0.0 <= sR) {
        Conserved<DIM> UR_star{};

        UR_star.rho = rhoR_star;

        std::array<double, DIM> u_star = TR.vel;
        const double delta = s_star - unR;

        for (int d = 0; d < DIM; ++d) {
            u_star[d] += delta * normal[d];
            UR_star.mom[d] = rhoR_star * u_star[d];
        }

        UR_star.E = ER_star;

        return FR + sR * (UR_star - UR);
    }

    return FR;
}



