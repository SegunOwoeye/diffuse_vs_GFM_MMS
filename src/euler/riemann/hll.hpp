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


// [0] HLL Flux using arbitrary interface normal
template<int DIM, typename EOS>
inline Conserved<DIM> hll_flux_normal(
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

    // [0.2] Normal magnitude and projected normal velocities
    const double mag = norm<DIM>(normal);
    const double inv_mag = (mag < tol) ? 1.0 : 1.0 / mag;

    const double unL = dot<DIM>(TL.vel, normal) * inv_mag;
    const double unR = dot<DIM>(TR.vel, normal) * inv_mag;

    const double cL = TL.c;
    const double cR = TR.c;

    // [0.3] Physical fluxes
    const Conserved<DIM> FL = compute_flux_normal<DIM, EOS>(UL, normal, paramsL);
    const Conserved<DIM> FR = compute_flux_normal<DIM, EOS>(UR, normal, paramsR);

    // [0.4] HLL wave speed estimates
    const double sL = std::min(unL - cL, unR - cR);
    const double sR = std::max(unL + cL, unR + cR);

    // [0.5] Supersonic regions
    if (0.0 <= sL) {
        return FL;
    }

    if (sR <= 0.0) {
        return FR;
    }

    // [0.6] HLL intermediate flux
    const double denom = sR - sL;

    if (std::abs(denom) < tol) {
        return 0.5 * (FL + FR);
    }

    return (sR * FL - sL * FR + sL * sR * (UR - UL)) / denom;


}


