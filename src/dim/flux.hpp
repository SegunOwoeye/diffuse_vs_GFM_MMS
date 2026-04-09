#pragma once

#include <algorithm>
#include <array>

#include "src/dim/state.hpp"
#include "src/dim/conservative.hpp"
#include "src/dim/eos.hpp"
#include "src/dim/eos_params.hpp"

#include "src/math/vector_ops.hpp"
#include "src/math/numerical_safety.hpp"


// [1] Diffuse Interface Method Flux in normal direction
template<int DIM, int NMAT, typename EOS>
inline Conserved<DIM, NMAT> compute_flux_normal(
    const Conserved<DIM, NMAT>& U,
    const std::array<double, DIM>& normal,
    const EOSParams<NMAT>& params
)
{
    Conserved<DIM, NMAT> F{};

    // [1.1] Recover primitive state
    const auto P = cons_to_prim<DIM, NMAT, EOS>(U, params);

    // [1.2] Total density
    double rho = 0.0;
    for (int k = 0; k < NMAT; ++k) {
        rho += U.arho[k];
    }

    rho = require_positive(rho, "flux: invalid density");

    const double p = require_positive(P.p, "flux: invalid pressure");

    // [1.3] Unit normal
    const double mag = norm<DIM>(normal);

    std::array<double, DIM> n{};
    if (mag < 1e-14) {
        n[0] = 1.0;
    } else {
        n = scale<DIM>(1.0 / mag, normal);
    }

    // [1.4] Normal velocity
    const double un = dot<DIM>(P.vel, n);

    // [1.5] Material mass fluxes
    for (int k = 0; k < NMAT; ++k) {
        F.arho[k] = U.arho[k] * un;
    }

    // [1.6] Momentum flux
    for (int d = 0; d < DIM; ++d) {
        F.mom[d] = rho * P.vel[d] * un + p * n[d];
    }

    // [1.7] Energy flux
    F.E = (U.E + p) * un;

    // [1.8] Volume fraction transport
    for (int k = 0; k < NMAT; ++k) {
        F.alpha[k] = P.alpha[k] * un;
    }

    return F;
}