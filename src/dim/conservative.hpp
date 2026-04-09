#pragma once

#include <algorithm>
#include <cmath>

#include "src/dim/state.hpp"
#include "src/dim/eos.hpp"
#include "src/dim/eos_params.hpp"
#include "src/math/numerical_safety.hpp"


// [0] Conserved -> Primitive (Diffuse Interface)
template<int DIM, int NMAT, typename EOS>
inline Primitive<DIM, NMAT> cons_to_prim(
    const Conserved<DIM, NMAT>& U,
    const EOSParams<NMAT>& params,
    double rho_floor = 1e-10,
    double p_floor = 1e-10,
    double alpha_floor = 1e-12
)
{
    Primitive<DIM, NMAT> P{};

    // [0.1] Alpha (clamp + renormalise)
    double alpha_sum = 0.0;

    for (int k = 0; k < NMAT; ++k) {
        P.alpha[k] = std::max(U.alpha[k], alpha_floor);
        alpha_sum += P.alpha[k];
    }

    if (alpha_sum <= 0.0) {
        P.alpha.fill(0.0);
        P.alpha[0] = 1.0;
        alpha_sum = 1.0;
    }

    for (int k = 0; k < NMAT; ++k) {
        P.alpha[k] /= alpha_sum;
    }

    // [0.2] Total density
    double rho = 0.0;

    for (int k = 0; k < NMAT; ++k) {
        rho += U.arho[k];
    }

    rho = std::max(rho, rho_floor);

    // [0.3] Velocity
    double v2 = 0.0;

    for (int d = 0; d < DIM; ++d) {
        const double u = safe_div(U.mom[d], rho);
        P.vel[d] = u;
        v2 += u * u;
    }

    // [0.4] Material densities
    for (int k = 0; k < NMAT; ++k) {
        P.rho[k] = safe_div(
            U.arho[k],
            std::max(P.alpha[k], alpha_floor)
        );
    }

    // [0.5] Internal energy
    const double kinetic = 0.5 * rho * v2;
    const double e = std::max(safe_div(U.E - kinetic, rho), 0.0);

    // [0.6] Pressure (mixture closure via EOS)
    const double p = EOS::template mixture_pressure<NMAT>(
        rho,
        e,
        P.alpha,
        params
    );

    P.p = std::max(p, p_floor);

    return P;
}

