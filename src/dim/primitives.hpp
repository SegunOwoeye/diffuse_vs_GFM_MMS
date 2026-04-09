#pragma once

#include <algorithm>
#include <cmath>

#include "src/dim/state.hpp"
#include "src/dim/eos.hpp"
#include "src/dim/eos_params.hpp"
#include "src/math/numerical_safety.hpp"


// [0] Primitive -> Conserved (Diffuse Interface)
template<int DIM, int NMAT, typename EOS>
inline Conserved<DIM, NMAT> prim_to_cons(
    const Primitive<DIM, NMAT>& P_in,
    const EOSParams<NMAT>& params,
    double rho_floor = 1e-10,
    double p_floor = 1e-10,
    double alpha_floor = 1e-12
)
{
    Conserved<DIM, NMAT> U{};

    Primitive<DIM, NMAT> P = P_in;

    // [0.1] Alpha clamp + renormalise
    double alpha_sum = 0.0;

    for (int k = 0; k < NMAT; ++k) {
        P.alpha[k] = std::max(P.alpha[k], alpha_floor);
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

    // [0.2] Pressure floor
    P.p = std::max(P.p, p_floor);

    // [0.3] Material partial densities
    double rho_total = 0.0;

    for (int k = 0; k < NMAT; ++k) {
        P.rho[k] = std::max(P.rho[k], rho_floor);
        U.arho[k] = P.alpha[k] * P.rho[k];
        rho_total += U.arho[k];
    }

    rho_total = std::max(rho_total, rho_floor);

    // [0.4] Momentum
    double v2 = 0.0;

    for (int d = 0; d < DIM; ++d) {
        U.mom[d] = rho_total * P.vel[d];
        v2 += P.vel[d] * P.vel[d];
    }

    // [0.5] Internal energy from EOS mixture closure
    const double gmix = EOS::template mixture_gamma<NMAT>(
        P.alpha,
        params
    );

    const double e = EOS::internal_energy(
        rho_total,
        P.p,
        gmix
    );

    const double kinetic = 0.5 * rho_total * v2;
    U.E = rho_total * e + kinetic;

    // [0.6] Store alpha
    U.alpha = P.alpha;

    return U;
}


