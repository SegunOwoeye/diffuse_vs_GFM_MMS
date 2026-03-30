#pragma once

#include <algorithm>
#include <cmath>

#include "src/euler/state.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/math/numerical_safety.hpp"


template<int DIM, typename EOS>
inline Primitive<DIM> cons_to_prim(
    const Conserved<DIM>& U,
    const EOSParams& params,
    double rho_floor = 1e-10,
    double p_floor = 1e-10
)
{
    Primitive<DIM> P{};

    // [0] Density (clamped)
    const double rho = std::max(U.rho, rho_floor);
    P.rho = rho;

    // [1] Velocity
    double v2 = 0.0;

    for (int d = 0; d < DIM; ++d) {
        const double u = safe_div(U.mom[d], rho);
        P.vel[d] = u;
        v2 += u * u;
    }

    // [2] Internal energy
    const double kinetic = 0.5 * rho * v2;

    double e = safe_div((U.E - kinetic), rho);

    // guard against negative internal energy
    if (e < 0.0) {
        e = 0.0;
    }

    // [3] Pressure via EOS
    double p = EOS::pressure(U, params);

    // [4] Clamp pressure
    P.p = std::max(p, p_floor);

    return P;
}