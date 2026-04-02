#pragma once

#include <algorithm>
#include "src/euler/eos_params.hpp"

template<int DIM, typename EOS>
inline Conserved<DIM> prim_to_cons(
    const Primitive<DIM>& PR,
    const EOSParams& params
)
{
    Conserved<DIM> U;

    double rho = std::max(PR.rho, 1e-10);
    double p = std::max(PR.p, 1e-10);

    U.rho = rho;

    double v2 = 0.0;

    for (int d = 0; d < DIM; ++d) {
        U.mom[d] = rho * PR.vel[d];
        v2 += PR.vel[d] * PR.vel[d];
    }

    double kinetic = 0.5 * rho * v2;

    double e = EOS::internal_energy(rho, p, params);

    U.E = rho * e + kinetic;

    return U;
}

