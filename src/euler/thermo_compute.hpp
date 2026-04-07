#pragma once

#include <algorithm>

#include "src/euler/thermo_state.hpp"
#include "src/euler/eos.hpp"

#include"src/math/numerical_safety.hpp"

template<int DIM, typename EOS>
inline ThermoState<DIM> compute_thermo(
    const Conserved<DIM>& U,
    const EOSParams& params
)
{
    ThermoState<DIM> T;

    const double rho = require_positive(U.rho, "thermo: invalid density");
    T.rho = rho;

    // velocity
    #pragma unroll
    for (int d = 0; d < DIM; ++d) {
        T.vel[d] = U.mom[d] / rho;
    }

    EOS::template compute_p_c<DIM>(U, params, T.p, T.c);
    
    return T;
}