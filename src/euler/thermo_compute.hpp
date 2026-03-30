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

    double rho = require_positive(U.rho, "thermo: invalid density");

    T.rho = rho;

    for (int d = 0; d < DIM; ++d)
        T.vel[d] = U.mom[d] / rho;

    T.p = require_positive(EOS::template pressure<DIM>(U, params), "thermo: invalid pressure");
    T.c = require_finite(EOS::template sound_speed<DIM>(U, params), "thermo: invalid sound speed");

    return T;
}