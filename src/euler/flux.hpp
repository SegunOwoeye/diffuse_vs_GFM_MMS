#pragma once

#include <algorithm>
#include <array>

#include "src/euler/state.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"

#include "src/math/vector_ops.hpp"
#include "src/math/numerical_safety.hpp"


// [1] Euler Flux in normal unit direction
template<int DIM, typename EOS>
inline Conserved<DIM> compute_flux_normal(
    const Conserved<DIM>& U,
    const std::array<double, DIM>& normal,
    const EOSParams& params
)
{
    Conserved<DIM> F{};

    const double rho = require_positive(U.rho, "flux: invalid density");

    double p = require_positive(EOS::template pressure<DIM>(U, params), "flux: invalid pressure");

    std::array<double, DIM> vel{};
    for (int d = 0; d < DIM; ++d) {
        vel[d] = U.mom[d] / rho;
    }

    const double mag = norm<DIM>(normal);

    std::array<double, DIM> n{};
    if (mag < 1e-14) {
        n[0] = 1.0;
    } else {
        n = scale<DIM>(1.0 / mag, normal);
    }

    const double un = dot<DIM>(vel, n);

    F.rho = rho * un;

    for (int d = 0; d < DIM; ++d) {
        F.mom[d] = rho * vel[d] * un + p * n[d];
    }

    F.E = (U.E + p) * un;

    return F;
}

