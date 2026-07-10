#pragma once

#include <cmath>
#include <stdexcept>
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/math/numerical_safety.hpp"

// Sound speed from primitive variables (rho, p)
inline double sound_speed_primitive(
    double rho,
    double p,
    const EOSParams& params,
    double tol = 1e-10
)
{
    rho = std::max(rho, tol);
    p = std::max(p, tol);

    Conserved<1> U{};
    U.rho = rho;
    U.mom[0] = 0.0;
    U.E = rho * MaterialEOS::internal_energy(rho, p, params);
    return MaterialEOS::sound_speed<1>(U, params);
}
