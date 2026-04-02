#pragma once

#include <cmath>
#include <stdexcept>
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

    if (params.type == "ideal_gas") {
        return std::sqrt(params.gamma * safe_div(p, rho));
    }
    else if (params.type == "stiffened_gas") {
        return std::sqrt(params.gamma * safe_div((p + params.p_inf), rho));
    }
    else {
        throw std::runtime_error("Unknown EOS type in sound_speed_primitive");
    }
}

