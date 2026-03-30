#pragma once

#include <cmath>
#include <algorithm>

#include "src/euler/state.hpp"
#include "src/euler/eos_params.hpp"
#include "src/math/numerical_safety.hpp"


// [1] Ideal Gas EOS
struct IdealGasEOS {

    template<int DIM>
    static double pressure(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = std::max(U.rho, 1e-10);

        double mom2 = 0.0;
        for (int d = 0; d < DIM; ++d)
            mom2 += U.mom[d] * U.mom[d];

        const double kineticE = 0.5 * safe_div(mom2, rho);

        const double p_raw = (params.gamma - 1.0) * (U.E - kineticE);

        return std::max(p_raw, 1e-10);
    }


    template<int DIM>
    static double sound_speed(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = std::max(U.rho, 1e-10);
        const double p = pressure(U, params);

        return std::sqrt(params.gamma * safe_div(p, rho));
    }


    // Internal energy
    static double internal_energy(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        return safe_div(p, ((params.gamma - 1.0) * rho));
    }
};



// [2] Stiffened Gas EOS
struct StiffenedGasEOS {

    template<int DIM>
    static double pressure(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = std::max(U.rho, 1e-10);

        double mom2 = 0.0;
        for (int d = 0; d < DIM; ++d)
            mom2 += U.mom[d] * U.mom[d];

        const double kineticE = 0.5 * safe_div(mom2,rho);

        const double p_raw = (params.gamma - 1.0) * (U.E - kineticE)
            - params.gamma * params.p_inf;

        return std::max(p_raw, 1e-10);
    }


    template<int DIM>
    static double sound_speed(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = std::max(U.rho, 1e-10);
        const double p = pressure(U, params);

        return std::sqrt(params.gamma * safe_div((p + params.p_inf), rho));
    }


    // Internal energy
    static double internal_energy(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        return safe_div((p + params.gamma * params.p_inf),
            ((params.gamma - 1.0) * rho));
    }
};




