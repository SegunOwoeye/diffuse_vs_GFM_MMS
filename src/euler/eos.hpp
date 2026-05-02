#pragma once

#include <cmath>
#include <algorithm>

#include "src/euler/state.hpp"
#include "src/euler/eos_params.hpp"
#include "src/math/numerical_safety.hpp"


// [1] Ideal Gas EOS
struct IdealGasEOS {
    // Pressure
    template<int DIM>
    static double pressure(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = clamp_min(U.rho);

        double mom2 = 0.0;
        for (int d = 0; d < DIM; ++d)
            mom2 += U.mom[d] * U.mom[d];

        const double kineticE = 0.5 * safe_div(mom2, rho);

        const double p_raw = (params.gamma - 1.0) * (U.E - kineticE);

        return clamp_min(p_raw);
    }

    // Speed of Sound
    template<int DIM>
    static double sound_speed(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = clamp_min(U.rho);
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

    // Combined Pressure and sound speed calculation
    template<int DIM>
    static inline void compute_p_c(
        const Conserved<DIM>& U,
        const EOSParams& params,
        double& p,
        double& c
    )
    {
        const double rho = clamp_min(U.rho);

        double mom2 = 0.0;
        for (int d = 0; d < DIM; ++d)
            mom2 += U.mom[d] * U.mom[d];

        const double kineticE = 0.5 * safe_div(mom2, rho);

        const double p_raw = (params.gamma - 1.0) * (U.E - kineticE);
        p = clamp_min(p_raw);

        c = std::sqrt(params.gamma * safe_div(p, rho));
    }

    /* 
        Recover ghost density from p* using EOS-consistent closure
        (preserves thermodynamic branch instead of fixing density)
    */

    // Entropy invariant (p / rho^gamma)
    static double entropy_invariant(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        rho = clamp_min(rho);
        p = clamp_min(p);

        return safe_div(p, std::pow(rho, params.gamma));
    }


    // Recover density from (p, invariant)
    static double density_from_p_invariant(
        double p,
        double K,
        const EOSParams& params
    )
    {
        p = clamp_min(p);
        K = clamp_min(K);

        return std::pow(safe_div(p, K), safe_div(1.0, params.gamma));
    }
};



// [2] Stiffened Gas EOS
struct StiffenedGasEOS {
    // Pressure
    template<int DIM>
    static double pressure(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = clamp_min(U.rho);

        double mom2 = 0.0;
        for (int d = 0; d < DIM; ++d)
            mom2 += U.mom[d] * U.mom[d];

        const double kineticE = 0.5 * safe_div(mom2,rho);

        const double p_raw = (params.gamma - 1.0) * (U.E - kineticE)
            - params.gamma * params.p_inf;

        return clamp_min(p_raw);
    }

    // Speed of Sound
    template<int DIM>
    static double sound_speed(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = clamp_min(U.rho);
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

    // Combined Pressure and sound speed calculation
    template<int DIM>
    static inline void compute_p_c(
        const Conserved<DIM>& U,
        const EOSParams& params,
        double& p,
        double& c
    )
    {
        const double rho = clamp_min(U.rho);

        double mom2 = 0.0;
        for (int d = 0; d < DIM; ++d)
            mom2 += U.mom[d] * U.mom[d];

        const double kineticE = 0.5 * safe_div(mom2, rho);

        const double p_raw = (params.gamma - 1.0) * (U.E - kineticE)
            - params.gamma * params.p_inf;

        p = clamp_min(p_raw);

        c = std::sqrt(params.gamma * safe_div((p + params.p_inf), rho));
    }


    /* 
        Recover ghost density from p* using EOS-consistent closure
        (preserves thermodynamic branch instead of fixing density)
    */

    // Entropy invariant ((p + p_inf) / rho^gamma)
    static double entropy_invariant(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        rho = clamp_min(rho);
        p = clamp_min(p);

        return safe_div((p + params.p_inf), std::pow(rho, params.gamma));
    }

    // Recover density from (p, invariant)
    static double density_from_p_invariant(
        double p,
        double K,
        const EOSParams& params
    )
    {
        p = clamp_min(p);
        K = clamp_min(K);

        return std::pow(safe_div(p + params.p_inf, K),
            safe_div(1.0, params.gamma)
        );
    }






};




