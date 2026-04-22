#pragma once

#include <array>
#include <cmath>

#include "src/dim/eos_params.hpp"
#include "src/math/numerical_safety.hpp"


// [0] Ideal Gas EOS
struct IdealGasEOS {

    // [0.1] Pressure from rho and e
        double rho,
        double e,
        double gamma
    )
    {
        rho = clamp_min(rho);
        e = std::max(e, 0.0);

        return clamp_min((gamma - 1.0) * rho * e);
    }

    // [0.2] Internal energy from rho and p
    static double internal_energy(
        double rho,
        double p,
        double gamma
    )
    {
        rho = clamp_min(rho);
        p = clamp_min(p);

        return safe_div(p, (gamma - 1.0) * rho);
    }

    // [0.3] Sound speed from rho and p
    static double sound_speed(
        double rho,
        double p,
        double gamma
    )
    {
        rho = clamp_min(rho);
        p = clamp_min(p);

        return std::sqrt(gamma * safe_div(p, rho));
    }


    // [0.4] Mixture gamma (pressure-equilibrium ideal-gas mixture)
    template<int NMAT>
    static double mixture_gamma(
        const std::array<double, NMAT>& alpha,
        const EOSParams<NMAT>& params
    )
    {
        double denom = 0.0;

        for (int k = 0; k < NMAT; ++k) {
            denom += alpha[k] / (params.gamma[k] - 1.0);
        }

        return 1.0 + safe_div(1.0, denom);
    }


    // [0.5] Mixture pressure
    template<int NMAT>
    static double mixture_pressure(
        double rho,
        double e,
        const std::array<double, NMAT>& alpha,
        const EOSParams<NMAT>& params
    )
    {
        rho = clamp_min(rho);
        e = std::max(e, 0.0);

        const double gmix = mixture_gamma<NMAT>(alpha, params);

        return clamp_min((gmix - 1.0) * rho * e);
    }


    // [0.6] Mixture sound speed
    template<int NMAT>
    static double mixture_sound_speed(
        const std::array<double, NMAT>& alpha,
        const std::array<double, NMAT>& rho_mat,
        double p,
        const EOSParams<NMAT>& params
    )
    {
        double denom = 0.0;

        for (int k = 0; k < NMAT; ++k) {

            const double rho_k = clamp_min(rho_mat[k]);

            const double c_k = sound_speed(
                rho_k,
                p,
                params.gamma[k]
            );

            denom += alpha[k] / (rho_k * c_k * c_k);
        }

        return std::sqrt(safe_div(1.0, denom));
    }
};


// [1] Stiffened Gas EOS
struct StiffenedGasEOS {

    // [1.1] Pressure from rho and e
    static double pressure(
        double rho,
        double e,
        double gamma,
        double p_inf
    )
    {
        rho = clamp_min(rho);
        e = std::max(e, 0.0);

        return clamp_min((gamma - 1.0) * rho * e - gamma * p_inf);
    }

    // [1.2] Internal energy from rho and p
    static double internal_energy(
        double rho,
        double p,
        double gamma,
        double p_inf
    )
    {
        rho = clamp_min(rho);
        p = clamp_min(p);

        return safe_div(p + gamma * p_inf, (gamma - 1.0) * rho);
    }

    // [1.3] Sound speed from rho and p
    static double sound_speed(
        double rho,
        double p,
        double gamma,
        double p_inf
    )
    {
        rho = clamp_min(rho);
        p = clamp_min(p);

        return std::sqrt(gamma * safe_div(p + p_inf, rho));
    }
};



