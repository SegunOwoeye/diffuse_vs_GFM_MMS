#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "src/dim/eos_params.hpp"
#include "src/math/numerical_safety.hpp"

namespace dim {

struct IdealGasEOS {
    static double pressure_from_internal_energy(
        double rho,
        double e,
        const std::vector<double>& alpha,
        const EOSParams& params
    )
    {
        params.validate();

        if (static_cast<int>(alpha.size()) != params.nmat()) {
            throw std::runtime_error("dim::IdealGasEOS::pressure_from_internal_energy: alpha size mismatch");
        }

        rho = clamp_min(rho);
        e = std::max(e, 0.0);

        double denom = 0.0;
        for (int k = 0; k < params.nmat(); ++k) {
            denom += alpha[k] / (params.gamma[k] - 1.0);
        }

        return clamp_min(rho * e / safe_denom(denom));
    }

    static double internal_energy_from_pressure(
        double rho,
        double p,
        const std::vector<double>& alpha,
        const EOSParams& params
    )
    {
        params.validate();

        if (static_cast<int>(alpha.size()) != params.nmat()) {
            throw std::runtime_error("dim::IdealGasEOS::internal_energy_from_pressure: alpha size mismatch");
        }

        rho = clamp_min(rho);
        p = clamp_min(p);

        double factor = 0.0;
        for (int k = 0; k < params.nmat(); ++k) {
            factor += alpha[k] / (params.gamma[k] - 1.0);
        }

        return p * factor / rho;
    }

    static double mixture_sound_speed(
        double rho_total,
        double p,
        const std::vector<double>& alpha,
        const EOSParams& params
    )
    {
        params.validate();

        if (static_cast<int>(alpha.size()) != params.nmat()) {
            throw std::runtime_error("dim::IdealGasEOS::mixture_sound_speed: alpha size mismatch");
        }

        rho_total = clamp_min(rho_total);
        p = clamp_min(p);

        double beta_mix = 0.0;
        for (int k = 0; k < params.nmat(); ++k) {
            beta_mix += alpha[k] / params.gamma[k];
        }

        return std::sqrt(p / (rho_total * safe_denom(beta_mix)));
    }

    static double material_sound_speed(
        double rho_k,
        double p,
        double gamma_k
    )
    {
        rho_k = clamp_min(rho_k);
        p = clamp_min(p);
        return std::sqrt(gamma_k * p / rho_k);
    }
};

} 
