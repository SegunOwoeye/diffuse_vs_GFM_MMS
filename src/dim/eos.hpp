#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/dim/eos_params.hpp"
#include "src/euler/eos.hpp"
#include "src/math/numerical_safety.hpp"

namespace dim {

struct MixtureEOS {
    static double material_pressure_from_density_energy(
        double rho,
        double e,
        const ::EOSParams& params
    )
    {
        Conserved<1> U{};
        U.rho = clamp_min(rho);
        U.mom[0] = 0.0;
        U.E = U.rho * std::max(e, 0.0);
        return MaterialEOS::pressure<1>(U, params);
    }

    static double material_sound_speed(
        double rho,
        double p,
        const ::EOSParams& params
    )
    {
        Conserved<1> U{};
        U.rho = clamp_min(rho);
        U.mom[0] = 0.0;
        U.E = U.rho * MaterialEOS::internal_energy(U.rho, p, params);
        return MaterialEOS::sound_speed<1>(U, params);
    }

    static double material_energy_pressure_derivative(
        double rho,
        double p,
        const ::EOSParams& params
    )
    {
        rho = clamp_min(rho);
        p = clamp_min(p);

        const double dp = std::max(1e-6 * std::max(std::abs(p), 1.0), 1e-8);
        const double p_lo = std::max(p - dp, 1e-12);
        const double p_hi = p + dp;
        const double e_lo = MaterialEOS::internal_energy(rho, p_lo, params);
        const double e_hi = MaterialEOS::internal_energy(rho, p_hi, params);
        return std::max(safe_div(e_hi - e_lo, p_hi - p_lo, 1e-14), 1e-14);
    }

    static double mixture_internal_energy_density(
        double p,
        const std::vector<double>& alpha,
        const std::vector<double>& rho,
        const EOSParams& params
    )
    {
        double energy_density = 0.0;
        for (int k = 0; k < params.nmat(); ++k) {
            const double alpha_k = std::max(finite_or(alpha[k]), 0.0);
            const double rho_k = clamp_min(rho[k]);
            const double p_k = clamp_min(p);
            energy_density += alpha_k * rho_k *
                MaterialEOS::internal_energy(rho_k, p_k, params.material[k]);
        }
        return finite_or(energy_density);
    }

    static double pressure_from_internal_energy(
        double rho_total,
        double e,
        const std::vector<double>& alpha,
        const std::vector<double>& rho,
        const EOSParams& params
    )
    {
        params.validate();

        if (static_cast<int>(alpha.size()) != params.nmat() ||
            static_cast<int>(rho.size()) != params.nmat()) {
            throw std::runtime_error("dim::MixtureEOS::pressure_from_internal_energy: mixture size mismatch");
        }

        rho_total = clamp_min(rho_total);
        e = finite_or(e);
        const double target = finite_or(rho_total * e);
        double fallback_pressure = 0.0;
        for (int k = 0; k < params.nmat(); ++k) {
            const double ek = target / rho_total;
            fallback_pressure += std::max(finite_or(alpha[k]), 0.0) *
                material_pressure_from_density_energy(clamp_min(rho[k]), ek, params.material[k]);
        }
        fallback_pressure = clamp_min(fallback_pressure);

        double lo = 1e-12;
        double hi = std::max(1.0, fallback_pressure);

        auto residual = [&](double p) {
            return mixture_internal_energy_density(p, alpha, rho, params) - target;
        };

        double flo = residual(lo);
        double fhi = residual(hi);

        for (int n = 0; n < 80 && std::isfinite(fhi) && fhi < 0.0; ++n) {
            hi *= 2.0;
            fhi = residual(hi);
        }

        if (!std::isfinite(flo) || !std::isfinite(fhi) || !(flo <= 0.0 && fhi >= 0.0)) {
            return clamp_min(fallback_pressure);
        }

        for (int n = 0; n < 80; ++n) {
            const double mid = 0.5 * (lo + hi);
            const double fmid = residual(mid);

            if (!std::isfinite(fmid)) {
                return clamp_min(fallback_pressure);
            }

            if (std::abs(fmid) <= 1e-10 * std::max(1.0, target)) {
                return clamp_min(mid);
            }

            if (fmid > 0.0) {
                hi = mid;
            }
            else {
                lo = mid;
            }
        }

        return clamp_min(0.5 * (lo + hi));
    }

    static double pressure_from_internal_energy(
        double rho,
        double e,
        const std::vector<double>& alpha,
        const EOSParams& params
    )
    {
        params.validate();

        if (static_cast<int>(alpha.size()) != params.nmat()) {
            throw std::runtime_error("dim::MixtureEOS::pressure_from_internal_energy: alpha size mismatch");
        }

        rho = clamp_min(rho);
        std::vector<double> rho_k(params.nmat(), clamp_min(rho));
        return pressure_from_internal_energy(rho, e, alpha, rho_k, params);
    }

    static double internal_energy_from_pressure(
        double rho_total,
        double p,
        const std::vector<double>& alpha,
        const std::vector<double>& rho,
        const EOSParams& params
    )
    {
        params.validate();

        if (static_cast<int>(alpha.size()) != params.nmat() ||
            static_cast<int>(rho.size()) != params.nmat()) {
            throw std::runtime_error("dim::MixtureEOS::internal_energy_from_pressure: mixture size mismatch");
        }

        rho_total = clamp_min(rho_total);
        p = clamp_min(p);

        return mixture_internal_energy_density(p, alpha, rho, params) / rho_total;
    }

    static double internal_energy_from_pressure(
        double rho,
        double p,
        const std::vector<double>& alpha,
        const EOSParams& params
    )
    {
        std::vector<double> rho_k(params.nmat(), clamp_min(rho));
        return internal_energy_from_pressure(rho, p, alpha, rho_k, params);
    }

    static double mixture_sound_speed(
        double rho_total,
        double p,
        const std::vector<double>& alpha,
        const std::vector<double>& rho,
        const EOSParams& params
    )
    {
        params.validate();

        if (static_cast<int>(alpha.size()) != params.nmat() ||
            static_cast<int>(rho.size()) != params.nmat()) {
            throw std::runtime_error("dim::MixtureEOS::mixture_sound_speed: mixture size mismatch");
        }

        rho_total = clamp_min(rho_total);
        p = clamp_min(p);

        double beta_mix = 0.0;
        for (int k = 0; k < params.nmat(); ++k) {
            const double rho_k = clamp_min(rho[k]);
            const double c = material_sound_speed(rho_k, p, params.material[k]);
            beta_mix += std::max(finite_or(alpha[k]), 0.0) / safe_denom(rho_k * c * c, 1e-14);
        }

        return std::sqrt(1.0 / (rho_total * safe_denom(beta_mix, 1e-14)));
    }

    static double generic_mixture_sound_speed(
        double rho_total,
        double p,
        const std::vector<double>& alpha,
        const std::vector<double>& rho,
        const EOSParams& params,
        const std::string& lambda_model
    )
    {
        if (lambda_model == "kapila" || lambda_model.empty()) {
            return mixture_sound_speed(rho_total, p, alpha, rho, params);
        }

        params.validate();
        if (static_cast<int>(alpha.size()) != params.nmat() ||
            static_cast<int>(rho.size()) != params.nmat()) {
            throw std::runtime_error("dim::MixtureEOS::generic_mixture_sound_speed: mixture size mismatch");
        }

        std::vector<double> sound_speed(params.nmat(), 0.0);
        std::vector<double> xi(params.nmat(), 0.0);
        double xi_mix = 0.0;
        double inverse_sound_sum = 0.0;

        for (int k = 0; k < params.nmat(); ++k) {
            const double alpha_k = std::max(finite_or(alpha[k]), 0.0);
            const double rho_k = clamp_min(rho[k]);
            sound_speed[k] = material_sound_speed(rho_k, p, params.material[k]);
            xi[k] = rho_k * material_energy_pressure_derivative(rho_k, p, params.material[k]);
            xi_mix += alpha_k * xi[k];
            inverse_sound_sum += alpha_k / safe_denom(sound_speed[k], 1e-14);
        }

        xi_mix = safe_denom(xi_mix, 1e-14);
        inverse_sound_sum = safe_denom(inverse_sound_sum, 1e-14);

        double rho_c_squared = 0.0;
        for (int k = 0; k < params.nmat(); ++k) {
            const double alpha_k = std::max(finite_or(alpha[k]), 0.0);
            double lambda_k = 1.0;

            if (lambda_model == "equal_velocity") {
                lambda_k = safe_div(1.0 / safe_denom(sound_speed[k], 1e-14), inverse_sound_sum, 1e-14);
            }
            else if (lambda_model != "allaire") {
                throw std::runtime_error("dim::MixtureEOS::generic_mixture_sound_speed: unknown lambda model");
            }

            rho_c_squared += lambda_k * alpha_k * xi[k] * sound_speed[k] * sound_speed[k] / xi_mix;
        }

        return std::sqrt(std::max(rho_c_squared / clamp_min(rho_total), 1e-14));
    }

    static double mixture_sound_speed(
        double rho,
        double p,
        const std::vector<double>& alpha,
        const EOSParams& params
    )
    {
        std::vector<double> rho_k(params.nmat(), clamp_min(rho));
        return mixture_sound_speed(rho, p, alpha, rho_k, params);
    }
};

using IdealGasEOS = MixtureEOS;

} 
