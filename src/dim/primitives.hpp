#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "src/dim/eos.hpp"
#include "src/dim/eos_params.hpp"
#include "src/dim/state.hpp"
#include "src/math/numerical_safety.hpp"

namespace dim {

    // [1] Builds the full alpha vector for all materials from the stored independent N-1 values
    inline std::vector<double> full_alpha_from_independent(
        const std::vector<double>& independent_alpha,
        int nmat
    )
    {
        if (nmat <= 0) {
            throw std::runtime_error("dim::full_alpha_from_independent: nmat must be positive");
        }

        const int expected = (nmat > 1) ? (nmat - 1) : 0;
        if (static_cast<int>(independent_alpha.size()) != expected) {
            throw std::runtime_error("dim::full_alpha_from_independent: alpha size mismatch");
        }

        std::vector<double> alpha(nmat, 0.0);
        double sum = 0.0;

        for (int k = 0; k < expected; ++k) {
            alpha[k] = independent_alpha[k];
            sum += alpha[k];
        }

        alpha[nmat - 1] = 1.0 - sum;
        return alpha;
    }

    // [2] Floors and renormalizes the volume fractions so they stay non-negative and sum to one
    inline void sanitise_alpha(
        std::vector<double>& alpha,
        double alpha_floor = 0.0
    )
    {
        if (alpha.empty()) {
            throw std::runtime_error("dim::sanitise_alpha: alpha cannot be empty");
        }

        double sum = 0.0;
        for (double& value : alpha) {
            value = std::max(value, alpha_floor);
            sum += value;
        }

        if (sum <= 0.0) {
            alpha.assign(alpha.size(), 0.0);
            alpha[0] = 1.0;
            return;
        }

        for (double& value : alpha) {
            value /= sum;
        }
    }

    // [3] A wrapper that extracts the full material fraction vector from a State function
    template<int DIM>
    inline std::vector<double> full_alpha(
        const State<DIM>& U,
        int nmat
    )
    {
        return full_alpha_from_independent(U.alpha, nmat);
    }

    // [4] Converts a full alpha vector to stored N-1 independent form by dropping the last entry
    inline std::vector<double> independent_alpha(
        const std::vector<double>& alpha
    )
    {
        if (alpha.empty()) {
            throw std::runtime_error("dim::independent_alpha: alpha cannot be empty");
        }

        if (alpha.size() == 1) {
            return {};
        }

        return std::vector<double>(alpha.begin(), alpha.end() - 1);
    }

    // [5] Computes the mixture density by summing all partial masses
    template<int DIM>
    inline double total_density(const State<DIM>& U)
    {
        double rho_total = 0.0;
        for (double value : U.partial_mass) {
            rho_total += value;
        }

        return rho_total;
    }

    // [6] Converts conservative variables into primitive variables
    template<int DIM>
    inline Primitive<DIM> cons_to_prim(
        const State<DIM>& U,
        const EOSParams& params,
        double rho_floor = 1e-12,
        double p_floor = 1e-12,
        double alpha_floor = 1e-12
    )
    {
        params.validate();
        require_compatible_state(U, params.nmat(), "dim::cons_to_prim");

        Primitive<DIM> P{};
        P.alpha = full_alpha(U, params.nmat());
        sanitise_alpha(P.alpha, alpha_floor);
        P.rho.assign(params.nmat(), rho_floor);

        // Density calc
        double rho_total = 0.0;
        for (double value : U.partial_mass) {
            rho_total += value;
        }
        rho_total = std::max(rho_total, rho_floor);

        // Velocity calc
        double velocity_squared = 0.0;
        for (int d = 0; d < DIM; ++d) {
            P.vel[d] = safe_div(U.mom[d], rho_total);
            velocity_squared += P.vel[d] * P.vel[d];
        }

        // Energy calc
        const double kinetic = 0.5 * rho_total * velocity_squared;
        const double internal_energy = std::max(safe_div(U.E - kinetic, rho_total), 0.0);
        P.p = std::max(
            IdealGasEOS::pressure_from_internal_energy(rho_total, internal_energy, P.alpha, params), p_floor
        );

        for (int k = 0; k < params.nmat(); ++k) {
            const double alpha_k = std::max(P.alpha[k], alpha_floor);
            P.rho[k] = std::max(safe_div(U.partial_mass[k], alpha_k), rho_floor);
        }

        return P;
    }

    // [7] Converts primitive variables back into conservative form
    template<int DIM>
    inline State<DIM> prim_to_cons(
        const Primitive<DIM>& P_in,
        const EOSParams& params,
        double rho_floor = 1e-12,
        double p_floor = 1e-12,
        double alpha_floor = 1e-12
    )
    {
        params.validate();

        if (static_cast<int>(P_in.alpha.size()) != params.nmat()) {
            throw std::runtime_error("dim::prim_to_cons: alpha size mismatch");
        }

        if (static_cast<int>(P_in.rho.size()) != params.nmat()) {
            throw std::runtime_error("dim::prim_to_cons: rho size mismatch");
        }

        Primitive<DIM> P = P_in;
        sanitise_alpha(P.alpha, alpha_floor);

        State<DIM> U = make_state<DIM>(params.nmat());

        double rho_total = 0.0;
        for (int k = 0; k < params.nmat(); ++k) {
            P.rho[k] = std::max(P.rho[k], rho_floor);
            U.partial_mass[k] = P.alpha[k] * P.rho[k];
            rho_total += U.partial_mass[k];
        }
        rho_total = std::max(rho_total, rho_floor);

        P.p = std::max(P.p, p_floor);

        double velocity_squared = 0.0;
        for (int d = 0; d < DIM; ++d) {
            U.mom[d] = rho_total * P.vel[d];
            velocity_squared += P.vel[d] * P.vel[d];
        }

        const double internal_energy = IdealGasEOS::internal_energy_from_pressure(
            rho_total, P.p, P.alpha, params);

        U.E = rho_total * internal_energy + 0.5 * rho_total * velocity_squared;
        U.alpha = independent_alpha(P.alpha);

        return U;
    }

    // [8] Computes the coefficients used in the non-conservative volume-fraction evolution term for the isobaric closure
    template<int DIM>
    inline std::vector<double> alpha_rhs_coefficients(
        const Primitive<DIM>& P,
        const EOSParams& params,
        double p_floor = 1e-12
    )
    {
        params.validate();

        if (static_cast<int>(P.alpha.size()) != params.nmat()) {
            throw std::runtime_error("dim::alpha_rhs_coefficients: alpha size mismatch");
        }

        const double p = std::max(P.p, p_floor);

        std::vector<double> rhs(params.nmat(), 0.0);
        double beta_mix = 0.0;

        for (int k = 0; k < params.nmat(); ++k) {
            beta_mix += P.alpha[k] / (params.gamma[k] * p);
        }

        beta_mix = safe_denom(beta_mix);

        for (int k = 0; k < params.nmat(); ++k) {
            const double beta_k = 1.0 / (params.gamma[k] * p);
            rhs[k] = P.alpha[k] * beta_k / beta_mix;
        }

        return rhs;
    }

    // [9] Applies safety fixes after update
    template<int DIM>
    inline void repair_state(
        State<DIM>& U,
        const EOSParams& params,
        double mass_floor = 1e-12,
        double p_floor = 1e-12,
        double inactive_alpha_floor = 1e-10
    )
    {
        params.validate();
        require_compatible_state(U, params.nmat(), "dim::repair_state");

        const double rho_before = std::max(total_density(U), mass_floor);
        std::array<double, DIM> velocity_before{};
        double velocity_squared_before = 0.0;
        for (int d = 0; d < DIM; ++d) {
            velocity_before[d] = safe_div(U.mom[d], rho_before);
            velocity_squared_before += velocity_before[d] * velocity_before[d];
        }

        std::vector<double> alpha_before = full_alpha(U, params.nmat());
        sanitise_alpha(alpha_before, 0.0);

        const double kinetic_before = 0.5 * rho_before * velocity_squared_before;
        const double internal_before =
            std::max(safe_div(U.E - kinetic_before, rho_before), 0.0);
        const double pressure_before = std::max(
            IdealGasEOS::pressure_from_internal_energy(
                rho_before,
                internal_before,
                alpha_before,
                params
            ),
            p_floor
        );

        // clamps invalid masses
        for (double& value : U.partial_mass) {
            value = std::max(value, 0.0);
        }
        
        // re-normalizes alphas
        std::vector<double> alpha_full = full_alpha(U, params.nmat());
        sanitise_alpha(alpha_full, 0.0);
        U.alpha = independent_alpha(alpha_full);

        bool removed_inactive_mass = false;
        for (int k = 0; k < params.nmat(); ++k) {
            if (alpha_full[k] <= inactive_alpha_floor &&
                U.partial_mass[k] > mass_floor)
            {
                U.partial_mass[k] = 0.0;
                removed_inactive_mass = true;
            }
        }

        double rho_total = total_density(U);
        if (rho_total <= 0.0) {
            int dominant_material = 0;
            for (int k = 1; k < params.nmat(); ++k) {
                if (alpha_full[k] > alpha_full[dominant_material]) {
                    dominant_material = k;
                }
            }
            U.partial_mass[dominant_material] = mass_floor;
            rho_total = mass_floor;
        }

        if (removed_inactive_mass) {
            for (int d = 0; d < DIM; ++d) {
                U.mom[d] = rho_total * velocity_before[d];
            }

            const double internal_energy =
                IdealGasEOS::internal_energy_from_pressure(
                    rho_total,
                    pressure_before,
                    alpha_full,
                    params
                );
            U.E = rho_total * internal_energy +
                  0.5 * rho_total * velocity_squared_before;
        }

        // enforces enough total energy to keep pressure/internal energy physical
        double velocity_squared = 0.0;
        for (int d = 0; d < DIM; ++d) {
            velocity_squared += std::pow(safe_div(U.mom[d], rho_total), 2);
        }

        double energy_factor = 0.0;
        for (int k = 0; k < params.nmat(); ++k) {
            energy_factor += alpha_full[k] / (params.gamma[k] - 1.0);
        }

        const double kinetic = 0.5 * rho_total * velocity_squared;
        const double minimum_internal = p_floor * energy_factor;
        U.E = std::max(U.E, kinetic + minimum_internal);
    }


} 



