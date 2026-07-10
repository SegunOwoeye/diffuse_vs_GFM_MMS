#pragma once

#include <algorithm>
#include <cmath>

#include "src/dim/barton_dim/primitives.hpp"
#include "src/solid/elastoplastic/barton/plasticity.hpp"

namespace dim::barton_dim {

template<int DIM>
inline double kinetic_energy_density(const State<DIM>& state)
{
    const double density = std::max(state.solid_mass + state.fluid_mass, 1.0e-12);
    double momentum_squared = 0.0;
    for (double value : state.momentum) momentum_squared += value * value;
    return 0.5 * momentum_squared / density;
}

template<int DIM>
inline std::array<double, 9> solid_deformation(const State<DIM>& state)
{
    const double mass = std::max(state.solid_mass, 1.0e-12);
    std::array<double, 9> deformation{};
    for (int q = 0; q < 9; ++q) deformation[q] = state.solid_rhoF[q] / mass;
    deformation[0] = std::max(deformation[0], 1.0e-10);
    deformation[4] = std::max(deformation[4], 1.0e-10);
    deformation[8] = std::max(deformation[8], 1.0e-10);
    return deformation;
}

template<int DIM>
inline double solid_internal_energy_at_pressure(
    const State<DIM>& state,
    const Materials& materials,
    double pressure)
{
    const double alpha = std::max(state.alpha_solid, 1.0e-12);
    const double rho = std::max(state.solid_mass / alpha, 1.0e-12);
    const double temperature = solid::barton::material_temperature_from_rho_p(
        rho, pressure, materials.solid);
    return solid::barton::tensor_energy_from_F_T(
        solid_deformation(state), temperature, materials.solid);
}

template<int DIM>
inline double mixture_internal_energy_at_pressure(
    const State<DIM>& state,
    const Materials& materials,
    double pressure)
{
    const double alpha_solid = std::clamp(state.alpha_solid, 0.0, 1.0);
    const double alpha_fluid = 1.0 - alpha_solid;
    const double rho_solid = state.solid_mass / std::max(alpha_solid, 1.0e-12);
    const double rho_fluid = state.fluid_mass / std::max(alpha_fluid, 1.0e-12);
    return alpha_solid * rho_solid * solid_internal_energy_at_pressure(state, materials, pressure) +
        alpha_fluid * rho_fluid * MaterialEOS::internal_energy(
            std::max(rho_fluid, 1.0e-12), pressure, materials.fluid);
}

template<int DIM>
inline void reinitialise_phase_energies(State<DIM>& state, const Materials& materials)
{
    constexpr double phase_floor = 1.0e-8;
    const double alpha_solid = std::clamp(state.alpha_solid, 0.0, 1.0);
    const double alpha_fluid = 1.0 - alpha_solid;
    if (alpha_solid <= phase_floor || state.solid_mass <= phase_floor) {
        state.solid_energy = 0.0;
        state.fluid_energy = state.total_energy;
        return;
    }
    if (alpha_fluid <= phase_floor || state.fluid_mass <= phase_floor) {
        state.solid_energy = state.total_energy;
        state.fluid_energy = 0.0;
        return;
    }

    const double internal_target = state.total_energy - kinetic_energy_density(state);
    double low = 1.0e-8;
    double high = std::max(1.0e8, std::abs(internal_target) * 100.0 + 1.0e8);
    const auto residual = [&](double pressure) {
        return mixture_internal_energy_at_pressure(state, materials, pressure) - internal_target;
    };
    double low_residual = residual(low);
    double high_residual = residual(high);
    for (int attempt = 0; attempt < 20 && high_residual < 0.0; ++attempt) {
        high *= 10.0;
        high_residual = residual(high);
    }
    double pressure = low;
    if (low_residual <= 0.0 && high_residual >= 0.0) {
        for (int iteration = 0; iteration < 80; ++iteration) {
            const double mid = 0.5 * (low + high);
            if (residual(mid) <= 0.0) low = mid;
            else high = mid;
        }
        pressure = 0.5 * (low + high);
    }

    const double density = std::max(state.solid_mass + state.fluid_mass, 1.0e-12);
    double velocity_squared = 0.0;
    for (double momentum : state.momentum) velocity_squared += (momentum / density) * (momentum / density);
    const double rho_solid = state.solid_mass / alpha_solid;
    const double rho_fluid = state.fluid_mass / alpha_fluid;
    const double solid_internal = solid_internal_energy_at_pressure(state, materials, pressure);
    const double fluid_internal = MaterialEOS::internal_energy(rho_fluid, pressure, materials.fluid);
    state.solid_energy = alpha_solid * rho_solid * (solid_internal + 0.5 * velocity_squared);
    state.fluid_energy = alpha_fluid * rho_fluid * (fluid_internal + 0.5 * velocity_squared);
}

template<int DIM>
inline void apply_plastic_relaxation(State<DIM>& state, const Materials& materials, double dt)
{
    if (state.alpha_solid <= 1.0e-8 || state.solid_mass <= 1.0e-8 || dt <= 0.0) return;
    const double alpha = state.alpha_solid;
    solid::barton::TensorState2D solid_state{};
    solid_state.rho = state.solid_mass / alpha;
    solid_state.mom[0] = state.momentum[0] / alpha;
    solid_state.mom[1] = DIM > 1 ? state.momentum[1] / alpha : 0.0;
    solid_state.E = state.solid_energy / alpha;
    for (int q = 0; q < 9; ++q) solid_state.rhoF[q] = state.solid_rhoF[q] / alpha;
    solid_state.rhoEqps = state.solid_rho_eqps / alpha;
    solid_state.rhoDamage = state.solid_rho_damage / alpha;
    solid::barton::apply_tensor_plastic_relaxation(solid_state, materials.solid, dt);
    for (int q = 0; q < 9; ++q) state.solid_rhoF[q] = alpha * solid_state.rhoF[q];
    state.solid_rho_eqps = alpha * solid_state.rhoEqps;
    state.solid_rho_damage = alpha * solid_state.rhoDamage;
}

template<int DIM>
inline void repair_state(State<DIM>& state, const Materials& materials)
{
    constexpr double phase_floor = 1.0e-12;
    state.alpha_solid = std::clamp(state.alpha_solid, 0.0, 1.0);
    state.solid_mass = std::max(state.solid_mass, 0.0);
    state.fluid_mass = std::max(state.fluid_mass, 0.0);
    if (state.solid_mass <= phase_floor) {
        state.alpha_solid = 0.0;
        state.solid_mass = 0.0;
        state.solid_energy = 0.0;
        state.solid_rhoF.fill(0.0);
        state.solid_rho_eqps = 0.0;
        state.solid_rho_damage = 0.0;
    }
    else if (state.fluid_mass <= phase_floor) {
        state.alpha_solid = 1.0;
        state.fluid_mass = 0.0;
        state.fluid_energy = 0.0;
    }
    else {
        const double rho_min = 0.2 * materials.solid.rho0;
        const double rho_max = 2.0 * materials.solid.rho0;
        const double alpha_min = state.solid_mass / rho_max;
        const double alpha_max = state.solid_mass / rho_min;
        state.alpha_solid = std::clamp(
            state.alpha_solid,
            std::clamp(alpha_min, 0.0, 1.0),
            std::clamp(alpha_max, 0.0, 1.0));
    }
    for (double& value : state.momentum) if (!std::isfinite(value)) value = 0.0;
    if (!std::isfinite(state.total_energy)) {
        state.total_energy = kinetic_energy_density(state) + 1.0e-6;
    }
    state.total_energy = std::max(state.total_energy, kinetic_energy_density(state) + 1.0e-8);
    for (int q = 0; q < 9; ++q) {
        if (!std::isfinite(state.solid_rhoF[q])) state.solid_rhoF[q] = 0.0;
    }
    if (state.solid_mass > 1.0e-12) {
        if (state.solid_rhoF[0] <= 0.0) state.solid_rhoF[0] = state.solid_mass;
        if (state.solid_rhoF[4] <= 0.0) state.solid_rhoF[4] = state.solid_mass;
        if (state.solid_rhoF[8] <= 0.0) state.solid_rhoF[8] = state.solid_mass;
    }
}

} // namespace dim::barton_dim
