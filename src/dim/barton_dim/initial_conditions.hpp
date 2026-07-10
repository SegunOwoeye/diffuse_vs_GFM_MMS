#pragma once

#include <stdexcept>
#include <vector>

#include "src/dim/barton_dim/config.hpp"
#include "src/dim/barton_dim/relaxation.hpp"

namespace dim::barton_dim {

template<int DIM>
inline bool contains(const Region<DIM>& region, const std::array<double, DIM>& position)
{
    if (region.shape == "circle") {
        if constexpr (DIM != 2) {
            throw std::runtime_error("Barton-DIM circle regions require dimension=2");
        }
        double distance_squared = 0.0;
        for (int d = 0; d < DIM; ++d) {
            const double delta = position[d] - region.center[d];
            distance_squared += delta * delta;
        }
        return distance_squared <= region.radius * region.radius;
    }
    if (region.shape != "box") throw std::runtime_error("Unknown Barton-DIM region shape: " + region.shape);
    for (int d = 0; d < DIM; ++d) {
        if (position[d] < region.lower[d] || position[d] >= region.upper[d]) return false;
    }
    return true;
}

template<int DIM>
inline State<DIM> state_from_region(const Region<DIM>& region, const Materials& materials)
{
    State<DIM> state{};
    state.alpha_solid = std::clamp(region.alpha_solid, 0.0, 1.0);
    const double alpha_fluid = 1.0 - state.alpha_solid;
    state.solid_mass = state.alpha_solid * region.rho_solid;
    state.fluid_mass = alpha_fluid * region.rho_fluid;
    const double density = state.solid_mass + state.fluid_mass;
    double velocity_squared = 0.0;
    for (int d = 0; d < DIM; ++d) {
        state.momentum[d] = density * region.velocity[d];
        velocity_squared += region.velocity[d] * region.velocity[d];
    }
    for (int q = 0; q < 9; ++q) state.solid_rhoF[q] = state.solid_mass * region.deformation[q];
    state.solid_rho_eqps = state.solid_mass * region.equivalent_plastic_strain;
    state.solid_rho_damage = state.solid_mass * region.damage;
    const double solid_temperature = region.has_solid_pressure
        ? solid::barton::material_temperature_from_rho_p(
            region.rho_solid, region.p_solid, materials.solid)
        : region.temperature;
    const double solid_internal = solid::barton::tensor_energy_from_F_T(
        region.deformation, solid_temperature, materials.solid);
    const double fluid_internal = MaterialEOS::internal_energy(
        std::max(region.rho_fluid, 1.0e-12), region.p_fluid, materials.fluid);
    state.solid_energy = state.solid_mass * (solid_internal + 0.5 * velocity_squared);
    state.fluid_energy = state.fluid_mass * (fluid_internal + 0.5 * velocity_squared);
    state.total_energy = state.solid_energy + state.fluid_energy;
    repair_state(state, materials);
    return state;
}

template<int DIM>
inline std::vector<State<DIM>> initialise(const Config<DIM>& config)
{
    int count = 1;
    std::array<double, DIM> spacing{};
    for (int d = 0; d < DIM; ++d) {
        count *= config.cells[d];
        spacing[d] = (config.domain_max[d] - config.domain_min[d]) / config.cells[d];
    }
    std::vector<State<DIM>> states(count);
    for (int linear = 0; linear < count; ++linear) {
        int value = linear;
        std::array<double, DIM> position{};
        for (int d = 0; d < DIM; ++d) {
            const int index = value % config.cells[d];
            value /= config.cells[d];
            position[d] = config.domain_min[d] + (index + 0.5) * spacing[d];
        }
        bool found = false;
        for (const Region<DIM>& region : config.regions) {
            if (contains<DIM>(region, position)) {
                states[linear] = state_from_region(region, config.materials);
                found = true;
                break;
            }
        }
        if (!found) throw std::runtime_error("Barton-DIM initial regions do not cover the domain");
    }
    return states;
}

} // namespace dim::barton_dim
