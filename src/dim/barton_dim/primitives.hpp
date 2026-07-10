#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "src/dim/barton_dim/material.hpp"
#include "src/dim/barton_dim/state.hpp"
#include "src/euler/eos.hpp"
#include "src/solid/elastoplastic/barton/state.hpp"

namespace dim::barton_dim {

inline std::array<double, 9> identity_stretch()
{
    return {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
}

inline std::array<double, 9> symmetric_unimodular_stretch(std::array<double, 9> stretch)
{
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 3; ++j) {
            const double value = 0.5 * (stretch[3 * i + j] + stretch[3 * j + i]);
            stretch[3 * i + j] = value;
            stretch[3 * j + i] = value;
        }
    }
    const double determinant = solid::barton::det3(stretch);
    if (!std::isfinite(determinant) || determinant <= 1.0e-14) return identity_stretch();
    const double scale = std::cbrt(determinant);
    for (double& value : stretch) value /= scale;
    return stretch;
}

inline std::array<double, 9> solid_deformation_from_stretch(
    const std::array<double, 9>& stretch,
    double rho_solid,
    const solid::barton::TensorMaterial& material)
{
    std::array<double, 9> deformation = symmetric_unimodular_stretch(stretch);
    const double volumetric_scale = std::cbrt(material.rho0 / std::max(rho_solid, 1.0e-12));
    for (double& value : deformation) value *= volumetric_scale;
    return deformation;
}

inline double mixture_internal_energy_density(
    double pressure,
    double alpha_solid,
    double rho_solid,
    double rho_fluid,
    const std::array<double, 9>& stretch,
    const Materials& materials)
{
    const double alpha_fluid = 1.0 - alpha_solid;
    const auto deformation = solid_deformation_from_stretch(stretch, rho_solid, materials.solid);
    const double temperature = solid::barton::material_temperature_from_rho_p(
        rho_solid, pressure, materials.solid);
    const double solid_energy = solid::barton::tensor_energy_from_F_T(
        deformation, temperature, materials.solid);
    const double fluid_energy = MaterialEOS::internal_energy(
        rho_fluid, pressure, materials.fluid);
    return alpha_solid * rho_solid * solid_energy +
        alpha_fluid * rho_fluid * fluid_energy;
}

template<int DIM>
inline Primitive<DIM> cons_to_prim(const State<DIM>& state, const Materials& materials)
{
    constexpr double floor = 1.0e-12;
    Primitive<DIM> primitive{};
    primitive.alpha_solid = std::clamp(state.alpha_solid, 0.0, 1.0);
    primitive.alpha_fluid = 1.0 - primitive.alpha_solid;
    primitive.rho_solid = state.solid_mass / std::max(primitive.alpha_solid, floor);
    primitive.rho_fluid = state.fluid_mass / std::max(primitive.alpha_fluid, floor);
    primitive.rho = std::max(state.solid_mass + state.fluid_mass, floor);
    double kinetic = 0.0;
    for (int d = 0; d < DIM; ++d) {
        primitive.velocity[d] = state.momentum[d] / primitive.rho;
        kinetic += state.momentum[d] * primitive.velocity[d];
    }
    const double target_internal = std::max(state.total_energy - 0.5 * kinetic, floor);
    if (primitive.alpha_solid <= floor) {
        primitive.pressure = MaterialEOS::pressure<1>(
            Conserved<1>{primitive.rho_fluid, {0.0}, target_internal}, materials.fluid);
    }
    else if (primitive.alpha_fluid <= floor) {
        const auto deformation = solid_deformation_from_stretch(
            state.elastic_stretch, primitive.rho_solid, materials.solid);
        const double temperature = solid::barton::material_temperature_from_rho_e(
            primitive.rho_solid, target_internal / primitive.rho_solid, materials.solid);
        primitive.pressure = solid::barton::material_pressure_from_rho_T(
            primitive.rho_solid, temperature, materials.solid);
        const auto sigma = solid::barton::tensor_stress_from_F_T(
            deformation, primitive.rho_solid, temperature, materials.solid);
        primitive.stress = sigma;
    }
    else {
        double lo = 1.0e-8;
        double hi = 1.0e12;
        for (int iteration = 0; iteration < 100; ++iteration) {
            const double mid = 0.5 * (lo + hi);
            if (mixture_internal_energy_density(mid, primitive.alpha_solid, primitive.rho_solid,
                    primitive.rho_fluid, state.elastic_stretch, materials) < target_internal) lo = mid;
            else hi = mid;
        }
        primitive.pressure = 0.5 * (lo + hi);
    }
    primitive.pressure = std::max(primitive.pressure, floor);

    if (primitive.alpha_solid > floor) {
        const auto deformation = solid_deformation_from_stretch(
            state.elastic_stretch, primitive.rho_solid, materials.solid);
        const double temperature = solid::barton::material_temperature_from_rho_p(
            primitive.rho_solid, primitive.pressure, materials.solid);
        const auto solid_stress = solid::barton::tensor_stress_from_F_T(
            deformation, primitive.rho_solid, temperature, materials.solid);
        const double mean = (solid_stress[0] + solid_stress[4] + solid_stress[8]) / 3.0;
        for (int q = 0; q < 9; ++q) primitive.stress[q] = primitive.alpha_solid * (solid_stress[q] - (q == 0 || q == 4 || q == 8 ? mean : 0.0));
    }
    primitive.stress[0] -= primitive.pressure;
    primitive.stress[4] -= primitive.pressure;
    primitive.stress[8] -= primitive.pressure;
    const double fluid_c = primitive.alpha_fluid > floor
        ? std::sqrt(std::max(materials.fluid.gamma * primitive.pressure / std::max(primitive.rho_fluid, floor), 0.0)) : 0.0;
    const double solid_c = std::sqrt(std::max(materials.solid.c0 * materials.solid.c0 +
        4.0 * materials.solid.b0 * materials.solid.b0 / 3.0, 0.0));
    primitive.wave_speed = std::sqrt(primitive.alpha_solid * solid_c * solid_c +
        primitive.alpha_fluid * fluid_c * fluid_c);
    primitive.p_solid = primitive.pressure;
    primitive.p_fluid = primitive.pressure;
    primitive.sigma_solid = primitive.stress;
    primitive.solid_wave_speed = solid_c;
    return primitive;
}

} // namespace dim::barton_dim
