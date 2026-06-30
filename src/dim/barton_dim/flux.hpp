#pragma once

#include "src/dim/barton_dim/primitives.hpp"

namespace dim::barton_dim {

template<int DIM>
inline double normal_pressure(const Primitive<DIM>& primitive, int normal)
{
    return -primitive.stress[3 * normal + normal];
}

template<int DIM>
struct Flux {
    double solid_mass = 0.0;
    double fluid_mass = 0.0;
    std::array<double, DIM> momentum{};
    double total_energy = 0.0;
    double solid_energy = 0.0;
    double fluid_energy = 0.0;
    std::array<double, 9> solid_rhoF{};
    std::array<double, 9> elastic_stretch{};
    double solid_rho_eqps = 0.0;
    double solid_rho_damage = 0.0;
};

template<int DIM>
inline Flux<DIM> operator+(const Flux<DIM>& left, const Flux<DIM>& right)
{
    Flux<DIM> result{};
    result.solid_mass = left.solid_mass + right.solid_mass;
    result.fluid_mass = left.fluid_mass + right.fluid_mass;
    for (int d = 0; d < DIM; ++d) result.momentum[d] = left.momentum[d] + right.momentum[d];
    result.total_energy = left.total_energy + right.total_energy;
    for (int q = 0; q < 9; ++q) result.elastic_stretch[q] = left.elastic_stretch[q] + right.elastic_stretch[q];
    result.solid_rho_eqps = left.solid_rho_eqps + right.solid_rho_eqps;
    result.solid_rho_damage = left.solid_rho_damage + right.solid_rho_damage;
    return result;
}

template<int DIM>
inline Flux<DIM> operator*(double scale, const Flux<DIM>& flux)
{
    Flux<DIM> result{};
    result.solid_mass = scale * flux.solid_mass;
    result.fluid_mass = scale * flux.fluid_mass;
    for (int d = 0; d < DIM; ++d) result.momentum[d] = scale * flux.momentum[d];
    result.total_energy = scale * flux.total_energy;
    for (int q = 0; q < 9; ++q) result.elastic_stretch[q] = scale * flux.elastic_stretch[q];
    result.solid_rho_eqps = scale * flux.solid_rho_eqps;
    result.solid_rho_damage = scale * flux.solid_rho_damage;
    return result;
}

template<int DIM>
inline Flux<DIM> physical_flux(const State<DIM>& state, const Materials& materials, int normal)
{
    const Primitive<DIM> primitive = cons_to_prim(state, materials);
    const double un = primitive.velocity[normal];
    Flux<DIM> flux{};
    flux.solid_mass = state.solid_mass * un;
    flux.fluid_mass = state.fluid_mass * un;
    for (int d = 0; d < DIM; ++d) flux.momentum[d] = state.momentum[d] * un - primitive.stress[3 * d + normal];
    double stress_power = 0.0;
    for (int d = 0; d < DIM; ++d) stress_power += primitive.velocity[d] * primitive.stress[3 * d + normal];
    flux.total_energy = state.total_energy * un - stress_power;
    for (int row = 0; row < 3; ++row) {
        const double velocity = row < DIM ? primitive.velocity[row] : 0.0;
        for (int column = 0; column < 3; ++column) {
            flux.elastic_stretch[3 * row + column] =
                state.elastic_stretch[3 * row + column] * un -
                state.elastic_stretch[3 * normal + column] * velocity;
        }
    }
    flux.solid_rho_eqps = state.solid_rho_eqps * un;
    flux.solid_rho_damage = state.solid_rho_damage * un;
    return flux;
}

template<int DIM>
inline double wave_speed(const State<DIM>& state, const Materials& materials, int normal)
{
    const Primitive<DIM> primitive = cons_to_prim(state, materials);
    return std::abs(primitive.velocity[normal]) + primitive.wave_speed;
}

} // namespace dim::barton_dim
