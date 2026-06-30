#pragma once

#include <algorithm>
#include <cmath>

#include "src/dim/barton_dim/flux.hpp"

namespace dim::barton_dim {

// The comparison path intentionally retains the HLLC contact construction used
// by rGFM. Barton (2019) supplies the diffuse constitutive closure, not a
// different Riemann-solver or reconstruction method.
template<int DIM>
struct RiemannResult { Flux<DIM> flux{}; double contact_speed = 0.0; };

template<int DIM>
inline Flux<DIM> state_difference(const State<DIM>& right, const State<DIM>& left)
{
    Flux<DIM> result{};
    result.solid_mass = right.solid_mass - left.solid_mass;
    result.fluid_mass = right.fluid_mass - left.fluid_mass;
    for (int d = 0; d < DIM; ++d) result.momentum[d] = right.momentum[d] - left.momentum[d];
    result.total_energy = right.total_energy - left.total_energy;
    for (int q = 0; q < 9; ++q) result.elastic_stretch[q] = right.elastic_stretch[q] - left.elastic_stretch[q];
    result.solid_rho_eqps = right.solid_rho_eqps - left.solid_rho_eqps;
    result.solid_rho_damage = right.solid_rho_damage - left.solid_rho_damage;
    return result;
}

template<int DIM>
inline State<DIM> hllc_star_state(const State<DIM>& state, const Primitive<DIM>& primitive,
                                  int normal, double signal_speed, double contact_speed,
                                  double pressure_star)
{
    const double un = primitive.velocity[normal];
    const double denominator = signal_speed - contact_speed;
    const double scale = std::abs(denominator) > 1.0e-12
        ? (signal_speed - un) / denominator : 1.0;
    State<DIM> star = state;
    star.solid_mass = std::max(scale * state.solid_mass, 0.0);
    star.fluid_mass = std::max(scale * state.fluid_mass, 0.0);
    const double rho_star = std::max(star.solid_mass + star.fluid_mass, 1.0e-12);
    for (int d = 0; d < DIM; ++d) {
        const double velocity = d == normal ? contact_speed : primitive.velocity[d];
        star.momentum[d] = rho_star * velocity;
    }
    const double pressure = normal_pressure(primitive, normal);
    star.total_energy = ((signal_speed - un) * state.total_energy - pressure * un +
        pressure_star * contact_speed) / denominator;
    star.solid_rho_eqps = scale * state.solid_rho_eqps;
    star.solid_rho_damage = scale * state.solid_rho_damage;
    return star;
}

template<int DIM>
inline RiemannResult<DIM> hllc_flux(const State<DIM>& left, const State<DIM>& right,
                                    const Materials& materials, int normal)
{
    const Primitive<DIM> l = cons_to_prim(left, materials);
    const Primitive<DIM> r = cons_to_prim(right, materials);
    const double un_l = l.velocity[normal];
    const double un_r = r.velocity[normal];
    const double s_l = std::min(un_l - l.wave_speed, un_r - r.wave_speed);
    const double s_r = std::max(un_l + l.wave_speed, un_r + r.wave_speed);
    const Flux<DIM> f_l = physical_flux(left, materials, normal);
    const Flux<DIM> f_r = physical_flux(right, materials, normal);
    if (s_l >= 0.0) return {f_l, un_l};
    if (s_r <= 0.0) return {f_r, un_r};
    const double p_l = normal_pressure(l, normal);
    const double p_r = normal_pressure(r, normal);
    const double denominator = l.rho * (s_l - un_l) - r.rho * (s_r - un_r);
    if (std::abs(denominator) <= 1.0e-12) return {0.5 * (f_l + f_r), 0.5 * (un_l + un_r)};
    const double s_star = (p_r - p_l + l.rho * un_l * (s_l - un_l) -
        r.rho * un_r * (s_r - un_r)) / denominator;
    const double p_star = p_l + l.rho * (s_l - un_l) * (s_star - un_l);
    if (s_star >= 0.0) {
        const State<DIM> star = hllc_star_state(left, l, normal, s_l, s_star, p_star);
        return {f_l + s_l * state_difference(star, left), s_star};
    }
    const State<DIM> star = hllc_star_state(right, r, normal, s_r, s_star, p_star);
    return {f_r + s_r * state_difference(star, right), s_star};
}

} // namespace dim::barton_dim
