#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "src/dim/barton_dim/config.hpp"
#include "src/dim/barton_dim/hllc.hpp"
#include "src/dim/barton_dim/relaxation.hpp"

namespace dim::barton_dim {

template<int DIM>
inline int total_cells(const std::array<int, DIM>& cells)
{
    int total = 1;
    for (int d = 0; d < DIM; ++d) total *= cells[d];
    return total;
}

template<int DIM>
inline int flatten(const std::array<int, DIM>& index, const std::array<int, DIM>& cells)
{
    int result = 0;
    int stride = 1;
    for (int d = 0; d < DIM; ++d) {
        result += index[d] * stride;
        stride *= cells[d];
    }
    return result;
}

template<int DIM>
inline std::array<int, DIM> unflatten(int linear, const std::array<int, DIM>& cells)
{
    std::array<int, DIM> index{};
    for (int d = 0; d < DIM; ++d) {
        index[d] = linear % cells[d];
        linear /= cells[d];
    }
    return index;
}

template<int DIM>
inline State<DIM> boundary_state(
    const State<DIM>& interior,
    int direction,
    const std::string& condition)
{
    State<DIM> ghost = interior;
    if (condition == "transmissive") return ghost;
    if (condition == "reflective") {
        ghost.momentum[direction] = -ghost.momentum[direction];
        return ghost;
    }
    throw std::runtime_error("Unsupported Barton-DIM boundary condition: " + condition);
}

template<int DIM>
inline double compute_dt_cfl(
    const std::vector<State<DIM>>& states,
    const Config<DIM>& config,
    double maximum_dt)
{
    if (states.empty()) throw std::runtime_error("Barton-DIM CFL calculation received an empty state vector");
    std::array<double, DIM> spacing{};
    for (int d = 0; d < DIM; ++d) {
        spacing[d] = (config.domain_max[d] - config.domain_min[d]) / config.cells[d];
    }
    double maximum_rate = 0.0;
    for (const State<DIM>& state : states) {
        for (int d = 0; d < DIM; ++d) {
            maximum_rate = std::max(maximum_rate, wave_speed(state, config.materials, d) / spacing[d]);
        }
    }
    if (!std::isfinite(maximum_rate) || maximum_rate <= 0.0) return maximum_dt;
    return std::min(maximum_dt, config.cfl / maximum_rate);
}

template<int DIM>
inline void apply_flux_difference(
    State<DIM>& state,
    const Flux<DIM>& left,
    const Flux<DIM>& right,
    double scale)
{
    state.solid_mass -= scale * (right.solid_mass - left.solid_mass);
    state.fluid_mass -= scale * (right.fluid_mass - left.fluid_mass);
    for (int d = 0; d < DIM; ++d) state.momentum[d] -= scale * (right.momentum[d] - left.momentum[d]);
    state.total_energy -= scale * (right.total_energy - left.total_energy);
    state.solid_energy -= scale * (right.solid_energy - left.solid_energy);
    state.fluid_energy -= scale * (right.fluid_energy - left.fluid_energy);
    for (int q = 0; q < 9; ++q) state.solid_rhoF[q] -= scale * (right.solid_rhoF[q] - left.solid_rhoF[q]);
    state.solid_rho_eqps -= scale * (right.solid_rho_eqps - left.solid_rho_eqps);
    state.solid_rho_damage -= scale * (right.solid_rho_damage - left.solid_rho_damage);
}

template<int DIM>
inline double alpha_flux(const State<DIM>& left, const State<DIM>& right, double contact_speed)
{
    return contact_speed >= 0.0
        ? contact_speed * left.alpha_solid
        : contact_speed * right.alpha_solid;
}

template<int DIM>
inline void sweep_direction(
    std::vector<State<DIM>>& states,
    const Config<DIM>& config,
    int direction,
    double dt)
{
    const std::array<int, DIM>& cells = config.cells;
    const double spacing = (config.domain_max[direction] - config.domain_min[direction]) / cells[direction];
    const std::vector<State<DIM>> old = states;
    std::vector<State<DIM>> next = old;

    const int cell_count = total_cells<DIM>(cells);
    #pragma omp parallel for if(cell_count > 256)
    for (int linear = 0; linear < cell_count; ++linear) {
        const auto index = unflatten<DIM>(linear, cells);
        auto left_index = index;
        auto right_index = index;
        const bool at_lower_boundary = index[direction] == 0;
        const bool at_upper_boundary = index[direction] == cells[direction] - 1;
        if (!at_lower_boundary) --left_index[direction];
        if (!at_upper_boundary) ++right_index[direction];
        const State<DIM> left_neighbour = at_lower_boundary
            ? boundary_state(old[linear], direction, config.bc_lo[direction])
            : old[flatten<DIM>(left_index, cells)];
        const State<DIM> right_neighbour = at_upper_boundary
            ? boundary_state(old[linear], direction, config.bc_hi[direction])
            : old[flatten<DIM>(right_index, cells)];

        const RiemannResult<DIM> left_face = hllc_flux(
            left_neighbour, old[linear], config.materials, direction);
        const RiemannResult<DIM> right_face = hllc_flux(
            old[linear], right_neighbour, config.materials, direction);
        apply_flux_difference(next[linear], left_face.flux, right_face.flux, dt / spacing);
        next[linear].alpha_solid = old[linear].alpha_solid - (dt / spacing) * (
            alpha_flux(old[linear], right_neighbour, right_face.contact_speed) -
            alpha_flux(left_neighbour, old[linear], left_face.contact_speed));

        const Primitive<DIM> primitive = cons_to_prim(old[linear], config.materials);
        for (int column = 0; column < 3; ++column) {
            const double left_rhoF = 0.5 * (
                left_neighbour.solid_rhoF[3 * direction + column] + old[linear].solid_rhoF[3 * direction + column]);
            const double right_rhoF = 0.5 * (
                old[linear].solid_rhoF[3 * direction + column] + right_neighbour.solid_rhoF[3 * direction + column]);
            const double compatibility_gradient = (right_rhoF - left_rhoF) / spacing;
            for (int row = 0; row < DIM; ++row) {
                next[linear].solid_rhoF[3 * row + column] -=
                    dt * primitive.velocity[row] * compatibility_gradient;
            }
        }
        repair_state(next[linear], config.materials);
    }
    states.swap(next);
}

template<int DIM>
inline double advance_one_timestep(
    std::vector<State<DIM>>& states,
    const Config<DIM>& config,
    double maximum_dt)
{
    const double dt = compute_dt_cfl(states, config, maximum_dt);
    for (int direction = 0; direction < DIM; ++direction) {
        sweep_direction(states, config, direction, dt);
    }
    for (State<DIM>& state : states) {
        apply_plastic_relaxation(state, config.materials, dt);
        reinitialise_phase_energies(state, config.materials);
    }
    return dt;
}

} // namespace dim::barton_dim
