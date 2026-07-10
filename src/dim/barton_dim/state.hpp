#pragma once

#include <array>

namespace dim::barton_dim {

// Barton (2019), Eq. (31), specialised to one solid and one fluid component.
// The stretch is a shared unimodular elastic stretch, not a phase deformation
// gradient. Phase energies are deliberately absent: pressure follows from the
// mechanically equilibrated mixture equation of state.
template<int DIM>
struct State {
    double alpha_solid = 0.0;
    double solid_mass = 0.0;
    double fluid_mass = 0.0;
    std::array<double, DIM> momentum{};
    double total_energy = 0.0;
    double solid_energy = 0.0;
    double fluid_energy = 0.0;
    std::array<double, 9> solid_rhoF{};
    std::array<double, 9> elastic_stretch{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double solid_rho_eqps = 0.0;
    double solid_rho_damage = 0.0;
};

template<int DIM>
struct Primitive {
    double alpha_solid = 0.0;
    double alpha_fluid = 1.0;
    double rho_solid = 0.0;
    double rho_fluid = 0.0;
    double rho = 0.0;
    std::array<double, DIM> velocity{};
    double pressure = 0.0;
    std::array<double, 9> stress{};
    double wave_speed = 0.0;
    double p_solid = 0.0;
    double p_fluid = 0.0;
    std::array<double, 9> sigma_solid{};
    double solid_wave_speed = 0.0;
};

} // namespace dim::barton_dim
