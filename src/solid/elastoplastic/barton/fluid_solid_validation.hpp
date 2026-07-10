#pragma once

// Focused checks for the first MGFM/rGFM-style fluid-solid interface bridge.
// These tests validate traction balance and normal-velocity handling before
// the interface state helper is connected to a full multimaterial time stepper.

#include <algorithm>
#include <cmath>
#include <iostream>

#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/primitives.hpp"
#include "src/sim/gfm/fluid_solid_rgfm.hpp"

namespace solid::barton {

inline bool fluid_solid_close_relative(double actual, double expected, double tolerance)
{
    const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    return std::abs(actual - expected) <= tolerance * scale;
}

inline Conserved<1> make_validation_fluid_state(
    double rho,
    double velocity,
    double pressure,
    const EOSParams& params)
{
    ::Primitive<1> P{};
    P.rho = rho;
    P.vel[0] = velocity;
    P.p = pressure;
    return ::prim_to_cons<1, MaterialEOS>(P, params);
}

inline State make_validation_solid_state(
    double velocity,
    double sigma11,
    const Material& mat)
{
    const State base = prim_to_cons(mat.rho0, velocity, mat);
    return barton_state_with_sigma11(base, mat, sigma11, velocity);
}

inline int validate_fluid_solid_rgfm_interface()
{
    EOSParams gas{};
    gas.kind = EOSKind::ideal_gas;
    gas.gamma = 1.4;

    Material solid{};
    const double rho_air = 1.225;
    const double p0 = 1.0e5;

    const Conserved<1> fluid_equilibrium =
        make_validation_fluid_state(rho_air, 0.0, p0, gas);
    const State solid_equilibrium =
        make_validation_solid_state(0.0, -p0, solid);

    bool ok = true;
    std::cout << "[SOLID][FSI] Fluid-solid rGFM interface validation\n";

    const FluidSolidRGFMState1D equilibrium =
        fluid_solid_rgfm_interface_states_1d(
            fluid_equilibrium,
            solid_equilibrium,
            gas,
            solid,
            1);
    const bool equilibrium_ok =
        fluid_solid_close_relative(equilibrium.interface_velocity, 0.0, 1.0e-10) &&
        fluid_solid_close_relative(equilibrium.interface_pressure, p0, 1.0e-10) &&
        fluid_solid_close_relative(equilibrium.interface_solid_stress, -p0, 1.0e-10);
    ok = ok && equilibrium_ok;
    std::cout << "  equilibrium: u*=" << equilibrium.interface_velocity
              << ", p*=" << equilibrium.interface_pressure
              << ", sigma11*=" << equilibrium.interface_solid_stress
              << (equilibrium_ok ? " [ok]\n" : " [FAIL]\n");

    const Conserved<1> impact_fluid =
        make_validation_fluid_state(rho_air, 100.0, p0, gas);
    const FluidSolidRGFMState1D impact =
        fluid_solid_rgfm_interface_states_1d(
            impact_fluid,
            solid_equilibrium,
            gas,
            solid,
            1);
    const bool impact_ok =
        impact.interface_velocity > 0.0 &&
        impact.interface_velocity < 100.0 &&
        impact.interface_pressure > p0 &&
        fluid_solid_close_relative(impact.interface_solid_stress, -impact.interface_pressure, 1.0e-12);
    ok = ok && impact_ok;
    std::cout << "  impact: u*=" << impact.interface_velocity
              << ", p*=" << impact.interface_pressure
              << ", sigma11*=" << impact.interface_solid_stress
              << (impact_ok ? " [ok]\n" : " [FAIL]\n");

    const Conserved<1> mirrored_fluid =
        make_validation_fluid_state(rho_air, -100.0, p0, gas);
    const FluidSolidRGFMState1D mirrored =
        fluid_solid_rgfm_interface_states_1d(
            mirrored_fluid,
            solid_equilibrium,
            gas,
            solid,
            -1);
    const bool mirror_ok =
        fluid_solid_close_relative(mirrored.interface_pressure, impact.interface_pressure, 1.0e-12) &&
        fluid_solid_close_relative(mirrored.interface_velocity, -impact.interface_velocity, 1.0e-12) &&
        fluid_solid_close_relative(mirrored.interface_solid_stress, impact.interface_solid_stress, 1.0e-12);
    ok = ok && mirror_ok;
    std::cout << "  mirrored normal: u*=" << mirrored.interface_velocity
              << ", p*=" << mirrored.interface_pressure
              << ", sigma11*=" << mirrored.interface_solid_stress
              << (mirror_ok ? " [ok]\n" : " [FAIL]\n");

    if (!ok) {
        std::cerr << "[SOLID][FSI] Fluid-solid rGFM interface validation failed\n";
        return 1;
    }
    std::cout << "[SOLID][FSI] Fluid-solid rGFM interface validation passed\n";
    return 0;
}

} // namespace solid::barton
