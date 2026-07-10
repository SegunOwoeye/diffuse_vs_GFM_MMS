#pragma once

// MGFM/rGFM-style interface states for a compressible fluid coupled to the
// Barton elastoplastic solid. This is the first reusable bridge: it solves a
// one-dimensional normal interface problem and constructs ghost states for each
// side without changing the existing fluid-fluid rGFM framework.

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "src/euler/conservative.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/primitives.hpp"
#include "src/solid/elastoplastic/barton/flux.hpp"

struct FluidSolidRGFMState1D {
    Conserved<1> fluid_interface{};
    solid::barton::State solid_interface{};
    Conserved<1> fluid_ghost{};
    solid::barton::State solid_ghost{};
    double interface_velocity = 0.0;
    double interface_pressure = 0.0;
    double interface_solid_stress = 0.0;
    double fluid_impedance = 0.0;
    double solid_impedance = 0.0;
};

inline solid::barton::State barton_state_with_sigma11(
    const solid::barton::State& state,
    const solid::barton::Material& mat,
    double target_sigma11,
    double target_velocity
)
{
    const solid::barton::Primitive P = solid::barton::cons_to_prim(state, mat);
    const double J = solid::barton::clamp_min(P.F11 * P.F22 * P.F33, 1.0e-18);
    const double J13 = std::pow(J, 1.0 / 3.0);

    auto candidate = [&](double q) {
        return solid::barton::make_state_from_deformation(
            P.rho,
            target_velocity,
            J13 * std::exp(q),
            J13 * std::exp(-0.5 * q),
            J13 * std::exp(-0.5 * q),
            state.rhoPlastic,
            mat);
    };

    auto residual = [&](double q) {
        return solid::barton::cons_to_prim(candidate(q), mat).sigma11 - target_sigma11;
    };

    double lo = -4.0;
    double hi = 4.0;
    double rlo = residual(lo);
    double rhi = residual(hi);

    for (int expand = 0; rlo * rhi > 0.0 && expand < 4; ++expand) {
        lo *= 1.5;
        hi *= 1.5;
        rlo = residual(lo);
        rhi = residual(hi);
    }

    if (rlo * rhi > 0.0 || !std::isfinite(rlo) || !std::isfinite(rhi)) {
        solid::barton::State fallback = solid::barton::make_state_from_deformation(
            P.rho,
            target_velocity,
            P.F11,
            P.F22,
            P.F33,
            state.rhoPlastic,
            mat);
        solid::barton::enforce(fallback, mat);
        return fallback;
    }

    for (int iter = 0; iter < 48; ++iter) {
        const double mid = 0.5 * (lo + hi);
        const double rm = residual(mid);
        if (rlo * rm <= 0.0) {
            hi = mid;
            rhi = rm;
        }
        else {
            lo = mid;
            rlo = rm;
        }
    }
    (void)rhi;
    return candidate(0.5 * (lo + hi));
}

template<typename FluidEOS = MaterialEOS>
inline Conserved<1> fluid_state_from_pressure_velocity_invariant(
    const Conserved<1>& state,
    const EOSParams& params,
    double pressure,
    double velocity
)
{
    const Primitive<1> P = cons_to_prim<1, FluidEOS>(state, params);
    const double bounded_pressure = std::max(pressure, 1.0e-12);
    const double invariant =
        FluidEOS::entropy_invariant(P.rho, P.p, params);
    const double rho =
        FluidEOS::density_from_p_invariant(bounded_pressure, invariant, params);

    Primitive<1> out{};
    out.rho = rho;
    out.vel[0] = velocity;
    out.p = bounded_pressure;
    return prim_to_cons<1, FluidEOS>(out, params);
}

/*
    Solve a linearised normal fluid-solid interface problem.

    Sign convention:
      - `normal_sign` maps global +x velocity into the interface normal pointing
        from fluid to solid. Use +1 when the fluid is on the left and solid is
        on the right, -1 for the opposite arrangement.
      - fluid pressure is positive in compression; solid sigma11 is positive in
        tension, so traction balance is sigma11* = -p*.
*/
template<typename FluidEOS = MaterialEOS>
inline FluidSolidRGFMState1D fluid_solid_rgfm_interface_states_1d(
    const Conserved<1>& fluid_state,
    const solid::barton::State& solid_state,
    const EOSParams& fluid_params,
    const solid::barton::Material& solid_mat,
    int normal_sign = 1
)
{
    if (normal_sign != 1 && normal_sign != -1) {
        throw std::runtime_error("fluid_solid_rgfm_interface_states_1d: normal_sign must be +/-1");
    }

    const Primitive<1> Pf = cons_to_prim<1, FluidEOS>(fluid_state, fluid_params);
    const solid::barton::Primitive Ps =
        solid::barton::cons_to_prim(solid_state, solid_mat);

    const double cf = FluidEOS::template sound_speed<1>(fluid_state, fluid_params);
    const double Zf = std::max(Pf.rho * cf, 1.0e-12);
    const double Zs = std::max(Ps.rho * Ps.wave_speed, 1.0e-12);

    const double uf_n = static_cast<double>(normal_sign) * Pf.vel[0];
    const double us_n = static_cast<double>(normal_sign) * Ps.u;
    const double p0 = std::max(Pf.p, 1.0e-12);
    const double sigma0 = Ps.sigma11;

    const double un_star =
        (p0 + sigma0 + Zf * uf_n + Zs * us_n) / (Zf + Zs);
    const double p_star = std::max(p0 + Zf * (uf_n - un_star), 1.0e-12);
    const double sigma_star = -p_star;
    const double u_star = static_cast<double>(normal_sign) * un_star;

    FluidSolidRGFMState1D out{};
    out.interface_velocity = u_star;
    out.interface_pressure = p_star;
    out.interface_solid_stress = sigma_star;
    out.fluid_impedance = Zf;
    out.solid_impedance = Zs;
    out.fluid_interface =
        fluid_state_from_pressure_velocity_invariant<FluidEOS>(
            fluid_state,
            fluid_params,
            p_star,
            u_star);
    out.fluid_ghost = out.fluid_interface;
    out.solid_interface =
        barton_state_with_sigma11(solid_state, solid_mat, sigma_star, u_star);
    out.solid_ghost = out.solid_interface;

    return out;
}
