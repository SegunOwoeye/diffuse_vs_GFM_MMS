#pragma once

#include <array>
#include <stdexcept>
#include <utility>

#include "src/euler/state.hpp"
#include "src/euler/primitives.hpp"
#include "src/euler/conservative.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/gfm/mcrs.hpp"
#include "src/math/numerical_safety.hpp"
#include "src/math/vector_ops.hpp"


template<int DIM>
struct RGFMInterfaceStates {
    Conserved<DIM> left{};
    Conserved<DIM> right{};
    double p_star = 0.0;
    double un_star = 0.0;
};


/*
[0] rGFM interface-state construction using the interface normal.

    Returns:
        left-> state used to redefine the real left node and left-material ghosts
        right -> state used to redefine the real right node and right-material ghosts

    Shared interface conditions:
        [0.1] pressure = p*
        [0.2] normal velocity = u_n*

    Side-specific quantities preserved:
        [0.3] tangential velocity
        [0.4] density branch from the mixed-material Riemann solve
*/
template<int DIM, typename EOSL, typename EOSR>
inline RGFMInterfaceStates<DIM> rgfm_interface_states_normal(
    const Conserved<DIM>& UL,
    const Conserved<DIM>& UR,
    const std::array<double, DIM>& normal,
    const EOSParams& paramsL,
    const EOSParams& paramsR,
    double tol = 1e-8,
    int max_iter = 50
)
{
    // [0.1] Validate EOS params
    if (paramsL.gamma <= 1.0) {
        throw std::runtime_error("rgfm_interface_states_normal: left gamma must be > 1");
    }

    if (paramsR.gamma <= 1.0) {
        throw std::runtime_error("rgfm_interface_states_normal: right gamma must be > 1");
    }

    // [0.2] Convert real states to primitive form
    const Primitive<DIM> PL = cons_to_prim<DIM, EOSL>(UL, paramsL);
    const Primitive<DIM> PR = cons_to_prim<DIM, EOSR>(UR, paramsR);

    if (PL.rho <= 0.0 || !std::isfinite(PL.rho)) {
        throw std::runtime_error("rgfm_interface_states_normal: invalid left density");
    }

    if (PL.p <= 0.0 || !std::isfinite(PL.p)) {
        throw std::runtime_error("rgfm_interface_states_normal: invalid left pressure");
    }

    if (PR.rho <= 0.0 || !std::isfinite(PR.rho)) {
        throw std::runtime_error("rgfm_interface_states_normal: invalid right density");
    }

    if (PR.p <= 0.0 || !std::isfinite(PR.p)) {
        throw std::runtime_error("rgfm_interface_states_normal: invalid right pressure");
    }

    // [0.3] Validate and normalise interface normal
    const double normal_mag2 = dot<DIM>(normal, normal);

    if (normal_mag2 < tol * tol) {
        throw std::runtime_error("rgfm_interface_states_normal: interface normal too small");
    }

    const std::array<double, DIM> n = normalize<DIM>(normal, tol);

    // [0.4] Decompose real velocities into normal / tangential parts
    const double unL = dot<DIM>(PL.vel, n);
    const double unR = dot<DIM>(PR.vel, n);

    const std::array<double, DIM> uL_t = sub<DIM>(PL.vel, scale<DIM>(unL, n));
    const std::array<double, DIM> uR_t = sub<DIM>(PR.vel, scale<DIM>(unR, n));

    // [0.5] Build 1D normal-direction primitive states for MCRS
    Primitive<1> PL_n{};
    Primitive<1> PR_n{};

    PL_n.rho = PL.rho;
    PL_n.vel[0] = unL;
    PL_n.p = PL.p;

    PR_n.rho = PR.rho;
    PR_n.vel[0] = unR;
    PR_n.p = PR.p;

    // [0.6] Solve mixed-material normal Riemann problem
    MCRS1D solver(
        PL_n,
        PR_n,
        paramsL,
        paramsR,
        tol,
        max_iter
    );

    const MCRSResult1D result = solver.solve();

    const double p_star = require_positive(result.p_star, "rgfm_interface_states_normal: p_star", tol);
    const double un_star = require_finite(result.u_star, "rgfm_interface_states_normal: u_star");

    // [0.7] Shared normal velocity at interface
    const std::array<double, DIM> u_star_n = scale<DIM>(un_star, n);

    /*
    [0.8] Real-GFM states

        The real-GFM correction uses the Riemann densities rho_IL/rho_IR
        directly when redefining both the real interface-adjacent nodes and
        ghost nodes.
    */

    Primitive<DIM> state_left{};
    Primitive<DIM> state_right{};

    const double rhoL_star = require_positive(
        result.rhoL_star,
        "rgfm_interface_states_normal: rhoL_star",
        tol
    );

    const double rhoR_star = require_positive(
        result.rhoR_star,
        "rgfm_interface_states_normal: rhoR_star",
        tol
    );

    state_left.rho = rhoL_star;
    state_left.vel = add<DIM>(uL_t, u_star_n);
    state_left.p = p_star;

    state_right.rho = rhoR_star;
    state_right.vel = add<DIM>(uR_t, u_star_n);
    state_right.p = p_star;

    const Conserved<DIM> U_left = prim_to_cons<DIM, EOSL>(state_left, paramsL);
    const Conserved<DIM> U_right = prim_to_cons<DIM, EOSR>(state_right, paramsR);

    return {U_left, U_right, p_star, un_star};
}


template<int DIM, typename EOSL, typename EOSR>
inline std::pair<Conserved<DIM>, Conserved<DIM>> ghost_states_normal(
    const Conserved<DIM>& UL,
    const Conserved<DIM>& UR,
    const std::array<double, DIM>& normal,
    const EOSParams& paramsL,
    const EOSParams& paramsR,
    double tol = 1e-8,
    int max_iter = 50
)
{
    const auto states = rgfm_interface_states_normal<DIM, EOSL, EOSR>(
        UL,
        UR,
        normal,
        paramsL,
        paramsR,
        tol,
        max_iter
    );

    return {states.left, states.right};
}


/*
[1] Axis-aligned wrapper for 1D tests and grid-aligned interfaces
*/
template<int DIM, int DIR, typename EOSL, typename EOSR>
inline std::pair<Conserved<DIM>, Conserved<DIM>> ghost_states_pair(
    const Conserved<DIM>& UL,
    const Conserved<DIM>& UR,
    const EOSParams& paramsL,
    const EOSParams& paramsR,
    double tol = 1e-8,
    int max_iter = 50
)
{
    static_assert(DIR >= 0 && DIR < DIM, "ghost_states_pair: invalid direction");

    std::array<double, DIM> normal{};
    normal[DIR] = 1.0;

    return ghost_states_normal<DIM, EOSL, EOSR>(
        UL,
        UR,
        normal,
        paramsL,
        paramsR,
        tol,
        max_iter
    );
}


