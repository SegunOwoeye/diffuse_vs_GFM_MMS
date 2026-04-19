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


/*
[0] Ghost Fluid Method ghost-state construction using the interface normal.

    Returns:
        first -> ghost state to be used on the RIGHT side of the LEFT-material solve
        second -> ghost state to be used on the LEFT side of the RIGHT-material solve

    Shared interface conditions:
        [0.1] pressure = p*
        [0.2] normal velocity = u_n*

    Side-specific quantities preserved:
        [0.3] tangential velocity
        [0.4] density branch from that side's thermodynamic relation
*/
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
    // [0.1] Validate EOS params
    if (paramsL.gamma <= 1.0) {
        throw std::runtime_error("ghost_states_normal: left gamma must be > 1");
    }

    if (paramsR.gamma <= 1.0) {
        throw std::runtime_error("ghost_states_normal: right gamma must be > 1");
    }

    // [0.2] Convert real states to primitive form
    const Primitive<DIM> PL = cons_to_prim<DIM, EOSL>(UL, paramsL);
    const Primitive<DIM> PR = cons_to_prim<DIM, EOSR>(UR, paramsR);

    if (PL.rho <= 0.0 || !std::isfinite(PL.rho)) {
        throw std::runtime_error("ghost_states_normal: invalid left density");
    }

    if (PL.p <= 0.0 || !std::isfinite(PL.p)) {
        throw std::runtime_error("ghost_states_normal: invalid left pressure");
    }

    if (PR.rho <= 0.0 || !std::isfinite(PR.rho)) {
        throw std::runtime_error("ghost_states_normal: invalid right density");
    }

    if (PR.p <= 0.0 || !std::isfinite(PR.p)) {
        throw std::runtime_error("ghost_states_normal: invalid right pressure");
    }

    // [0.3] Validate and normalise interface normal
    const double normal_mag2 = dot<DIM>(normal, normal);

    if (normal_mag2 < tol * tol) {
        throw std::runtime_error("ghost_states_normal: interface normal too small");
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

    const double p_star = require_positive(result.p_star, "ghost_states_normal: p_star", tol);
    const double un_star = require_finite(result.u_star, "ghost_states_normal: u_star");

    // [0.7] Shared normal velocity at interface
    const std::array<double, DIM> u_star_n = scale<DIM>(un_star, n);

    /*
    [0.8] ghost states
    
        ghost_for_left:
            used as the right state in a left-material solve
            preserves LEFT tangential structure and LEFT density branch
    
        ghost_for_right:
            used as the left state in a right-material solve
            preserves RIGHT tangential structure and right density branch
    */

    // [0.8] True ghost states
    Primitive<DIM> ghost_for_left{};
    Primitive<DIM> ghost_for_right{};

    // [0.8.1] Compute entropy invariants
    const double KL = EOSL::entropy_invariant(PL.rho, PL.p, paramsL);
    const double KR = EOSR::entropy_invariant(PR.rho, PR.p, paramsR);

    // [0.8.2] Recover densities from p*
    const double rhoL_star = EOSL::density_from_p_invariant(p_star, KL, paramsL);
    const double rhoR_star = EOSR::density_from_p_invariant(p_star, KR, paramsR);

    // [0.8.3] Fill ghost states
    ghost_for_left.rho = rhoL_star;
    ghost_for_left.vel = add<DIM>(uL_t, u_star_n);
    ghost_for_left.p = p_star;

    ghost_for_right.rho = rhoR_star;
    ghost_for_right.vel = add<DIM>(uR_t, u_star_n);
    ghost_for_right.p = p_star;

    const Conserved<DIM> UG_left = prim_to_cons<DIM, EOSL>(ghost_for_left, paramsL);
    const Conserved<DIM> UG_right = prim_to_cons<DIM, EOSR>(ghost_for_right, paramsR);

    return {UG_left, UG_right};
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



