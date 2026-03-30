#pragma once

#include <algorithm>
#include <array>
#include <utility>

#include "src/euler/state.hpp"
#include "src/euler/primitives.hpp"
#include "src/euler/conservative.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/gfm/mcrs.hpp"

#include "src/math/vector_ops.hpp"


/*
[0] Pairwise Ghost Fluid construction using the true interface normal
This enforces:
    [0.1] Shared interface pressure p*
    [0.2] Shared interface-normal velocity u_n*
    [0.3] Preserved tangential velocity on each side
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
    // [0.4] Convert left and right states to primitive form
    const Primitive<DIM> PL = cons_to_prim<DIM, EOSL>(UL, paramsL);
    const Primitive<DIM> PR = cons_to_prim<DIM, EOSR>(UR, paramsR);

    // [0.5] Normalise interface normal
    const std::array<double, DIM> n = normalize<DIM>(normal, tol);

    // [0.6] Decompose velocity into normal and tangential components
    const double unL = dot<DIM>(PL.vel, n);
    const double unR = dot<DIM>(PR.vel, n);

    const std::array<double, DIM> uL_n = scale<DIM>(unL, n);
    const std::array<double, DIM> uR_n = scale<DIM>(unR, n);

    const std::array<double, DIM> uL_t = sub<DIM>(PL.vel, uL_n);
    const std::array<double, DIM> uR_t = sub<DIM>(PR.vel, uR_n);

    // [0.7] Build 1D normal states for mixed-material Riemann problem
    Primitive<1> PL_n{};
    Primitive<1> PR_n{};

    PL_n.rho = PL.rho;
    PL_n.vel[0] = unL;
    PL_n.p = PL.p;

    PR_n.rho = PR.rho;
    PR_n.vel[0] = unR;
    PR_n.p = PR.p;

    // [0.8] Solve mixed-material Riemann problem in normal direction
    MCRS1D solver(
        PL_n,
        PR_n,
        paramsL,
        paramsR,
        tol,
        max_iter
    );

    const MCRSResult1D result = solver.solve();

    const double p_star = std::max(result.p_star, tol);
    const double un_star = result.u_star;

    const double rhoL_star = std::max(result.rhoL_star, tol);
    const double rhoR_star = std::max(result.rhoR_star, tol);

    // [0.9] Reconstruct full velocity from normal + tangential
    const std::array<double, DIM> u_star_n = scale<DIM>(un_star, n);

    const std::array<double, DIM> uL_star = add<DIM>(uL_t, u_star_n);
    const std::array<double, DIM> uR_star = add<DIM>(uR_t, u_star_n);

    // [0.10] Construct ghost primitive states
    Primitive<DIM> PL_star{};
    Primitive<DIM> PR_star{};

    PL_star.rho = rhoL_star;
    PL_star.vel = uL_star;
    PL_star.p = p_star;

    PR_star.rho = rhoR_star;
    PR_star.vel = uR_star;
    PR_star.p = p_star;

    // [0.11] Convert back to conservative form using respective EOS
    const Conserved<DIM> UL_star = prim_to_cons<DIM, EOSL>(PL_star, paramsL);
    const Conserved<DIM> UR_star = prim_to_cons<DIM, EOSR>(PR_star, paramsR);

    return {UL_star, UR_star};
}


/*
[1] Axis-Aligned Wrapper (for debugging / 1D / grid-aligned interfaces)
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

