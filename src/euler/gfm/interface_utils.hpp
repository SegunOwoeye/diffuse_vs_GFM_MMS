#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "src/euler/state.hpp"
#include "src/euler/primitives.hpp"
#include "src/euler/conservative.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/gfm/mcrs.hpp"
#include "src/euler/level_set/level_set_core.hpp"
#include "src/math/numerical_safety.hpp"
#include "src/math/vector_ops.hpp"


// [0] Check if a face crosses the interface via strict sign change
inline bool is_interface_face(
    double phiL,
    double phiR,
    double tol = 1e-12
)
{
    if (std::abs(phiL) < tol && std::abs(phiR) < tol) {
        return false;
    }

    return (phiL * phiR < 0.0);
}


// [1] Linearised interface position fraction
inline double interface_fraction(
    double phiL,
    double phiR
)
{
    const double denom = phiL - phiR;

    if (std::abs(denom) < 1e-14) {
        return 0.5;
    }

    return std::clamp(safe_div(phiL, denom, 1e-14), 0.0, 1.0);
}


// [2] Access precomputed interface normal by cell index
template<int DIM>
inline std::array<double, DIM> interface_normal(
    const std::vector<std::array<double, DIM>>& normals,
    int id
)
{
    if (id < 0 || id >= static_cast<int>(normals.size())) {
        throw std::runtime_error("interface_normal: invalid index");
    }

    return normals[id];
}


// [3] Distance from left cell centre to interface along face direction
inline double interface_distance(
    double phiL,
    double phiR,
    double dx
)
{
    return interface_fraction(phiL, phiR) * dx;
}

/*
[4] Extract interface normal speed from mixed-material Riemann solve

    - This is the speed that should move the level set in sharp GFM.
    - It is consistent with ghost_states_normal function in ghost.hpp.
*/
template<int DIM, typename EOSL, typename EOSR>
inline double interface_normal_speed_mcrs(
    const Conserved<DIM>& UL,
    const Conserved<DIM>& UR,
    const std::array<double, DIM>& n,
    const EOSParams& paramsL,
    const EOSParams& paramsR,
    double tol = 1e-8,
    int max_iter = 50
)
{
    const double nmag2 = dot<DIM>(n, n);

    if (nmag2 < tol * tol) {
        throw std::runtime_error("interface_normal_speed_mcrs: interface normal too small");
    }

    const std::array<double, DIM> nhat = normalize<DIM>(n, tol);

    const Primitive<DIM> PL = cons_to_prim<DIM, EOSL>(UL, paramsL);
    const Primitive<DIM> PR = cons_to_prim<DIM, EOSR>(UR, paramsR);

    if (!std::isfinite(PL.rho) || !std::isfinite(PL.p) || PL.rho <= 0.0 || PL.p <= 0.0) {
        throw std::runtime_error("interface_normal_speed_mcrs: invalid left primitive state");
    }

    if (!std::isfinite(PR.rho) || !std::isfinite(PR.p) || PR.rho <= 0.0 || PR.p <= 0.0) {
        throw std::runtime_error("interface_normal_speed_mcrs: invalid right primitive state");
    }

    Primitive<1> PL_n{};
    Primitive<1> PR_n{};

    PL_n.rho = PL.rho;
    PL_n.vel[0] = dot<DIM>(PL.vel, nhat);
    PL_n.p = PL.p;

    PR_n.rho = PR.rho;
    PR_n.vel[0] = dot<DIM>(PR.vel, nhat);
    PR_n.p = PR.p;

    MCRS1D solver(
        PL_n,
        PR_n,
        paramsL,
        paramsR,
        tol,
        max_iter
    );

    const MCRSResult1D result = solver.solve();

    if (!std::isfinite(result.u_star)) {
        throw std::runtime_error("interface_normal_speed_mcrs: non-finite u_star");
    }

    return result.u_star;
}
