#pragma once

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#include "src/euler/state.hpp"
#include "src/euler/primitives.hpp"
#include "src/euler/conservative.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/flux.hpp"
#include "src/euler/reconstruction/limiter.hpp"


// USE NUMERICAL SAFTEY

// [0] Primitive slope container
template<int DIM>
struct PrimitiveSlope {
    double rho = 0.0;
    std::array<double, DIM> vel{};
    double p = 0.0;
};


// [1] Apply positivity floor to a primitive state
template<int DIM>
inline void enforce_positive_primitive(
    Primitive<DIM>& P,
    double rho_floor = 1e-10,
    double p_floor = 1e-10
)
{
    P.rho = std::max(P.rho, rho_floor);
    P.p = std::max(P.p, p_floor);
}


// [2] Apply positivity floor to a conservative state via primitive projection
template<int DIM, typename EOS>
inline void enforce_positive_conserved(
    Conserved<DIM>& U,
    const EOSParams& params,
    double rho_floor = 1e-10,
    double p_floor = 1e-10
)
{
    Primitive<DIM> P = cons_to_prim<DIM, EOS>(U, params);
    enforce_positive_primitive(P, rho_floor, p_floor);
    U = prim_to_cons<DIM, EOS>(P, params);
}


// [3] Compute TVD limited primitive slope from left, centre, right states
template<int DIM>
inline PrimitiveSlope<DIM> compute_limited_slope(
    const Primitive<DIM>& PL,
    const Primitive<DIM>& PC,
    const Primitive<DIM>& PR
)
{
    PrimitiveSlope<DIM> slope;

    slope.rho = minmod(PC.rho - PL.rho, PR.rho - PC.rho);
    slope.p = minmod(PC.p - PL.p, PR.p - PC.p);

    for (int d = 0; d < DIM; ++d) {
        slope.vel[d] = minmod(
            PC.vel[d] - PL.vel[d],
            PR.vel[d] - PC.vel[d]
        );
    }

    return slope;
}


// [4] Reconstruct left and right primitive states inside one cell
template<int DIM>
inline void reconstruct_cell_faces(
    const Primitive<DIM>& PL,
    const Primitive<DIM>& PC,
    const Primitive<DIM>& PR,
    Primitive<DIM>& P_left_face,
    Primitive<DIM>& P_right_face,
    double rho_floor = 1e-10,
    double p_floor = 1e-10
)
{
    const PrimitiveSlope<DIM> slope = compute_limited_slope<DIM>(PL, PC, PR);

    P_left_face.rho = PC.rho - 0.5 * slope.rho;
    P_right_face.rho = PC.rho + 0.5 * slope.rho;

    P_left_face.p = PC.p - 0.5 * slope.p;
    P_right_face.p = PC.p + 0.5 * slope.p;

    for (int d = 0; d < DIM; ++d) {
        P_left_face.vel[d] = PC.vel[d] - 0.5 * slope.vel[d];
        P_right_face.vel[d] = PC.vel[d] + 0.5 * slope.vel[d];
    }

    enforce_positive_primitive(P_left_face, rho_floor, p_floor);
    enforce_positive_primitive(P_right_face, rho_floor, p_floor);
}


// [5] MUSCL-Hancock half-step predictor for one sweep direction
template<int DIM, int DIR, typename EOS>
inline Conserved<DIM> hancock_predict(
    const Conserved<DIM>& U_centre,
    const Conserved<DIM>& U_left_face,
    const Conserved<DIM>& U_right_face,
    double dt,
    double dx,
    const EOSParams& params_centre
)
{
    std::array<double, DIM> normal{};
    normal[DIR] = 1.0;

    const Conserved<DIM> F_left = compute_flux_normal<DIM, EOS>(
        U_left_face,
        normal,
        params_centre
    );

    const Conserved<DIM> F_right = compute_flux_normal<DIM, EOS>(
        U_right_face,
        normal,
        params_centre
    );

    const double factor = 0.5 * dt / dx;

    return U_centre - factor * (F_right - F_left);
}


/*
[6] Reconstruct interface states for a 1D line in direction DIR
    U_line has N cell-centred conservative states.
    cell_params has N EOS parameter objects, one per cell.
    UL_face and UR_face are returned with size N-1, one state per interface.

    This is a pure MUSCL-Hancock reconstruction. 
*/
template<int DIM, int DIR, typename EOS>
inline void reconstruct_line_interfaces(
    const std::vector<Conserved<DIM>>& U_line,
    const std::vector<EOSParams>& cell_params,
    double dt,
    double dx,
    std::vector<Conserved<DIM>>& UL_face,
    std::vector<Conserved<DIM>>& UR_face,
    double rho_floor = 1e-10,
    double p_floor = 1e-10
)
{
    const int N = static_cast<int>(U_line.size());

    if (N < 3) {
        throw std::runtime_error("reconstruct_line_interfaces: need at least 3 cells");
    }

    if (static_cast<int>(cell_params.size()) != N) {
        throw std::runtime_error("reconstruct_line_interfaces: cell_params size mismatch");
    }

    UL_face.assign(N - 1, Conserved<DIM>{});
    UR_face.assign(N - 1, Conserved<DIM>{});

    // [6.1] Convert full line to primitive variables
    std::vector<Primitive<DIM>> P_line(N);

    for (int i = 0; i < N; ++i) {
        P_line[i] = cons_to_prim<DIM, EOS>(U_line[i], cell_params[i]);
        enforce_positive_primitive(P_line[i], rho_floor, p_floor);
    }

    // [6.2] Reconstruct piecewise linear states at time level n
    std::vector<Conserved<DIM>> U_left_n(N);
    std::vector<Conserved<DIM>> U_right_n(N);

    U_left_n[0] = U_line[0];
    U_right_n[0] = U_line[0];
    U_left_n[N - 1] = U_line[N - 1];
    U_right_n[N - 1] = U_line[N - 1];

    for (int i = 1; i < N - 1; ++i) {
        Primitive<DIM> P_left_face{};
        Primitive<DIM> P_right_face{};

        reconstruct_cell_faces<DIM>(
            P_line[i - 1],
            P_line[i],
            P_line[i + 1],
            P_left_face,
            P_right_face,
            rho_floor,
            p_floor
        );

        U_left_n[i] = prim_to_cons<DIM, EOS>(P_left_face, cell_params[i]);
        U_right_n[i] = prim_to_cons<DIM, EOS>(P_right_face, cell_params[i]);

        enforce_positive_conserved<DIM, EOS>(
            U_left_n[i],
            cell_params[i],
            rho_floor,
            p_floor
        );

        enforce_positive_conserved<DIM, EOS>(
            U_right_n[i],
            cell_params[i],
            rho_floor,
            p_floor
        );
    }

    // [6.3] Hancock half-step predictor at cell centres
    std::vector<Conserved<DIM>> U_half(N);

    U_half[0] = U_line[0];
    U_half[N - 1] = U_line[N - 1];

    for (int i = 1; i < N - 1; ++i) {
        U_half[i] = hancock_predict<DIM, DIR, EOS>(
            U_line[i],
            U_left_n[i],
            U_right_n[i],
            dt,
            dx,
            cell_params[i]
        );

        enforce_positive_conserved<DIM, EOS>(
            U_half[i],
            cell_params[i],
            rho_floor,
            p_floor
        );
    }

    // [6.4] Convert predicted cell-centred states to primitives
    std::vector<Primitive<DIM>> P_half(N);

    for (int i = 0; i < N; ++i) {
        P_half[i] = cons_to_prim<DIM, EOS>(U_half[i], cell_params[i]);
        enforce_positive_primitive(P_half[i], rho_floor, p_floor);
    }

    // [6.5] Reconstruct piecewise linear states from predicted values
    std::vector<Conserved<DIM>> U_left_half(N);
    std::vector<Conserved<DIM>> U_right_half(N);

    U_left_half[0] = U_half[0];
    U_right_half[0] = U_half[0];
    U_left_half[N - 1] = U_half[N - 1];
    U_right_half[N - 1] = U_half[N - 1];

    for (int i = 1; i < N - 1; ++i) {
        Primitive<DIM> P_left_face{};
        Primitive<DIM> P_right_face{};

        reconstruct_cell_faces<DIM>(
            P_half[i - 1],
            P_half[i],
            P_half[i + 1],
            P_left_face,
            P_right_face,
            rho_floor,
            p_floor
        );

        U_left_half[i] = prim_to_cons<DIM, EOS>(P_left_face, cell_params[i]);
        U_right_half[i] = prim_to_cons<DIM, EOS>(P_right_face, cell_params[i]);

        enforce_positive_conserved<DIM, EOS>(
            U_left_half[i],
            cell_params[i],
            rho_floor,
            p_floor
        );

        enforce_positive_conserved<DIM, EOS>(
            U_right_half[i],
            cell_params[i],
            rho_floor,
            p_floor
        );
    }

    // [6.6] Assemble face states
    for (int i = 0; i < N - 1; ++i) {
        UL_face[i] = U_right_half[i];
        UR_face[i] = U_left_half[i + 1];
    }
}


