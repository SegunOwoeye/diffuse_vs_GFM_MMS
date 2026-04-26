#pragma once

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#include "src/dim/conservative.hpp"
#include "src/dim/flux.hpp"
#include "src/dim/primitives.hpp"
#include "src/euler/reconstruction/limiter.hpp"

namespace dim {

    // [1] Primitive slope container for DIM variables
    template<int DIM>
    struct PrimitiveSlope {
        std::vector<double> rho{};
        std::vector<double> alpha{};
        std::array<double, DIM> vel{};
        double p = 0.0;
    };

    // [2] Enforce primitive admissibility after reconstruction
    template<int DIM>
    inline void enforce_positive_primitive(
        Primitive<DIM>& P,
        double rho_floor = 1e-12,
        double p_floor = 1e-12,
        double alpha_floor = 0.0
    )
    {
        for (double& rho_k : P.rho) {
            rho_k = std::max(rho_k, rho_floor);
        }

        sanitise_alpha(P.alpha, alpha_floor);
        P.p = std::max(P.p, p_floor);
    }

    // [3] Limited primitive slope from left/centre/right cell primitives
    template<int DIM>
    inline PrimitiveSlope<DIM> compute_limited_slope(
        const Primitive<DIM>& PL,
        const Primitive<DIM>& PC,
        const Primitive<DIM>& PR
    )
    {
        if (PL.rho.size() != PC.rho.size() || PC.rho.size() != PR.rho.size() ||
            PL.alpha.size() != PC.alpha.size() || PC.alpha.size() != PR.alpha.size()) {
            throw std::runtime_error("dim::compute_limited_slope: primitive size mismatch");
        }

        PrimitiveSlope<DIM> slope{};
        slope.rho.assign(PC.rho.size(), 0.0);
        slope.alpha.assign(PC.alpha.size(), 0.0);

        for (int k = 0; k < static_cast<int>(PC.rho.size()); ++k) {
            slope.rho[k] = minmod(PC.rho[k] - PL.rho[k], PR.rho[k] - PC.rho[k]);
        }

        for (int k = 0; k < static_cast<int>(PC.alpha.size()); ++k) {
            slope.alpha[k] = minmod(PC.alpha[k] - PL.alpha[k], PR.alpha[k] - PC.alpha[k]);
        }

        for (int d = 0; d < DIM; ++d) {
            slope.vel[d] = minmod(PC.vel[d] - PL.vel[d], PR.vel[d] - PC.vel[d]);
        }

        slope.p = minmod(PC.p - PL.p, PR.p - PC.p);
        return slope;
    }

    // [4] Reconstruct one cell's left/right primitive face states
    template<int DIM>
    inline void reconstruct_cell_faces(
        const Primitive<DIM>& PL,
        const Primitive<DIM>& PC,
        const Primitive<DIM>& PR,
        Primitive<DIM>& P_left_face,
        Primitive<DIM>& P_right_face,
        double rho_floor = 1e-12,
        double p_floor = 1e-12
    )
    {
        const PrimitiveSlope<DIM> slope = compute_limited_slope<DIM>(PL, PC, PR);

        P_left_face = PC;
        P_right_face = PC;

        for (int k = 0; k < static_cast<int>(PC.rho.size()); ++k) {
            P_left_face.rho[k] = PC.rho[k] - 0.5 * slope.rho[k];
            P_right_face.rho[k] = PC.rho[k] + 0.5 * slope.rho[k];
        }

        for (int k = 0; k < static_cast<int>(PC.alpha.size()); ++k) {
            P_left_face.alpha[k] = PC.alpha[k] - 0.5 * slope.alpha[k];
            P_right_face.alpha[k] = PC.alpha[k] + 0.5 * slope.alpha[k];
        }

        for (int d = 0; d < DIM; ++d) {
            P_left_face.vel[d] = PC.vel[d] - 0.5 * slope.vel[d];
            P_right_face.vel[d] = PC.vel[d] + 0.5 * slope.vel[d];
        }

        P_left_face.p = PC.p - 0.5 * slope.p;
        P_right_face.p = PC.p + 0.5 * slope.p;

        enforce_positive_primitive(P_left_face, rho_floor, p_floor);
        enforce_positive_primitive(P_right_face, rho_floor, p_floor);
    }

    // [5] Apply a flux difference to the conservative DIM variables
    template<int DIM>
    inline State<DIM> apply_flux_difference(
        const State<DIM>& U,
        const Flux<DIM>& F_left,
        const Flux<DIM>& F_right,
        double factor
    )
    {
        State<DIM> result = U;

        if (F_left.partial_mass.size() != F_right.partial_mass.size() ||
            F_left.partial_mass.size() != U.partial_mass.size()) {
            throw std::runtime_error("dim::apply_flux_difference: partial_mass size mismatch");
        }

        for (int k = 0; k < static_cast<int>(U.partial_mass.size()); ++k) {
            result.partial_mass[k] -= factor * (F_right.partial_mass[k] - F_left.partial_mass[k]);
        }

        for (int d = 0; d < DIM; ++d) {
            result.mom[d] -= factor * (F_right.mom[d] - F_left.mom[d]);
        }

        result.E -= factor * (F_right.E - F_left.E);
        return result;
    }

    // [6] MUSCL-Hancock half-step predictor for one split sweep direction
    template<int DIM>
    inline State<DIM> hancock_predict(
        const State<DIM>& U_center,
        const Primitive<DIM>& P_center,
        const State<DIM>& U_left_face,
        const State<DIM>& U_right_face,
        const Primitive<DIM>& P_left_face,
        const Primitive<DIM>& P_right_face,
        const EOSParams& params,
        int dir,
        double dt,
        double dx
    )
    {
        std::array<double, DIM> normal{};
        normal[dir] = 1.0;

        const Flux<DIM> F_left = compute_flux_normal<DIM>(U_left_face, P_left_face, normal);
        const Flux<DIM> F_right = compute_flux_normal<DIM>(U_right_face, P_right_face, normal);
        const double factor = 0.5 * dt / dx;

        State<DIM> predicted = apply_flux_difference<DIM>(U_center, F_left, F_right, factor);

        std::vector<double> alpha_left = P_left_face.alpha;
        std::vector<double> alpha_right = P_right_face.alpha;
        const double u_left = P_left_face.vel[dir];
        const double u_right = P_right_face.vel[dir];
        const double div_u = (u_right - u_left) / dx;
        const std::vector<double> rhs = alpha_rhs_coefficients<DIM>(P_center, params);

        std::vector<double> alpha_new = P_center.alpha;
        for (int k = 0; k < params.nmat(); ++k) {
            alpha_new[k] -= factor * (alpha_right[k] * u_right - alpha_left[k] * u_left);
            alpha_new[k] += 0.5 * dt * rhs[k] * div_u;
        }

        sanitise_alpha(alpha_new, 0.0);
        predicted.alpha = independent_alpha(alpha_new);
        repair_state<DIM>(predicted, params);
        return predicted;
    }

    /*
    [7] Reconstruct left/right states for every interface in a split line.
        UL_face[i] and UR_face[i] are the time-centred states on interface i+1/2.
    */
    template<int DIM>
    inline void reconstruct_line_interfaces(
        const std::vector<State<DIM>>& U_line,
        const EOSParams& params,
        int dir,
        double dt,
        double dx,
        std::vector<State<DIM>>& UL_face,
        std::vector<State<DIM>>& UR_face
    )
    {
        const int n = static_cast<int>(U_line.size());
        const int nmat = params.nmat();

        if (n < 2) {
            throw std::runtime_error("dim::reconstruct_line_interfaces: need at least two cells");
        }

        UL_face.assign(n-1, make_state<DIM>(nmat));
        UR_face.assign(n-1, make_state<DIM>(nmat));

        if (n < 3) {
            for (int face = 0; face < n-1; ++face) {
                UL_face[face] = U_line[face];
                UR_face[face] = U_line[face + 1];
            }
            return;
        }

        std::vector<Primitive<DIM>> P_line(n);
        for (int i = 0; i < n; ++i) {
            P_line[i] = cons_to_prim<DIM>(U_line[i], params);
            enforce_positive_primitive(P_line[i]);
        }

        std::vector<Primitive<DIM>> P_left(n);
        std::vector<Primitive<DIM>> P_right(n);
        std::vector<State<DIM>> U_left(n);
        std::vector<State<DIM>> U_right(n);

        P_left[0] = P_line[0];
        P_right[0] = P_line[0];
        U_left[0] = U_line[0];
        U_right[0] = U_line[0];

        P_left[n-1] = P_line[n-1];
        P_right[n-1] = P_line[n-1];
        U_left[n-1] = U_line[n-1];
        U_right[n-1] = U_line[n-1];

        for (int i = 1; i < n-1; ++i) {
            reconstruct_cell_faces<DIM>(
                P_line[i-1],
                P_line[i],
                P_line[i+1],
                P_left[i],
                P_right[i]
            );

            U_left[i] = prim_to_cons<DIM>(P_left[i], params);
            U_right[i] = prim_to_cons<DIM>(P_right[i], params);
            repair_state<DIM>(U_left[i], params);
            repair_state<DIM>(U_right[i], params);
        }

        std::vector<State<DIM>> U_half(n);
        U_half[0] = U_line[0];
        U_half[n-1] = U_line[n-1];

        for (int i = 1; i < n-1; ++i) {
            U_half[i] = hancock_predict<DIM>(
                U_line[i],
                P_line[i],
                U_left[i],
                U_right[i],
                P_left[i],
                P_right[i],
                params,
                dir,
                dt,
                dx
            );
        }

        std::vector<Primitive<DIM>> P_half(n);
        for (int i = 0; i < n; ++i) {
            P_half[i] = cons_to_prim<DIM>(U_half[i], params);
            enforce_positive_primitive(P_half[i]);
        }

        std::vector<State<DIM>> U_left_half(n);
        std::vector<State<DIM>> U_right_half(n);
        U_left_half[0] = U_half[0];
        U_right_half[0] = U_half[0];
        U_left_half[n-1] = U_half[n-1];
        U_right_half[n-1] = U_half[n-1];

        for (int i = 1; i < n-1; ++i) {
            Primitive<DIM> P_left_half{};
            Primitive<DIM> P_right_half{};

            reconstruct_cell_faces<DIM>(
                P_half[i-1],
                P_half[i],
                P_half[i+1],
                P_left_half,
                P_right_half
            );

            U_left_half[i] = prim_to_cons<DIM>(P_left_half, params);
            U_right_half[i] = prim_to_cons<DIM>(P_right_half, params);
            repair_state<DIM>(U_left_half[i], params);
            repair_state<DIM>(U_right_half[i], params);
        }

        for (int face = 0; face < n-1; ++face) {
            UL_face[face] = U_right_half[face];
            UR_face[face] = U_left_half[face + 1];
        }
    }

}
