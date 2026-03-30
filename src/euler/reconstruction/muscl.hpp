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


// [3] Compute limited primitive slope from left, centre, right states
template<int DIM>
inline PrimitiveSlope<DIM> compute_limited_slope(
    const Primitive<DIM>& PL,
    const Primitive<DIM>& PC,
    const Primitive<DIM>& PR
)
{
    PrimitiveSlope<DIM> slope;

    slope.rho = mc_limiter(PC.rho - PL.rho, PR.rho - PC.rho);
    slope.p = mc_limiter(PC.p - PL.p, PR.p - PC.p);

    for (int d = 0; d < DIM; ++d) {
        slope.vel[d] = mc_limiter(
            PC.vel[d] - PL.vel[d],
            PR.vel[d] - PC.vel[d]
        );
    }

    return slope;
}


// [4] Reconstruct left and right primitive states inside one cell
template<int DIM, typename EOS>
inline void reconstruct_cell_faces_material_aware(
    const Conserved<DIM>& U_left,
    const Conserved<DIM>& U_centre,
    const Conserved<DIM>& U_right,
    const EOSParams& params_left,
    const EOSParams& params_centre,
    const EOSParams& params_right,
    bool allow_second_order,
    Primitive<DIM>& P_left_face,
    Primitive<DIM>& P_right_face,
    double rho_floor = 1e-10,
    double p_floor = 1e-10
)
{
    const Primitive<DIM> PC = cons_to_prim<DIM, EOS>(U_centre, params_centre);

    if (!allow_second_order) {
        P_left_face = PC;
        P_right_face = PC;

        enforce_positive_primitive(P_left_face, rho_floor, p_floor);
        enforce_positive_primitive(P_right_face, rho_floor, p_floor);
        return;
    }

    const Primitive<DIM> PL = cons_to_prim<DIM, EOS>(U_left, params_left);
    const Primitive<DIM> PR = cons_to_prim<DIM, EOS>(U_right, params_right);

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
    U_line has N cell-centred states.
    mat_line has N material ids, one per cell.
    UL_face and UR_face are returned with size N-1, one state per interface.
*/
template<int DIM, int DIR, typename EOS>
inline void reconstruct_line_interfaces_material_aware(
    const std::vector<Conserved<DIM>>& U_line,
    const std::vector<int>& mat_line,
    const std::vector<EOSParams>& material_params,
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
        throw std::runtime_error("reconstruct_line_interfaces_material_aware: need at least 3 cells");
    }

    if (static_cast<int>(mat_line.size()) != N) {
        throw std::runtime_error("reconstruct_line_interfaces_material_aware: mat_line size mismatch");
    }

    UL_face.assign(N - 1, Conserved<DIM>{});
    UR_face.assign(N - 1, Conserved<DIM>{});

    std::vector<Conserved<DIM>> U_half(N);
    U_half[0] = U_line[0];
    U_half[N - 1] = U_line[N - 1];

    // [6.1] Half-step predictor at cell centres
    for (int i = 1; i < N - 1; ++i) {
        const int mat_left = mat_line[i - 1];
        const int mat_centre = mat_line[i];
        const int mat_right = mat_line[i + 1];

        if (mat_left < 0 || mat_left >= static_cast<int>(material_params.size()) ||
            mat_centre < 0 || mat_centre >= static_cast<int>(material_params.size()) ||
            mat_right < 0 || mat_right >= static_cast<int>(material_params.size())) {
            throw std::runtime_error("reconstruct_line_interfaces_material_aware: invalid material id");
        }

        const EOSParams& params_left = material_params[mat_left];
        const EOSParams& params_centre = material_params[mat_centre];
        const EOSParams& params_right = material_params[mat_right];

        const bool allow_second_order =
            (mat_left == mat_centre) && (mat_centre == mat_right);

        Primitive<DIM> P_left_face{};
        Primitive<DIM> P_right_face{};

        reconstruct_cell_faces_material_aware<DIM, EOS>(
            U_line[i - 1],
            U_line[i],
            U_line[i + 1],
            params_left,
            params_centre,
            params_right,
            allow_second_order,
            P_left_face,
            P_right_face,
            rho_floor,
            p_floor
        );

        const Conserved<DIM> U_left_face =
            prim_to_cons<DIM, EOS>(P_left_face, params_centre);

        const Conserved<DIM> U_right_face =
            prim_to_cons<DIM, EOS>(P_right_face, params_centre);

        U_half[i] = hancock_predict<DIM, DIR, EOS>(
            U_line[i],
            U_left_face,
            U_right_face,
            dt,
            dx,
            params_centre
        );

        enforce_positive_conserved<DIM, EOS>(
            U_half[i],
            params_centre,
            rho_floor,
            p_floor
        );
    }

    // [6.2] Reconstruct predicted interface states
    for (int i = 0; i < N - 1; ++i) {
        if (i == 0 || i == N - 2) {
            UL_face[i] = U_half[i];
            UR_face[i] = U_half[i + 1];
            continue;
        }

        const int mat_im1 = mat_line[i - 1];
        const int mat_i = mat_line[i];
        const int mat_ip1 = mat_line[i + 1];
        const int mat_ip2 = mat_line[i + 2];

        if (mat_im1 < 0 || mat_im1 >= static_cast<int>(material_params.size()) ||
            mat_i < 0 || mat_i >= static_cast<int>(material_params.size()) ||
            mat_ip1 < 0 || mat_ip1 >= static_cast<int>(material_params.size()) ||
            mat_ip2 < 0 || mat_ip2 >= static_cast<int>(material_params.size())) {
            throw std::runtime_error("reconstruct_line_interfaces_material_aware: invalid material id");
        }

        const EOSParams& params_im1 = material_params[mat_im1];
        const EOSParams& params_i = material_params[mat_i];
        const EOSParams& params_ip1 = material_params[mat_ip1];
        const EOSParams& params_ip2 = material_params[mat_ip2];

        const bool allow_second_order_i =
            (mat_im1 == mat_i) && (mat_i == mat_ip1);

        const bool allow_second_order_ip1 =
            (mat_i == mat_ip1) && (mat_ip1 == mat_ip2);

        Primitive<DIM> P_left_i{};
        Primitive<DIM> P_right_i{};
        Primitive<DIM> P_left_ip1{};
        Primitive<DIM> P_right_ip1{};

        reconstruct_cell_faces_material_aware<DIM, EOS>(
            U_half[i - 1],
            U_half[i],
            U_half[i + 1],
            params_im1,
            params_i,
            params_ip1,
            allow_second_order_i,
            P_left_i,
            P_right_i,
            rho_floor,
            p_floor
        );

        reconstruct_cell_faces_material_aware<DIM, EOS>(
            U_half[i],
            U_half[i + 1],
            U_half[i + 2],
            params_i,
            params_ip1,
            params_ip2,
            allow_second_order_ip1,
            P_left_ip1,
            P_right_ip1,
            rho_floor,
            p_floor
        );

        UL_face[i] = prim_to_cons<DIM, EOS>(P_right_i, params_i);
        UR_face[i] = prim_to_cons<DIM, EOS>(P_left_ip1, params_ip1);
    }
}



