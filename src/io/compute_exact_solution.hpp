#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "src/io/config.hpp"
#include "src/euler/state.hpp"
#include "src/euler/primitives.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/riemann/exact_riemann.hpp"


// [0] Exact state container
template<int DIM>
struct ExactState {
    double rho = 0.0;
    std::array<double, DIM> vel{};
    double p = 0.0;
    double e = 0.0;
    double gamma = 0.0;
    int material_id = -1;
};


// [1] Compute 1D cell centre
inline double compute_cell_center_1d(
    int i,
    double xL,
    double dx
)
{
    return xL + (static_cast<double>(i) + 0.5) * dx;
}


// [2] Build EOS params from material config
inline EOSParams build_exact_material_params(
    const MaterialConfig& mat
)
{
    EOSParams params{};

    if (mat.type == "ideal_gas" || mat.type == "stiffened_gas") {
        auto it = mat.params.find("gamma");

        if (it == mat.params.end()) {
            throw std::runtime_error("compute_exact_solution: missing gamma in material");
        }

        params.kind = eos_kind_from_string(mat.type);
        params.gamma = it->second;

        if (mat.type == "stiffened_gas") {
            auto p_inf = mat.params.find("p_inf");
            params.p_inf = (p_inf == mat.params.end()) ? 0.0 : p_inf->second;
        }

        return params;
    }

    throw std::runtime_error("compute_exact_solution: unsupported EOS type for exact solution");
}


// [3] Build primitive state from region
inline Primitive<1> primitive_from_region_1d(
    const Region<1>& region
)
{
    Primitive<1> P{};

    P.rho = region.rho;
    P.vel[0] = region.vel[0];
    P.p = region.p;

    return P;
}


// [4] Compute exact solution on 1D grid
template<typename EOS>
inline void compute_exact_solution(
    std::vector<ExactState<1>>& exact,
    const Config<1>& cfg,
    const std::array<int, 1>& N,
    double time
)
{
    if (cfg.regions.size() != 2) {
        throw std::runtime_error("compute_exact_solution: exact 1D Riemann solver requires exactly 2 regions");
    }

    if (cfg.materials.empty()) {
        throw std::runtime_error("compute_exact_solution: no materials defined");
    }

    const Region<1>& left_region = cfg.regions[0];
    const Region<1>& right_region = cfg.regions[1];

    const int matL = left_region.material_id;
    const int matR = right_region.material_id;

    if (matL < 0 || matL >= static_cast<int>(cfg.materials.size()) ||
        matR < 0 || matR >= static_cast<int>(cfg.materials.size())) {
        throw std::runtime_error("compute_exact_solution: invalid material id");
    }

    const MaterialConfig& materialL = cfg.materials[matL];
    const MaterialConfig& materialR = cfg.materials[matR];

    const EOSParams paramsL = build_exact_material_params(materialL);
    const EOSParams paramsR = build_exact_material_params(materialR);

    if (!eos_has_stiffened_gas_wave_curve(paramsL) ||
        !eos_has_stiffened_gas_wave_curve(paramsR)) {
        throw std::runtime_error("compute_exact_solution: unsupported EOS for exact wave curves");
    }

    const Primitive<1> PL = primitive_from_region_1d(left_region);
    const Primitive<1> PR = primitive_from_region_1d(right_region);

    const Conserved<1> UL = prim_to_cons<1, EOS>(PL, paramsL);
    const Conserved<1> UR = prim_to_cons<1, EOS>(PR, paramsR);

    ExactRiemannSolver1D<EOS> solver(paramsL, paramsR);

    const ExactStarState1D star = solver.star_solver(UL, UR);

    const int Nx = N[0];
    const double xL = cfg.domain_min[0];
    const double xR = cfg.domain_max[0];
    const double dx = (xR - xL) / static_cast<double>(Nx);

    const double x0 = left_region.upper[0];

    exact.resize(Nx);

    for (int i = 0; i < Nx; ++i) {
        const double x = compute_cell_center_1d(i, xL, dx);

        if (time <= 0.0) {
            exact[i].rho = (x < x0) ? PL.rho : PR.rho;
            exact[i].vel[0] = (x < x0) ? PL.vel[0] : PR.vel[0];
            exact[i].p = (x < x0) ? PL.p : PR.p;
            exact[i].gamma = (x < x0) ? paramsL.gamma : paramsR.gamma;
            exact[i].material_id = (x < x0) ? matL : matR;

            const double rho = exact[i].rho;
            const double p = exact[i].p;
            const EOSParams& params = (x < x0) ? paramsL : paramsR;

            exact[i].e = EOS::internal_energy(rho, p, params);
        }
        else {
            const double xi = (x - x0) / time;

            const ExactSampleState1D S =
                solver.sample_state(UL, UR, star, xi);

            exact[i].rho = S.primitive.rho;
            exact[i].vel[0] = S.primitive.vel[0];
            exact[i].p = S.primitive.p;
            exact[i].e = S.e;
            exact[i].gamma = S.gamma;
            exact[i].material_id = (S.material_id == 0) ? matL : matR;
        }
    }
}

