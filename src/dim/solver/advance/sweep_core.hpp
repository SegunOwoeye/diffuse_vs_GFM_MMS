#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <array>
#include <string>
#include <vector>

#include "src/dim/primitives.hpp"
#include "src/dim/reconstruction/muscl.hpp"
#include "src/dim/riemann/hllc.hpp"
#include "src/dim/solver/advance/geometry.hpp"
#include "src/dim/solver/advance/line_ops.hpp"

namespace dim {
    // DIM sweep a split five-equation update
    template<int DIM>
    inline void solve_line(
        const std::vector<State<DIM>>& U_in,
        std::vector<State<DIM>>& U_out,
        double dx,
        double dt,
        const EOSParams& params,
        int dir,
        double alpha_source_floor,
        const std::string& lambda_model
    )
    {
        const int n = static_cast<int>(U_in.size());
        const int nmat = params.nmat();

        if (n == 0) {
            throw std::runtime_error("dim::solve_line: empty line");
        }

        if (n == 1) {
            U_out = U_in;
            return;
        }

        const double lambda = dt / dx;

        std::vector<Flux<DIM>> F(n + 1, make_flux<DIM>(nmat));
        std::vector<double> face_velocity(n + 1, 0.0);
        std::vector<Primitive<DIM>> primitive(n);
        std::vector<std::vector<double>> alpha_full(n);
        std::vector<State<DIM>> UL_face;
        std::vector<State<DIM>> UR_face;
        std::vector<Primitive<DIM>> PL_face;
        std::vector<Primitive<DIM>> PR_face;

        std::array<double, DIM> normal{};
        normal[dir] = 1.0;

        #pragma omp parallel for if(!omp_in_parallel() && n > 256)
        for (int i = 0; i < n; ++i) {
            primitive[i] = cons_to_prim<DIM>(U_in[i], params);
            alpha_full[i] = primitive[i].alpha;
        }

        reconstruct_line_interfaces<DIM>(
            U_in,
            params,
            dir,
            dt,
            dx,
            alpha_source_floor,
            lambda_model,
            UL_face,
            UR_face
        );

        PL_face.resize(n - 1);
        PR_face.resize(n - 1);
        #pragma omp parallel for if(!omp_in_parallel() && n > 256)
        for (int face = 0; face < n - 1; ++face) {
            PL_face[face] = cons_to_prim<DIM>(UL_face[face], params);
            PR_face[face] = cons_to_prim<DIM>(UR_face[face], params);
        }

        // MUSCL-Hancock predicted HLLC interface fluxes
        #pragma omp parallel for if(!omp_in_parallel() && n > 256)
        for (int i = 0; i < n - 1; ++i) {
            const RiemannResult<DIM> result = hllc_flux_normal<DIM>(
                UL_face[i],
                UR_face[i],
                params,
                normal,
                lambda_model
            );
            F[i + 1] = result.flux;
            face_velocity[i + 1] = result.face_velocity;
        }

        F[0] = F[1];
        F[n] = F[n - 1];
        face_velocity[0] = face_velocity[1];
        face_velocity[n] = face_velocity[n - 1];

        U_out = U_in;
        #pragma omp parallel for if(!omp_in_parallel() && n > 256)
        for (int i = 0; i < n; ++i) {
            U_out[i] = conservative_update<DIM>(U_in[i], F[i], F[i + 1], lambda);
            repair_state<DIM>(U_out[i], params);
        }

        // Non-conservative alpha update
        std::vector<std::vector<double>> alpha_flux(n + 1, std::vector<double>(nmat, 0.0));

        for (int k = 0; k < nmat; ++k) {
            alpha_flux[0][k] = alpha_full[0][k] * face_velocity[0];
            alpha_flux[n][k] = alpha_full[n - 1][k] * face_velocity[n];
        }

        #pragma omp parallel for if(!omp_in_parallel() && n > 256)
        for (int face = 1; face < n; ++face) {
            const int iface = face - 1;
            const std::vector<double>& alpha_upwind =
                (face_velocity[face] >= 0.0) ? PL_face[iface].alpha : PR_face[iface].alpha;

            for (int k = 0; k < nmat; ++k) {
                alpha_flux[face][k] = alpha_upwind[k] * face_velocity[face];
            }
        }

        #pragma omp parallel for if(!omp_in_parallel() && n > 256)
        for (int i = 0; i < n; ++i) {
            std::vector<double> alpha_new = alpha_full[i];
            const std::vector<double> rhs = alpha_rhs_coefficients<DIM>(
                primitive[i],
                params,
                1e-12,
                alpha_source_floor,
                lambda_model
            );
            const double div_u = (face_velocity[i + 1] - face_velocity[i]) / dx;

            for (int k = 0; k < nmat; ++k) {
                alpha_new[k] -= lambda * (alpha_flux[i + 1][k] - alpha_flux[i][k]);
                alpha_new[k] += dt * rhs[k] * div_u;
            }

            sanitise_alpha(alpha_new, 0.0);
            U_out[i].alpha = independent_alpha(alpha_new);
            repair_state<DIM>(U_out[i], params);
        }
    }

    template<int DIM>
    inline void sweep_direction_dispatch(
        int dir,
        const std::vector<State<DIM>>& U_in,
        const std::array<int, DIM>& N,
        const std::array<double, DIM>& dx,
        const EOSParams& params,
        double dt,
        double alpha_source_floor,
        const std::string& lambda_model,
        std::vector<State<DIM>>& U_out
    )
    {
        const auto stride = compute_strides<DIM>(N);
        const int total_lines = static_cast<int>(U_in.size()) / N[dir];

        #pragma omp parallel for
        for (int linear = 0; linear < total_lines; ++linear) {
            std::array<int, DIM> idx{};
            std::vector<State<DIM>> line_in;
            std::vector<State<DIM>> line_out;
            int tmp = linear;

            for (int d = DIM - 1; d >= 0; --d) {
                if (d == dir) {
                    continue;
                }

                idx[d] = tmp % N[d];
                tmp /= N[d];
            }

            extract_line<DIM>(U_in, N, stride, dir, idx, line_in);
            solve_line<DIM>(
                line_in,
                line_out,
                dx[dir],
                dt,
                params,
                dir,
                alpha_source_floor,
                lambda_model
            );
            write_line<DIM>(U_out, N, stride, dir, idx, line_out);
        }
    }

} 
