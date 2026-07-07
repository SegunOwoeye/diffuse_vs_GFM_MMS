#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "src/dim/primitives.hpp"
#include "src/dim/solver/advance/geometry.hpp"

namespace dim {

    // [1] Write CSV (DIM)
    template<int DIM>
    inline void write_csv(
        const std::string& filename,
        const std::vector<State<DIM>>& U,
        const std::array<int, DIM>& N,
        const std::array<double, DIM>& domain_min,
        const std::array<double, DIM>& domain_max,
        const EOSParams& params
    )
    {
        std::ofstream file(filename);
        if (!file) {
            throw std::runtime_error("dim::write_csv: failed to open file");
        }
        
        // [1.1] Compute dx
        std::array<double, DIM> dx{};
        for (int d = 0; d < DIM; ++d) {
            dx[d] = (domain_max[d] - domain_min[d]) / static_cast<double>(N[d]);
        }

        // [1.2] Header
        for (int d = 0; d < DIM; ++d) {
            file << "x" << d << ",";
        }

        file << "rho,";
        for (int d = 0; d < DIM; ++d) {
            file << "u" << d << ",";
        }

        file << "p,e";

        for (int k = 0; k < params.nmat(); ++k) {
            file << ",alpha" << k;
        }

        for (int k = 0; k < params.nmat(); ++k) {
            file << ",rho_mat" << k;
        }

        for (int k = 0; k < params.nmat(); ++k) {
            file << ",mass" << k;
        }

        file << "\n";

        // [1.3.1] Primitive
        for (int linear = 0; linear < static_cast<int>(U.size()); ++linear) {
            const auto idx = unflatten_index<DIM>(linear, N);
            const auto x = compute_cell_center<DIM>(idx, domain_min, dx);
            const Primitive<DIM> P = cons_to_prim<DIM>(U[linear], params);
            const double rho = total_density(U[linear]);
            double velocity_squared = 0.0;

            for (int d = 0; d < DIM; ++d) {
                velocity_squared += P.vel[d] * P.vel[d];
            }

            const double kinetic = 0.5 * rho * velocity_squared;
            const double internal_energy = std::max(safe_div(U[linear].E - kinetic, rho), 0.0);
            
            // [1.4] Write row
            for (int d = 0; d < DIM; ++d) {
                file << x[d] << ",";
            }

            file << rho << ",";
            for (int d = 0; d < DIM; ++d) {
                file << P.vel[d] << ",";
            }

            file << P.p << "," << internal_energy;

            for (double value : P.alpha) {
                file << "," << value;
            }

            for (double value : P.rho) {
                file << "," << value;
            }

            for (double value : U[linear].partial_mass) {
                file << "," << value;
            }

            file << "\n";
        }
    }

} 

