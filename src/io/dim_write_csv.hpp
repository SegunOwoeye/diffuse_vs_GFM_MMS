#pragma once

#include <fstream>
#include <algorithm>
#include <sstream>
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

        // [1.3] Format rows in bounded parallel batches, then write in order.
        const int total_cells = static_cast<int>(U.size());
        const int block_size = 4096;
        const int batch_blocks = 32;
        const int num_blocks = (total_cells + block_size - 1) / block_size;

        for (int block0 = 0; block0 < num_blocks; block0 += batch_blocks) {
            const int block_count = std::min(batch_blocks, num_blocks - block0);
            std::vector<std::string> blocks(block_count);

            #pragma omp parallel for schedule(static)
            for (int local_block = 0; local_block < block_count; ++local_block) {
                const int block = block0 + local_block;
                const int begin = block * block_size;
                const int end = std::min(total_cells, begin + block_size);
                std::ostringstream rows;

                for (int linear = begin; linear < end; ++linear) {
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

                    for (int d = 0; d < DIM; ++d) {
                        rows << x[d] << ",";
                    }

                    rows << rho << ",";
                    for (int d = 0; d < DIM; ++d) {
                        rows << P.vel[d] << ",";
                    }

                    rows << P.p << "," << internal_energy;

                    for (double value : P.alpha) {
                        rows << "," << value;
                    }

                    for (double value : P.rho) {
                        rows << "," << value;
                    }

                    for (double value : U[linear].partial_mass) {
                        rows << "," << value;
                    }

                    rows << "\n";
                }

                blocks[local_block] = rows.str();
            }

            for (const auto& block : blocks) {
                file << block;
            }
        }
    }

}

