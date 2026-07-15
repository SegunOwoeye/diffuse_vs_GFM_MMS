#pragma once

#include <fstream>
#include <algorithm>
#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <stdexcept>

#include "src/euler/state.hpp"
#include "src/euler/primitives.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
//#include "scr/euler/conservative.hpp"


// [0] Compute cell centre
template<int DIM>
inline std::array<double, DIM> compute_cell_center_csv(
    const std::array<int, DIM>& idx,
    const std::array<double, DIM>& domain_min,
    const std::array<double, DIM>& dx
)
{
    std::array<double, DIM> x{};

    for (int d = 0; d < DIM; ++d) {
        x[d] = domain_min[d] + (idx[d] + 0.5) * dx[d];
    }

    return x;
}

template<int DIM>
inline std::array<int, DIM> unflatten_raw_index_csv(
    int linear,
    const std::array<int, DIM>& N
)
{
    std::array<int, DIM> idx{};
    int stride = 1;

    for (int d = 0; d < DIM; ++d) {
        idx[d] = (linear / stride) % N[d];
        stride *= N[d];
    }

    return idx;
}


// [1] Write CSV
template<int DIM, typename EOS>
inline void write_csv(
    const std::string& filename,
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::array<int, DIM>& N,
    const std::array<double, DIM>& domain_min,
    const std::array<double, DIM>& domain_max,
    const std::vector<EOSParams>& material_params,
    const std::vector<std::vector<double>>* phi_list = nullptr
)
{
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("write_csv: failed to open file");
    }

    // compute dx
    std::array<double, DIM> dx{};
    for (int d = 0; d < DIM; ++d) {
        dx[d] = (domain_max[d] - domain_min[d]) / N[d];
    }

    // list headers
    for (int d = 0; d < DIM; ++d) {
        file << "x" << d << ",";
    }

    file << "rho,";

    for (int d = 0; d < DIM; ++d) {
        file << "u" << d << ",";
    }

    file << "p,e,mat";

    if (phi_list != nullptr) {
        for (int k = 0; k < static_cast<int>(phi_list->size()); ++k) {
            file << ",phi" << k;
        }
    }

    file << "\n";

    const int total_cells = U.size();

    if (phi_list != nullptr) {
        for (const auto& phi : *phi_list) {
            if (static_cast<int>(phi.size()) != total_cells) {
                throw std::runtime_error("write_csv: phi field size mismatch");
            }
        }
    }

    // [2] Format rows in bounded batches, then write in order.
    const int block_size = 4096;
    const int batch_blocks = 32;
    const int num_blocks = (total_cells + block_size - 1) / block_size;

    for (int block0 = 0; block0 < num_blocks; block0 += batch_blocks) {
        const int block_count = std::min(batch_blocks, num_blocks - block0);
        std::vector<std::string> blocks(block_count);

        for (int local_block = 0; local_block < block_count; ++local_block) {
            const int block = block0 + local_block;
            const int begin = block * block_size;
            const int end = std::min(total_cells, begin + block_size);
            std::ostringstream rows;

            for (int linear = begin; linear < end; ++linear) {
                const auto idx = unflatten_raw_index_csv<DIM>(linear, N);

                const auto x = compute_cell_center_csv<DIM>(idx, domain_min, dx);

                const int mat = material_id.empty() ? 0 : material_id[linear];
                const EOSParams& params = material_params[mat];

                const Primitive<DIM> P = cons_to_prim<DIM, EOS>(U[linear], params);

                for (int d = 0; d < DIM; ++d) {
                    rows << x[d] << ",";
                }

                rows << P.rho << ",";

                for (int d = 0; d < DIM; ++d) {
                    rows << P.vel[d] << ",";
                }

                double e = EOS::internal_energy(
                    P.rho,
                    P.p,
                    params
                );

                rows << P.p << ",";
                rows << e << ",";
                rows << mat;

                if (phi_list != nullptr) {
                    for (const auto& phi : *phi_list) {
                        rows << "," << phi[linear];
                    }
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
