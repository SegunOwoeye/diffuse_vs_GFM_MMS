#pragma once

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/io/compute_exact_solution.hpp"


// [0] Write exact solution CSV
template<int DIM>
inline void write_exact_csv(
    const std::string& filename,
    const std::vector<ExactState<DIM>>& exact,
    const std::array<int, DIM>& N,
    const std::array<double, DIM>& domain_min,
    const std::array<double, DIM>& domain_max
)
{
    std::ofstream file(filename);

    if (!file) {
        throw std::runtime_error("write_exact_csv: failed to open file");
    }

    std::array<double, DIM> dx{};
    for (int d = 0; d < DIM; ++d) {
        dx[d] = (domain_max[d] - domain_min[d]) / static_cast<double>(N[d]);
    }

    // [0.1] Write header
    for (int d = 0; d < DIM; ++d) {
        file << "x" << d << ",";
    }

    file << "rho,";

    for (int d = 0; d < DIM; ++d) {
        file << "u" << d << ",";
    }

    file << "p,e,gamma,material_id\n";

    // [0.2] Loop over all cells
    std::array<int, DIM> idx{};

    const int total_cells = static_cast<int>(exact.size());

    for (int linear = 0; linear < total_cells; ++linear) {
        int tmp = linear;

        for (int d = DIM - 1; d >= 0; --d) {
            idx[d] = tmp % N[d];
            tmp /= N[d];
        }

        for (int d = 0; d < DIM; ++d) {
            const double x =
                domain_min[d] + (static_cast<double>(idx[d]) + 0.5) * dx[d];

            file << x << ",";
        }

        file << exact[linear].rho << ",";

        for (int d = 0; d < DIM; ++d) {
            file << exact[linear].vel[d] << ",";
        }


        file << exact[linear].p << ","
             << exact[linear].e << ","
             << exact[linear].gamma << ","
             << exact[linear].material_id << "\n";
    }
}


