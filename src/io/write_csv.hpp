#pragma once

#include <fstream>
#include <vector>
#include <array>
#include <string>
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


// [1] Write CSV
template<int DIM, typename EOS>
inline void write_csv(
    const std::string& filename,
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::array<int, DIM>& N,
    const std::array<double, DIM>& domain_min,
    const std::array<double, DIM>& domain_max,
    const std::vector<EOSParams>& material_params
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

    file << "p,e,mat\n";

    // loop over cells
    std::array<int, DIM> idx{};

    const int total_cells = U.size();

    for (int linear = 0; linear < total_cells; ++linear) {

        // unflatten
        int tmp = linear;
        for (int d = DIM - 1; d >= 0; --d) {
            idx[d] = tmp % N[d];
            tmp /= N[d];
        }

        // detail position
        const auto x = compute_cell_center_csv<DIM>(idx, domain_min, dx);

        // detail material
        const int mat = material_id.empty() ? 0 : material_id[linear];
        const EOSParams& params = material_params[mat];

        const Primitive<DIM> P = cons_to_prim<DIM, EOS>(U[linear], params);


        // write row
        for (int d = 0; d < DIM; ++d) {
            file << x[d] << ",";
        }

        file << P.rho << ",";

        for (int d = 0; d < DIM; ++d) {
            file << P.vel[d] << ",";
        }

        double e = EOS::internal_energy(
            P.rho,
            P.p,
            params
        );

        file << P.p << ",";
        file << e << ",";
        file << mat << "\n";
    }
}


