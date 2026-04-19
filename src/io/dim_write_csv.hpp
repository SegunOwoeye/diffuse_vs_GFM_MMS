#pragma once

#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <stdexcept>

#include "src/dim/state.hpp"
#include "src/dim/primitives.hpp"
#include "src/dim/eos.hpp"
#include "src/dim/eos_params.hpp"


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


// [1] Write CSV (DIM)
template<int DIM, int NMAT, typename EOS>
inline void write_csv(
    const std::string& filename,
    const std::vector<Conserved<DIM, NMAT>>& U,
    const std::array<int, DIM>& N,
    const std::array<double, DIM>& domain_min,
    const std::array<double, DIM>& domain_max,
    const EOSParams<NMAT>& params
)
{
    std::ofstream file(filename);

    if (!file) {
        throw std::runtime_error("write_csv: failed to open file");
    }

    // [1.1] Compute dx
    std::array<double, DIM> dx{};
    for (int d = 0; d < DIM; ++d) {
        dx[d] = (domain_max[d] - domain_min[d]) / N[d];
    }

    // [1.2] Header
    for (int d = 0; d < DIM; ++d) {
        file << "x" << d << ",";
    }

    file << "rho,";

    for (int d = 0; d < DIM; ++d) {
        file << "u" << d << ",";
    }

    file << "p,e,";

    for (int k = 0; k < NMAT; ++k) {
        file << "alpha" << k;
        if (k < NMAT - 1) file << ",";
    }

    file << "\n";

    // [1.3] Loop cells
    std::array<int, DIM> idx{};

    const int total_cells = static_cast<int>(U.size());

    for (int linear = 0; linear < total_cells; ++linear) {

        // unflatten
        int tmp = linear;
        for (int d = DIM - 1; d >= 0; --d) {
            idx[d] = tmp % N[d];
            tmp /= N[d];
        }

        const auto x = compute_cell_center_csv<DIM>(idx, domain_min, dx);

        const auto& Ui = U[linear];

        // [1.3.1] Primitive
        const Primitive<DIM, NMAT> P =
            cons_to_prim<DIM, NMAT, EOS>(Ui, params);

        // [1.3.2] Total density
        double rho = 0.0;
        for (int k = 0; k < NMAT; ++k) {
            rho += Ui.arho[k];
        }

        // [1.3.3] Internal energy (mixture-consistent)
        const double gmix =
            EOS::template mixture_gamma<NMAT>(P.alpha, params);

        const double e =
            EOS::internal_energy(rho, P.p, gmix);

        // [1.4] Write row
        for (int d = 0; d < DIM; ++d) {
            file << x[d] << ",";
        }

        file << rho << ",";

        for (int d = 0; d < DIM; ++d) {
            file << P.vel[d] << ",";
        }

        file << P.p << ",";
        file << e << ",";

        for (int k = 0; k < NMAT; ++k) {
            file << P.alpha[k];
            if (k < NMAT - 1) file << ",";
        }

        file << "\n";
    }
}


