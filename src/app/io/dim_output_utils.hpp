#pragma once

#include <array>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "src/dim/state.hpp"
#include "src/dim/eos_params.hpp"
#include "src/dim/conservative.hpp"
#include "src/io/config.hpp"


// [0] Ensure parent directory exists
inline void ensure_directory(
    const std::string& dir
)
{
    if (!dir.empty()) {
        std::filesystem::create_directories(dir);
    }
}


// [1] Build numerical solution output path
template<int DIM>
inline std::string build_solution_filename(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N
)
{
    std::string filename = cfg.output_prefix;

    if (!cfg.output_dir.empty()) {
        filename = cfg.output_dir + "/" + filename + "/" + filename;
    }

    for (int d = 0; d < DIM; ++d) {
        filename += "_N" + std::to_string(N[d]);
    }

    filename += ".csv";

    return filename;
}


// [2] Write numerical solution CSV (DIM)
template<int DIM, int NMAT>
inline void write_dim_output(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N,
    const std::vector<Conserved<DIM, NMAT>>& U,
    const EOSParams<NMAT>& params
)
{
    const std::string filename = build_solution_filename<DIM>(cfg, N);

    std::cout << "Writing to: " << filename << "\n";

    std::filesystem::path p(filename);
    ensure_directory(p.parent_path().string());

    std::ofstream file(filename);

    file << "x";
    if constexpr (DIM >= 2) file << ",y";
    if constexpr (DIM >= 3) file << ",z";

    file << ",rho";

    for (int d = 0; d < DIM; ++d) {
        file << ",u" << d;
    }

    file << ",p\n";

    const int total_cells = static_cast<int>(U.size());

    std::array<double, DIM> dx{};
    for (int d = 0; d < DIM; ++d) {
        dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / N[d];
    }

    std::array<int, DIM> idx{};

    for (int linear = 0; linear < total_cells; ++linear) {

        int tmp = linear;
        for (int d = DIM - 1; d >= 0; --d) {
            idx[d] = tmp % N[d];
            tmp /= N[d];
        }

        std::array<double, DIM> x{};
        for (int d = 0; d < DIM; ++d) {
            x[d] = cfg.domain_min[d] + (idx[d] + 0.5) * dx[d];
        }

        // --- USE SOLVER CONSISTENT PATH ---
        const Primitive<DIM, NMAT> P =
            cons_to_prim<DIM, NMAT>(U[linear], params);

        // total density (explicit, not from P)
        double rho = 0.0;
        for (int k = 0; k < NMAT; ++k) {
            rho += U[linear].arho[k];
        }

        // write coordinates
        file << x[0];
        if constexpr (DIM >= 2) file << "," << x[1];
        if constexpr (DIM >= 3) file << "," << x[2];

        // write state
        file << "," << rho;

        for (int d = 0; d < DIM; ++d) {
            file << "," << P.vel[d];
        }

        file << "," << P.p << "\n";
    }

    file.close();

    std::cout << "Written: " << filename << "\n";
}

