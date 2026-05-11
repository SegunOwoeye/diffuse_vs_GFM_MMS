#pragma once

#include <array>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "src/euler/state.hpp"
#include "src/euler/eos_params.hpp"
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


// [2] Build exact solution output path for 1D
inline std::string build_exact_filename(
    const std::string& output_dir,
    const std::string& output_prefix,
    int N_exact
)
{
    std::string exact_file = output_prefix + "_exact";

    if (!output_dir.empty()) {
        exact_file = output_dir + "/" + output_prefix + "/" + exact_file;
    }

    exact_file += "_N" + std::to_string(N_exact);
    exact_file += ".csv";

    return exact_file;
}


// [3] Write numerical solution CSV
template<int DIM, typename EOS>
inline void write_numerical_output(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N,
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::vector<EOSParams>& material_params,
    const std::vector<std::vector<double>>* phi_list = nullptr
)
{
    const std::string filename = build_solution_filename<DIM>(cfg, N);

    std::cout << "Writing to: " << filename << "\n";

    std::filesystem::path p(filename);
    ensure_directory(p.parent_path().string());

    write_csv<DIM, EOS>(
        filename,
        U,
        material_id,
        N,
        cfg.domain_min,
        cfg.domain_max,
        material_params,
        phi_list
    );

    std::cout << "Written: " << filename << "\n";
}


// [4] Write exact 1D solution CSV
template<typename EOS>
inline void write_exact_output_1d(
    const Config<1>& cfg
)
{
    if (!cfg.exact_riemann) {
        return;
    }

    std::vector<ExactState<1>> exact;

    std::array<int, 1> N_exact{};
    N_exact[0] = 2000;

    compute_exact_solution<EOS>(
        exact,
        cfg,
        N_exact,
        cfg.tfinal
    );

    const std::string exact_file = build_exact_filename(
        cfg.output_dir,
        cfg.output_prefix,
        N_exact[0]
    );

    std::filesystem::path p_exact(exact_file);
    ensure_directory(p_exact.parent_path().string());

    std::array<double, 1> domain_min_1d{cfg.domain_min[0]};
    std::array<double, 1> domain_max_1d{cfg.domain_max[0]};

    write_exact_csv<1>(
        exact_file,
        exact,
        N_exact,
        domain_min_1d,
        domain_max_1d
    );

    std::cout << "Written exact: " << exact_file << "\n";
}

