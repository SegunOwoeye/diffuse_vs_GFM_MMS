#pragma once

#include <array>
#include <filesystem>
#include <iostream>
#include <string>

#include "src/dim/eos_params.hpp"
#include "src/dim/state.hpp"
#include "src/io/config.hpp"
#include "src/io/dim_write_csv.hpp"

namespace dim_app {

    // [0] Ensure parent directory exists
    inline void ensure_directory(const std::string& dir)
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
    template<int DIM>
    inline void write_numerical_output(
        const Config<DIM>& cfg,
        const std::array<int, DIM>& N,
        const std::vector<dim::State<DIM>>& U,
        const dim::EOSParams& params
    )
    {
        const std::string filename = dim_app::build_solution_filename<DIM>(cfg, N);

        std::cout << "Writing to: " << filename << "\n";

        std::filesystem::path path(filename);
        ensure_directory(path.parent_path().string());

        dim::write_csv<DIM>(filename, U, N, cfg.domain_min, cfg.domain_max, params);

        std::cout << "Written: " << filename << "\n";
    }

}




