#pragma once

#include <array>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
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

    inline std::string format_time_tag(double time)
    {
        std::ostringstream stream;
        stream << std::scientific << std::setprecision(6) << time;
        std::string text = stream.str();

        for (char& c : text) {
            if (c == '.') {
                c = 'p';
            }
            else if (c == '+') {
                c = 'p';
            }
            else if (c == '-') {
                c = 'm';
            }
        }

        return "t" + text;
    }

    // [1] Build numerical solution output path
    template<int DIM>
    inline std::string build_solution_filename(
        const Config<DIM>& cfg,
        const std::array<int, DIM>& N,
        const std::string& suffix = ""
    )
    {
        std::string filename = cfg.output_prefix;

        if (!cfg.output_dir.empty()) {
            filename = cfg.output_dir + "/" + filename + "/" + filename;
        }

        if (!suffix.empty()) {
            filename += "_" + suffix;
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
        const dim::EOSParams& params,
        const std::string& suffix = ""
    )
    {
        const std::string filename = dim_app::build_solution_filename<DIM>(cfg, N, suffix);

        std::cout << "Writing to: " << filename << "\n";

        std::filesystem::path path(filename);
        ensure_directory(path.parent_path().string());

        dim::write_csv<DIM>(filename, U, N, cfg.domain_min, cfg.domain_max, params);

        std::cout << "Written: " << filename << "\n";
    }

}



