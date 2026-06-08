#pragma once

#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "src/io/config.hpp"

namespace app_io {
    // [0] Build output folder path used by solution CSVs
    template<int DIM>
    inline std::filesystem::path build_case_output_dir(
        const Config<DIM>& cfg
    )
    {
        if (cfg.output_dir.empty()) {
            return std::filesystem::path{};
        }

        return std::filesystem::path(cfg.output_dir) / cfg.output_prefix;
    }


    // [1] Build runtime report path
    template<int DIM>
    inline std::filesystem::path build_runtime_report_filename(
        const Config<DIM>& cfg,
        const std::array<int, DIM>& N
    )
    {
        std::string stem = cfg.output_prefix;

        for (int d = 0; d < DIM; ++d) {
            stem += "_N" + std::to_string(N[d]);
        }

        stem += "_runtime.txt";

        const std::filesystem::path dir = build_case_output_dir<DIM>(cfg);
        if (dir.empty()) {
            return std::filesystem::path(stem);
        }

        return dir / stem;
    }


    // [2] Count cells in a structured grid
    template<int DIM>
    inline long long count_cells(
        const std::array<int, DIM>& N
    )
    {
        long long cells = 1;
        for (int d = 0; d < DIM; ++d) {
            cells *= static_cast<long long>(N[d]);
        }
        return cells;
    }

    // [3] Write one runtime report for one resolution
    template<int DIM>
    inline void write_runtime_report(
        const Config<DIM>& cfg,
        const std::array<int, DIM>& N,
        int steps,
        double final_time,
        double wall_seconds
    )
    {
        const std::filesystem::path filename =
            build_runtime_report_filename<DIM>(cfg, N);

        if (!filename.parent_path().empty()) {
            std::filesystem::create_directories(filename.parent_path());
        }

        std::ofstream file(filename);
        if (!file) {
            throw std::runtime_error(
                "write_runtime_report: failed to open " + filename.string()
            );
        }

        const long long cells = count_cells<DIM>(N);
        const double cell_updates =
            static_cast<double>(cells) * static_cast<double>(steps);
        const double cost_per_cell_update =
            (cell_updates > 0.0) ? wall_seconds / cell_updates : 0.0;

        file << "interface_method = " << cfg.interface_method << "\n";
        file << "dimension = " << DIM << "\n";
        file << "N = ";
        for (int d = 0; d < DIM; ++d) {
            if (d > 0) {
                file << "x";
            }
            file << N[d];
        }
        file << "\n";
        file << "cells = " << cells << "\n";
        file << "steps = " << steps << "\n";
        file << "final_time = " << final_time << "\n";
        file << "wall_time_seconds = " << wall_seconds << "\n";
        file << "cost_per_cell_update_seconds = "
             << cost_per_cell_update << "\n";
    }

}





