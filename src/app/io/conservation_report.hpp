#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/dim/state.hpp"
#include "src/euler/state.hpp"
#include "src/io/config.hpp"

namespace app_io {

    // [0] Runtime toggle for conservation diagnostics
    inline bool conservation_tracking_enabled()
    {
        const char* value = std::getenv("SOLVER_CONSERVATION");
        if (value == nullptr) {
            return false;
        }

        const std::string flag(value);
        return flag == "1" || flag == "true" || flag == "TRUE" ||
               flag == "on" || flag == "ON" || flag == "yes" ||
               flag == "YES";
    }

    // [1] Optional sampling interval
    inline int conservation_tracking_interval()
    {
        const char* value = std::getenv("SOLVER_CONSERVATION_INTERVAL");
        if (value == nullptr) {
            return 1;
        }

        try {
            return std::max(1, std::stoi(value));
        }
        catch (...) {
            return 1;
        }
    }

    // [2] Build output folder path used by solution CSVs
    template<int DIM>
    inline std::filesystem::path build_conservation_case_output_dir(
        const Config<DIM>& cfg
    )
    {
        if (cfg.output_dir.empty()) {
            return std::filesystem::path{};
        }

        return std::filesystem::path(cfg.output_dir) / cfg.output_prefix;
    }

    // [3] Build conservation report path
    template<int DIM>
    inline std::filesystem::path build_conservation_report_filename(
        const Config<DIM>& cfg,
        const std::array<int, DIM>& N
    )
    {
        std::string stem = cfg.output_prefix;

        for (int d = 0; d < DIM; ++d) {
            stem += "_N" + std::to_string(N[d]);
        }

        stem += "_conservation.csv";

        const std::filesystem::path dir =
            build_conservation_case_output_dir<DIM>(cfg);
        if (dir.empty()) {
            return std::filesystem::path(stem);
        }

        return dir / stem;
    }

    // [4] Cell volume for integral diagnostics
    template<int DIM>
    inline double cell_volume(
        const std::array<double, DIM>& dx
    )
    {
        double volume = 1.0;
        for (int d = 0; d < DIM; ++d) {
            volume *= dx[d];
        }
        return volume;
    }

    template<int DIM>
    struct ConservationTotals {
        double mass = 0.0;
        std::array<double, DIM> momentum{};
        double energy = 0.0;
        std::vector<double> material_mass{};
    };

    // [5] DIM totals
    template<int DIM>
    inline ConservationTotals<DIM> compute_dim_conservation_totals(
        const std::vector<dim::State<DIM>>& U,
        const std::array<double, DIM>& dx,
        int nmat
    )
    {
        ConservationTotals<DIM> totals{};
        totals.material_mass.assign(nmat, 0.0);

        const double volume = cell_volume<DIM>(dx);

        for (const auto& state : U) {
            double cell_mass = 0.0;
            for (int k = 0; k < nmat; ++k) {
                const double partial = state.partial_mass[k] * volume;
                totals.material_mass[k] += partial;
                cell_mass += partial;
            }

            totals.mass += cell_mass;
            totals.energy += state.E * volume;

            for (int d = 0; d < DIM; ++d) {
                totals.momentum[d] += state.mom[d] * volume;
            }
        }

        return totals;
    }

    // [6] SIM totals
    template<int DIM>
    inline ConservationTotals<DIM> compute_sharp_conservation_totals(
        const std::vector<Conserved<DIM>>& U,
        const std::vector<int>& material_id,
        const std::array<double, DIM>& dx,
        int nmat
    )
    {
        ConservationTotals<DIM> totals{};
        totals.material_mass.assign(nmat, 0.0);

        const double volume = cell_volume<DIM>(dx);

        for (int i = 0; i < static_cast<int>(U.size()); ++i) {
            const double cell_mass = U[i].rho * volume;
            totals.mass += cell_mass;
            totals.energy += U[i].E * volume;

            if (!material_id.empty()) {
                const int mat = material_id[i];
                if (mat >= 0 && mat < nmat) {
                    totals.material_mass[mat] += cell_mass;
                }
            }

            for (int d = 0; d < DIM; ++d) {
                totals.momentum[d] += U[i].mom[d] * volume;
            }
        }

        return totals;
    }

    inline double relative_change(
        double value,
        double initial,
        double floor = 1.0e-12
    )
    {
        return std::abs(value - initial) / std::max(std::abs(initial), floor);
    }

    // Class to write the conservation report
    template<int DIM>
    class ConservationReport {
    public:
        ConservationReport(
            const Config<DIM>& cfg,
            const std::array<int, DIM>& N,
            const ConservationTotals<DIM>& initial
        )
            : initial_(initial)
        {
            const std::filesystem::path filename =
                build_conservation_report_filename<DIM>(cfg, N);

            if (!filename.parent_path().empty()) {
                std::filesystem::create_directories(filename.parent_path());
            }

            file_.open(filename);
            if (!file_) {
                throw std::runtime_error(
                    "ConservationReport: failed to open " + filename.string()
                );
            }

            write_header();
        }

        void write(
            int step,
            double time,
            const ConservationTotals<DIM>& current
        )
        {
            file_ << step << "," << time
                  << "," << current.mass;

            for (int d = 0; d < DIM; ++d) {
                file_ << "," << current.momentum[d];
            }

            file_ << "," << current.energy
                  << "," << relative_change(current.mass, initial_.mass);

            for (int d = 0; d < DIM; ++d) {
                file_ << ","
                      << relative_change(
                             current.momentum[d],
                             initial_.momentum[d]
                         );
            }

            file_ << "," << relative_change(current.energy, initial_.energy);

            for (int k = 0; k < static_cast<int>(current.material_mass.size()); ++k) {
                file_ << "," << current.material_mass[k]
                      << ","
                      << relative_change(
                             current.material_mass[k],
                             initial_.material_mass[k]
                         );
            }

            file_ << "\n";
        }

    private:
        void write_header()
        {
            file_ << "step,time,mass";

            for (int d = 0; d < DIM; ++d) {
                file_ << ",momentum" << d;
            }

            file_ << ",energy,eps_mass";

            for (int d = 0; d < DIM; ++d) {
                file_ << ",eps_momentum" << d;
            }

            file_ << ",eps_energy";

            for (int k = 0; k < static_cast<int>(initial_.material_mass.size()); ++k) {
                file_ << ",material_mass" << k
                      << ",eps_material_mass" << k;
            }

            file_ << "\n";
        }

        std::ofstream file_{};
        ConservationTotals<DIM> initial_{};
    };

}
