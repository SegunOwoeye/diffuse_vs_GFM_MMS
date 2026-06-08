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

#include "src/dim/flux.hpp"
#include "src/dim/primitives.hpp"
#include "src/dim/state.hpp"
#include "src/euler/flux.hpp"
#include "src/euler/state.hpp"
#include "src/euler/eos_params.hpp"
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

    template<int DIM>
    inline ConservationTotals<DIM> zero_conservation_totals(int nmat)
    {
        ConservationTotals<DIM> totals{};
        totals.material_mass.assign(nmat, 0.0);
        return totals;
    }

    template<int DIM>
    inline int conservation_linear_index(
        const std::array<int, DIM>& idx,
        const std::array<int, DIM>& N
    )
    {
        int linear = idx[0];
        int stride = N[0];
        for (int d = 1; d < DIM; ++d) {
            linear += idx[d] * stride;
            stride *= N[d];
        }
        return linear;
    }

    template<int DIM>
    inline double boundary_face_area(
        const std::array<double, DIM>& dx,
        int normal_dir
    )
    {
        double area = 1.0;
        for (int d = 0; d < DIM; ++d) {
            if (d != normal_dir) {
                area *= dx[d];
            }
        }
        return area;
    }

    template<int DIM>
    inline void add_dim_flux_to_totals(
        ConservationTotals<DIM>& totals,
        const dim::Flux<DIM>& flux,
        double scale
    )
    {
        double mass_flux = 0.0;
        for (int k = 0; k < static_cast<int>(flux.partial_mass.size()); ++k) {
            const double value = scale * flux.partial_mass[k];
            totals.material_mass[k] += value;
            mass_flux += value;
        }
        totals.mass += mass_flux;
        totals.energy += scale * flux.E;
        for (int d = 0; d < DIM; ++d) {
            totals.momentum[d] += scale * flux.mom[d];
        }
    }

    template<int DIM>
    inline void add_sharp_flux_to_totals(
        ConservationTotals<DIM>& totals,
        const Conserved<DIM>& flux,
        double scale,
        int material = -1
    )
    {
        totals.mass += scale * flux.rho;
        totals.energy += scale * flux.E;
        for (int d = 0; d < DIM; ++d) {
            totals.momentum[d] += scale * flux.mom[d];
        }
        if (material >= 0 && material < static_cast<int>(totals.material_mass.size())) {
            totals.material_mass[material] += scale * flux.rho;
        }
    }

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

    template<int DIM>
    inline ConservationTotals<DIM> compute_dim_boundary_flux(
        const std::vector<dim::State<DIM>>& U,
        const std::array<int, DIM>& N,
        const std::array<double, DIM>& dx,
        const dim::EOSParams& params
    )
    {
        ConservationTotals<DIM> totals = zero_conservation_totals<DIM>(params.nmat());
        std::array<int, DIM> idx{};

        for (int dir = 0; dir < DIM; ++dir) {
            const double area = boundary_face_area<DIM>(dx, dir);
            std::array<double, DIM> normal{};
            normal[dir] = 1.0;

            const int transverse_count = [&]() {
                int count = 1;
                for (int d = 0; d < DIM; ++d) {
                    if (d != dir) count *= N[d];
                }
                return count;
            }();

            for (int line = 0; line < transverse_count; ++line) {
                int tmp = line;
                for (int d = DIM - 1; d >= 0; --d) {
                    if (d == dir) {
                        idx[d] = 0;
                    }
                    else {
                        idx[d] = tmp % N[d];
                        tmp /= N[d];
                    }
                }

                idx[dir] = 0;
                const auto& left = U[conservation_linear_index<DIM>(idx, N)];
                const auto left_prim = dim::cons_to_prim<DIM>(left, params);
                const auto left_flux = dim::compute_flux_normal<DIM>(left, left_prim, normal);

                idx[dir] = N[dir] - 1;
                const auto& right = U[conservation_linear_index<DIM>(idx, N)];
                const auto right_prim = dim::cons_to_prim<DIM>(right, params);
                const auto right_flux = dim::compute_flux_normal<DIM>(right, right_prim, normal);

                add_dim_flux_to_totals<DIM>(totals, right_flux, area);
                add_dim_flux_to_totals<DIM>(totals, left_flux, -area);
            }
        }

        return totals;
    }

    template<int DIM, typename EOS>
    inline ConservationTotals<DIM> compute_sharp_boundary_flux(
        const std::vector<Conserved<DIM>>& U,
        const std::vector<int>& material_id,
        const std::array<int, DIM>& N,
        const std::array<double, DIM>& dx,
        const std::vector<EOSParams>& material_params
    )
    {
        ConservationTotals<DIM> totals =
            zero_conservation_totals<DIM>(static_cast<int>(material_params.size()));
        std::array<int, DIM> idx{};

        for (int dir = 0; dir < DIM; ++dir) {
            const double area = boundary_face_area<DIM>(dx, dir);
            std::array<double, DIM> normal{};
            normal[dir] = 1.0;

            int transverse_count = 1;
            for (int d = 0; d < DIM; ++d) {
                if (d != dir) transverse_count *= N[d];
            }

            for (int line = 0; line < transverse_count; ++line) {
                int tmp = line;
                for (int d = DIM - 1; d >= 0; --d) {
                    if (d == dir) {
                        idx[d] = 0;
                    }
                    else {
                        idx[d] = tmp % N[d];
                        tmp /= N[d];
                    }
                }

                idx[dir] = 0;
                int left_id = conservation_linear_index<DIM>(idx, N);
                int left_mat = material_id.empty() ? 0 : material_id[left_id];
                left_mat = std::clamp(left_mat, 0, static_cast<int>(material_params.size()) - 1);
                const auto left_flux =
                    compute_flux_normal<DIM, EOS>(U[left_id], normal, material_params[left_mat]);

                idx[dir] = N[dir] - 1;
                int right_id = conservation_linear_index<DIM>(idx, N);
                int right_mat = material_id.empty() ? 0 : material_id[right_id];
                right_mat = std::clamp(right_mat, 0, static_cast<int>(material_params.size()) - 1);
                const auto right_flux =
                    compute_flux_normal<DIM, EOS>(U[right_id], normal, material_params[right_mat]);

                add_sharp_flux_to_totals<DIM>(totals, right_flux, area, right_mat);
                add_sharp_flux_to_totals<DIM>(totals, left_flux, -area, left_mat);
            }
        }

        return totals;
    }

    template<int DIM, typename EOS>
    inline ConservationTotals<DIM> compute_sharp_interface_flux_mismatch(
        const std::vector<Conserved<DIM>>& U,
        const std::vector<int>& material_id,
        const std::array<int, DIM>& N,
        const std::array<double, DIM>& dx,
        const std::vector<EOSParams>& material_params
    )
    {
        ConservationTotals<DIM> totals =
            zero_conservation_totals<DIM>(static_cast<int>(material_params.size()));
        if (material_id.empty()) {
            return totals;
        }

        std::array<int, DIM> idx{};
        for (int dir = 0; dir < DIM; ++dir) {
            const double area = boundary_face_area<DIM>(dx, dir);
            std::array<double, DIM> normal{};
            normal[dir] = 1.0;

            for (int linear = 0; linear < static_cast<int>(U.size()); ++linear) {
                int tmp = linear;
                for (int d = 0; d < DIM; ++d) {
                    idx[d] = tmp % N[d];
                    tmp /= N[d];
                }
                if (idx[dir] + 1 >= N[dir]) {
                    continue;
                }

                const int left_id = linear;
                std::array<int, DIM> right_idx = idx;
                right_idx[dir] += 1;
                const int right_id = conservation_linear_index<DIM>(right_idx, N);
                if (material_id[left_id] == material_id[right_id]) {
                    continue;
                }

                int left_mat = std::clamp(
                    material_id[left_id],
                    0,
                    static_cast<int>(material_params.size()) - 1
                );
                int right_mat = std::clamp(
                    material_id[right_id],
                    0,
                    static_cast<int>(material_params.size()) - 1
                );

                const auto left_flux =
                    compute_flux_normal<DIM, EOS>(U[left_id], normal, material_params[left_mat]);
                const auto right_flux =
                    compute_flux_normal<DIM, EOS>(U[right_id], normal, material_params[right_mat]);

                add_sharp_flux_to_totals<DIM>(totals, left_flux, area, left_mat);
                add_sharp_flux_to_totals<DIM>(totals, right_flux, -area, right_mat);
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

    template<int DIM>
    inline void add_scaled_totals(
        ConservationTotals<DIM>& target,
        const ConservationTotals<DIM>& source,
        double scale
    )
    {
        target.mass += scale * source.mass;
        target.energy += scale * source.energy;
        for (int d = 0; d < DIM; ++d) {
            target.momentum[d] += scale * source.momentum[d];
        }
        if (target.material_mass.size() < source.material_mass.size()) {
            target.material_mass.resize(source.material_mass.size(), 0.0);
        }
        for (int k = 0; k < static_cast<int>(source.material_mass.size()); ++k) {
            target.material_mass[k] += scale * source.material_mass[k];
        }
    }

    template<int DIM>
    inline void update_characteristic_scale(
        ConservationTotals<DIM>& scale,
        const ConservationTotals<DIM>& current
    )
    {
        scale.mass = std::max(scale.mass, std::abs(current.mass));
        scale.energy = std::max(scale.energy, std::abs(current.energy));
        for (int d = 0; d < DIM; ++d) {
            scale.momentum[d] = std::max(scale.momentum[d], std::abs(current.momentum[d]));
        }
        if (scale.material_mass.size() < current.material_mass.size()) {
            scale.material_mass.resize(current.material_mass.size(), 0.0);
        }
        for (int k = 0; k < static_cast<int>(current.material_mass.size()); ++k) {
            scale.material_mass[k] =
                std::max(scale.material_mass[k], std::abs(current.material_mass[k]));
        }
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
            : initial_(initial),
              boundary_flux_integral_(zero_conservation_totals<DIM>(
                  static_cast<int>(initial.material_mass.size()))),
              interface_flux_mismatch_integral_(zero_conservation_totals<DIM>(
                  static_cast<int>(initial.material_mass.size()))),
              characteristic_scale_(initial)
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

        void accumulate_fluxes(
            double dt,
            const ConservationTotals<DIM>& boundary_flux,
            const ConservationTotals<DIM>* interface_flux_mismatch = nullptr
        )
        {
            add_scaled_totals<DIM>(boundary_flux_integral_, boundary_flux, dt);
            if (interface_flux_mismatch != nullptr) {
                add_scaled_totals<DIM>(
                    interface_flux_mismatch_integral_,
                    *interface_flux_mismatch,
                    dt
                );
            }
        }

        void write(
            int step,
            double time,
            const ConservationTotals<DIM>& current
        )
        {
            update_characteristic_scale<DIM>(characteristic_scale_, current);

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

            write_balance_columns("mass", current.mass, initial_.mass,
                boundary_flux_integral_.mass, interface_flux_mismatch_integral_.mass,
                characteristic_scale_.mass);

            for (int d = 0; d < DIM; ++d) {
                write_balance_columns(
                    "momentum" + std::to_string(d),
                    current.momentum[d],
                    initial_.momentum[d],
                    boundary_flux_integral_.momentum[d],
                    interface_flux_mismatch_integral_.momentum[d],
                    characteristic_scale_.momentum[d]
                );
            }

            write_balance_columns("energy", current.energy, initial_.energy,
                boundary_flux_integral_.energy, interface_flux_mismatch_integral_.energy,
                characteristic_scale_.energy);

            for (int k = 0; k < static_cast<int>(current.material_mass.size()); ++k) {
                write_balance_columns(
                    "material_mass" + std::to_string(k),
                    current.material_mass[k],
                    initial_.material_mass[k],
                    boundary_flux_integral_.material_mass[k],
                    interface_flux_mismatch_integral_.material_mass[k],
                    characteristic_scale_.material_mass[k]
                );
            }

            file_ << "\n";
        }

    private:
        static double normalized_balance_residual(
            double residual,
            double initial,
            double characteristic_scale,
            double floor = 1.0e-12
        )
        {
            const double denom = std::max(
                {std::abs(initial), std::abs(characteristic_scale), floor}
            );
            return std::abs(residual) / denom;
        }

        static int near_zero_initial_flag(
            double initial,
            double characteristic_scale,
            double floor = 1.0e-12
        )
        {
            return (std::abs(initial) <=
                    std::max(floor, 1.0e-10 * std::max(std::abs(characteristic_scale), floor)))
                ? 1
                : 0;
        }

        void write_balance_columns(
            const std::string&,
            double current,
            double initial,
            double boundary_flux,
            double interface_flux_mismatch,
            double characteristic_scale
        )
        {
            const double raw_drift = current - initial;
            const double residual = raw_drift + boundary_flux;
            file_ << "," << raw_drift
                  << "," << boundary_flux
                  << "," << residual
                  << "," << normalized_balance_residual(
                         residual,
                         initial,
                         characteristic_scale
                     )
                  << "," << interface_flux_mismatch
                  << "," << near_zero_initial_flag(
                         initial,
                         characteristic_scale
                     );
        }

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

            write_balance_header("mass");

            for (int d = 0; d < DIM; ++d) {
                write_balance_header("momentum" + std::to_string(d));
            }

            write_balance_header("energy");

            for (int k = 0; k < static_cast<int>(initial_.material_mass.size()); ++k) {
                write_balance_header("material_mass" + std::to_string(k));
            }

            file_ << "\n";
        }

        void write_balance_header(const std::string& variable)
        {
            file_ << ",raw_drift_" << variable
                  << ",boundary_flux_" << variable
                  << ",balance_residual_" << variable
                  << ",normalized_balance_residual_" << variable
                  << ",interface_flux_mismatch_" << variable
                  << ",near_zero_initial_" << variable;
        }

        std::ofstream file_{};
        ConservationTotals<DIM> initial_{};
        ConservationTotals<DIM> boundary_flux_integral_{};
        ConservationTotals<DIM> interface_flux_mismatch_integral_{};
        ConservationTotals<DIM> characteristic_scale_{};
    };

}
