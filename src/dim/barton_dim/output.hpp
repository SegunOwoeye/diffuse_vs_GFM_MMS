#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "src/dim/barton_dim/config.hpp"
#include "src/dim/barton_dim/primitives.hpp"
#include "src/dim/barton_dim/solver.hpp"

namespace dim::barton_dim {

template<int DIM>
inline void write_rgfm_compatible_csv(
    const std::string& filename,
    const std::vector<State<DIM>>& states,
    const Config<DIM>& config,
    double time)
{
    if constexpr (DIM == 1) {
        const std::filesystem::path path(filename);
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
        std::ofstream output(filename);
        if (!output) throw std::runtime_error("Cannot write Barton-DIM rGFM-compatible CSV: " + filename);
        output << std::setprecision(17);
        output << "x,material,rho,u,p,sigma11,sigma22,sigma33,time\n";
        const double dx = (config.domain_max[0] - config.domain_min[0]) / config.cells[0];
        for (int linear = 0; linear < static_cast<int>(states.size()); ++linear) {
            const State<DIM>& state = states[linear];
            const Primitive<DIM> primitive = cons_to_prim(state, config.materials);
            const bool solid = state.alpha_solid >= 0.5;
            const double x = config.domain_min[0] + (linear + 0.5) * dx;
            output << x << "," << (solid ? "solid" : "fluid") << ","
                   << state.solid_mass + state.fluid_mass << "," << primitive.velocity[0] << ","
                   << (solid ? 0.0 : primitive.p_fluid) << ","
                   << (solid ? primitive.sigma_solid[0] : 0.0) << ","
                   << (solid ? primitive.sigma_solid[4] : 0.0) << ","
                   << (solid ? primitive.sigma_solid[8] : 0.0) << "," << time << "\n";
        }
        return;
    }
    else if constexpr (DIM != 2) {
        throw std::runtime_error("rGFM-compatible Barton-DIM output requires dimension=2");
    }
    else {
    const std::filesystem::path path(filename);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(filename);
    if (!output) throw std::runtime_error("Cannot write Barton-DIM rGFM-compatible CSV: " + filename);
    output << std::setprecision(17);
    output << "x,y,phi,x_lag,y_lag,x_mat,y_mat,material,rho,u,v,p,sigma_xx,sigma_xy,sigma_yy,"
              "sigma_nn,sigma_ss,sigma_sn,vn,time\n";
    const double dx = (config.domain_max[0] - config.domain_min[0]) / config.cells[0];
    const double dy = (config.domain_max[1] - config.domain_min[1]) / config.cells[1];
    for (int linear = 0; linear < static_cast<int>(states.size()); ++linear) {
        const auto index = unflatten<DIM>(linear, config.cells);
        const State<DIM>& state = states[linear];
        const Primitive<DIM> primitive = cons_to_prim(state, config.materials);
        const double x = config.domain_min[0] + (index[0] + 0.5) * dx;
        const double y = config.domain_min[1] + (index[1] + 0.5) * dy;
        const bool solid = state.alpha_solid >= 0.5;
        const double sigma_xx = solid ? primitive.sigma_solid[0] : 0.0;
        const double sigma_xy = solid ? 0.5 * (primitive.sigma_solid[1] + primitive.sigma_solid[3]) : 0.0;
        const double sigma_yy = solid ? primitive.sigma_solid[4] : 0.0;
        output << x << "," << y << "," << state.alpha_solid - 0.5 << ","
               << x << "," << y << ",";
        if (solid) output << x << "," << y;
        else output << "nan,nan";
        output << "," << (solid ? "solid" : "fluid") << ","
               << state.solid_mass + state.fluid_mass << "," << primitive.velocity[0] << ","
               << primitive.velocity[1] << "," << (solid ? 0.0 : primitive.p_fluid) << ","
               << sigma_xx << "," << sigma_xy << "," << sigma_yy << ","
               << sigma_xx << "," << sigma_yy << "," << sigma_xy << ","
               << primitive.velocity[0] << "," << time << "\n";
    }
    }
}

template<int DIM>
inline void write_csv(
    const std::string& filename,
    const std::vector<State<DIM>>& states,
    const Config<DIM>& config,
    double time)
{
    if (config.output_format == "rgfm_compatible") {
        write_rgfm_compatible_csv(filename, states, config, time);
        return;
    }
    const std::filesystem::path path(filename);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(filename);
    if (!output) throw std::runtime_error("Cannot write Barton-DIM CSV: " + filename);
    output << std::setprecision(17);
    for (int d = 0; d < DIM; ++d) output << "x" << d << ",";
    output << "time,alpha_solid,rho_solid,rho_fluid,rho,u0";
    for (int d = 1; d < DIM; ++d) output << ",u" << d;
    output << ",p_solid,p_fluid,p_mixture,sigma_xx,sigma_xy,sigma_yy,solid_mass,fluid_mass,total_energy,solid_energy,fluid_energy,eqps,damage";
    for (int q = 0; q < 9; ++q) output << ",F" << q;
    output << "\n";
    std::array<double, DIM> spacing{};
    for (int d = 0; d < DIM; ++d) spacing[d] = (config.domain_max[d] - config.domain_min[d]) / config.cells[d];
    for (int linear = 0; linear < static_cast<int>(states.size()); ++linear) {
        const auto index = unflatten<DIM>(linear, config.cells);
        const State<DIM>& state = states[linear];
        const Primitive<DIM> primitive = cons_to_prim(state, config.materials);
        for (int d = 0; d < DIM; ++d) {
            output << config.domain_min[d] + (index[d] + 0.5) * spacing[d] << ",";
        }
        const double density = state.solid_mass + state.fluid_mass;
        output << time << "," << state.alpha_solid << "," << primitive.rho_solid << ","
               << primitive.rho_fluid << "," << density << "," << primitive.velocity[0];
        for (int d = 1; d < DIM; ++d) output << "," << primitive.velocity[d];
        output << "," << primitive.p_solid << "," << primitive.p_fluid << ","
               << normal_pressure(primitive, 0) << "," << primitive.sigma_solid[0] << ","
               << 0.5 * (primitive.sigma_solid[1] + primitive.sigma_solid[3]) << ","
               << primitive.sigma_solid[4] << "," << state.solid_mass << "," << state.fluid_mass
               << "," << state.total_energy << "," << state.solid_energy << "," << state.fluid_energy
               << "," << state.solid_rho_eqps / std::max(state.solid_mass, 1.0e-12)
               << "," << state.solid_rho_damage / std::max(state.solid_mass, 1.0e-12);
        const auto deformation = solid_deformation(state);
        for (double value : deformation) output << "," << value;
        output << "\n";
    }
}

} // namespace dim::barton_dim
