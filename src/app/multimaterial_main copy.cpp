#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <cmath>
#include <limits>

#include "src/app/dimension.hpp"
#include "src/io/config.hpp"
#include "src/io/config_loader.hpp"
#include "src/setup/initial_conditions.hpp"
#include "src/euler/solver/advance_step.hpp"
#include "src/euler/solver/solver_context.hpp"
#include "src/euler/level_set/level_set.hpp"
#include "src/io/write_csv.hpp"
#include "src/io/compute_exact_solution.hpp"
#include "src/io/write_exact_csv.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/grid/grid_utils.hpp"

// Multimaterial / Multidimensional Compressible Euler Flow Solver

using EOS = IdealGasEOS;


// [0] Directory helper
inline void ensure_directory(const std::string& dir)
{
    if (!dir.empty()) {
        std::filesystem::create_directories(dir);
    }
}


// [1] Build EOS params
inline std::vector<EOSParams> build_material_params(
    const std::vector<MaterialConfig>& materials
)
{
    std::vector<EOSParams> params(materials.size());

    for (const auto& m : materials) {
        if (m.id < 0 || m.id >= static_cast<int>(materials.size())) {
            throw std::runtime_error("build_material_params: invalid material id");
        }

        if (m.type == "ideal_gas") {
            if (!m.params.count("gamma")) {
                throw std::runtime_error("build_material_params: missing gamma in material");
            }

            params[m.id].gamma = m.params.at("gamma");
        }
        else {
            throw std::runtime_error("build_material_params: unsupported EOS type: " + m.type);
        }
    }

    return params;
}


// [2] Region volume helper
template<int DIM>
inline double region_volume(
    const Region<DIM>& r
)
{
    double vol = 1.0;

    for (int d = 0; d < DIM; ++d) {
        const double width = r.upper[d] - r.lower[d];

        if (width <= 0.0) {
            throw std::runtime_error("region_volume: non-positive region width");
        }

        vol *= width;
    }

    return vol;
}


// [3] Signed distance to axis-aligned box
template<int DIM>
inline double signed_distance_to_box(
    const std::array<double, DIM>& x,
    const Region<DIM>& region
)
{
    double outside_sq = 0.0;
    double inside_dist = std::numeric_limits<double>::max();
    bool inside = true;

    for (int d = 0; d < DIM; ++d) {
        if (x[d] < region.lower[d]) {
            const double dist = region.lower[d] - x[d];
            outside_sq += dist * dist;
            inside = false;
        }
        else if (x[d] > region.upper[d]) {
            const double dist = x[d] - region.upper[d];
            outside_sq += dist * dist;
            inside = false;
        }
        else {
            const double dist_to_lower = x[d] - region.lower[d];
            const double dist_to_upper = region.upper[d] - x[d];
            inside_dist = std::min(inside_dist, std::min(dist_to_lower, dist_to_upper));
        }
    }

    return inside ? -inside_dist : std::sqrt(outside_sq);
}


// [4] Initial level set data
template<int DIM>
struct InitialLevelSetData {
    std::vector<std::vector<double>> phi_list;
    std::vector<int> phi_material_ids;
    int background_material_id = 0;
};


// [5] Build initial level sets for multimaterial GFM
template<int DIM>
inline InitialLevelSetData<DIM> initialise_phi_data_from_regions(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N
)
{
    InitialLevelSetData<DIM> out{};

    if (cfg.interface_method != "GFM") {
        return out;
    }

    if (!cfg.use_level_set) {
        throw std::runtime_error("initialise_phi_data_from_regions: GFM requires use_level_set = true");
    }

    if (cfg.initial_condition != "regions") {
        throw std::runtime_error(
            "initialise_phi_data_from_regions: current multimaterial GFM initialisation only supports region ICs"
        );
    }

    if (cfg.regions.size() < 2) {
        throw std::runtime_error(
            "initialise_phi_data_from_regions: GFM requires at least 2 regions/materials"
        );
    }

    // [5.1] Choose background as the largest-volume region
    int bg_region_idx = 0;
    double bg_volume = region_volume<DIM>(cfg.regions[0]);

    for (int r = 1; r < static_cast<int>(cfg.regions.size()); ++r) {
        const double vol = region_volume<DIM>(cfg.regions[r]);

        if (vol > bg_volume) {
            bg_volume = vol;
            bg_region_idx = r;
        }
    }

    out.background_material_id = cfg.regions[bg_region_idx].material_id;

    int total_cells = 1;
    for (int d = 0; d < DIM; ++d) {
        if (N[d] <= 0) {
            throw std::runtime_error("initialise_phi_data_from_regions: invalid grid size");
        }
        total_cells *= N[d];
    }

    std::array<double, DIM> dx{};
    for (int d = 0; d < DIM; ++d) {
        dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / N[d];
    }

    // [5.2] Build one signed-distance field per non-background region
    for (int r = 0; r < static_cast<int>(cfg.regions.size()); ++r) {
        if (r == bg_region_idx) {
            continue;
        }

        const Region<DIM>& region = cfg.regions[r];

        std::vector<double> phi(total_cells, 0.0);
        std::array<int, DIM> idx{};

        for (int linear = 0; linear < total_cells; ++linear) {
            int tmp = linear;
            for (int d = DIM - 1; d >= 0; --d) {
                idx[d] = tmp % N[d];
                tmp /= N[d];
            }

            const auto x = compute_cell_center<DIM>(idx, cfg.domain_min, dx);
            phi[linear] = signed_distance_to_box<DIM>(x, region);
        }

        out.phi_list.push_back(std::move(phi));
        out.phi_material_ids.push_back(region.material_id);
    }

    return out;
}


// [6] Build solver context
template<int DIM>
inline SolverContext<DIM> build_solver_context(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N,
    const std::vector<int>& material_id,
    const std::vector<EOSParams>& material_params,
    const InitialLevelSetData<DIM>& ls_data
)
{
    SolverContext<DIM> ctx{};

    ctx.N = N;

    for (int d = 0; d < DIM; ++d) {
        ctx.dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / N[d];
    }

    ctx.cfl = cfg.cfl;
    ctx.dt_max = 1e-3;

    ctx.material_id = material_id;
    ctx.material_params = material_params;

    ctx.initialise_level_set_grid();

    ctx.phi_list = ls_data.phi_list;
    ctx.phi_material_ids = ls_data.phi_material_ids;
    ctx.background_material_id = ls_data.background_material_id;

    ctx.reinit_enabled = (cfg.interface_method == "GFM" && cfg.use_level_set && !ctx.phi_list.empty());
    ctx.reinit_frequency = (cfg.reinit_interval > 0) ? cfg.reinit_interval : 5;
    ctx.reinit_iterations = 5;

    return ctx;
}


// [7] Run sharp-interface Euler solver path
template<int DIM>
inline void run_sharp_interface_case(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N,
    const std::vector<EOSParams>& material_params
)
{
    std::vector<Conserved<DIM>> U;
    std::vector<int> material_id;

    initialise_from_config<DIM, EOS>(
        U,
        material_id,
        cfg,
        N
    );

    InitialLevelSetData<DIM> ls_data{};

    if (cfg.interface_method == "GFM") {
        ls_data = initialise_phi_data_from_regions<DIM>(cfg, N);
    }

    SolverContext<DIM> ctx = build_solver_context<DIM>(
        cfg,
        N,
        material_id,
        material_params,
        ls_data
    );

    // [7.1] Make material assignment consistent with initial level sets
    if (ctx.reassign_material_from_phi && !ctx.phi_list.empty()) {
        assign_material_ids_from_phi<DIM>(
            ctx.phi_list,
            ctx.phi_material_ids,
            ctx.background_material_id,
            ctx.material_id,
            ctx.level_set_grid
        );
    }

    double time = 0.0;
    int step = 0;

    while (time < cfg.tfinal - 1e-14) {
        ctx.dt_max = cfg.tfinal - time;

        StepResult<DIM> result = advance_one_step<DIM, EOS>(U, ctx);

        if (result.dt <= 0.0) {
            throw std::runtime_error("Non-positive timestep");
        }

        U = result.U_new;
        ctx.phi_list = result.phi_list_new;

        if (ctx.reassign_material_from_phi && !ctx.phi_list.empty()) {
            assign_material_ids_from_phi<DIM>(
                ctx.phi_list,
                ctx.phi_material_ids,
                ctx.background_material_id,
                ctx.material_id,
                ctx.level_set_grid
            );
        }

        if (ctx.reinit_enabled && !ctx.phi_list.empty() &&
            step > 0 && step % ctx.reinit_frequency == 0) {
            for (int k = 0; k < ctx.n_interfaces(); ++k) {
                ctx.phi_list[k] = reinitialise_phi<DIM>(
                    ctx.phi_list[k],
                    ctx.level_set_grid,
                    ctx.reinit_iterations
                );
            }

            assign_material_ids_from_phi<DIM>(
                ctx.phi_list,
                ctx.phi_material_ids,
                ctx.background_material_id,
                ctx.material_id,
                ctx.level_set_grid
            );
        }

        time += result.dt;
        ++step;

        if (step % 25 == 0 || time >= cfg.tfinal - 1e-14) {
            std::cout << "step=" << step
                      << " time=" << time
                      << " dt=" << result.dt << "\n";
        }
    }

    std::string filename = cfg.output_prefix;

    if (!cfg.output_dir.empty()) {
        filename = cfg.output_dir + "/" + filename + "/" + filename;
    }

    for (int d = 0; d < DIM; ++d) {
        filename += "_N" + std::to_string(N[d]);
    }

    filename += ".csv";

    std::cout << "Writing to: " << filename << "\n";

    std::filesystem::path p(filename);
    ensure_directory(p.parent_path().string());

    write_csv<DIM, EOS>(
        filename,
        U,
        ctx.material_id,
        N,
        cfg.domain_min,
        cfg.domain_max,
        material_params
    );

    std::cout << "Written: " << filename << "\n";

    #if APP_DIM == 1
    if (cfg.exact_riemann) {
        std::vector<ExactState<1>> exact;

        std::array<int, 1> N_exact;
        N_exact[0] = 2000;

        compute_exact_solution<EOS>(
            exact,
            cfg,
            N_exact,
            cfg.tfinal
        );

        std::string exact_file = cfg.output_prefix + "_exact";

        if (!cfg.output_dir.empty()) {
            exact_file = cfg.output_dir + "/" + cfg.output_prefix + "/" + exact_file;
        }

        exact_file += "_N" + std::to_string(N_exact[0]);
        exact_file += ".csv";

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
    #endif
}


// [8] Run diffuse-interface solver path
template<int DIM>
inline void run_dim_case(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N,
    const std::vector<EOSParams>& material_params
)
{
    (void)cfg;
    (void)N;
    (void)material_params;

    throw std::runtime_error(
        "DIM mode selected, but diffuse-interface solver path is not implemented in this driver yet"
    );
}


// [9] MAIN
int main(int argc, char** argv)
{
    try {
        if (argc < 2) {
            throw std::runtime_error("Usage: ./multimaterial_main <config_file>");
        }

        const std::string config_file = argv[1];

        const Config<DIM_> cfg = load_config<DIM_>(config_file);

        std::cout << "Regions parsed: " << cfg.regions.size() << "\n";
        std::cout << "Interface method: " << cfg.interface_method << "\n";

        const auto material_params = build_material_params(cfg.materials);

        for (const auto& N : cfg.N_list) {
            std::cout << "Running N = ";
            for (int d = 0; d < DIM_; ++d) {
                std::cout << N[d] << " ";
            }
            std::cout << "\n";

            if (cfg.interface_method == "GFM" || cfg.interface_method == "SM") {
                run_sharp_interface_case<DIM_>(cfg, N, material_params);
            }
            else if (cfg.interface_method == "DIM") {
                run_dim_case<DIM_>(cfg, N, material_params);
            }
            else {
                throw std::runtime_error("Unknown interface_method: " + cfg.interface_method);
            }
        }

        std::cout << "Finished.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}



