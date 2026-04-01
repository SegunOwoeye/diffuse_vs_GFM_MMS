#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <stdexcept>

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


// [2] Build a single level set from region layout for 2-material sharp-interface cases
template<int DIM>
inline std::vector<double> initialise_phi_from_regions(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N
)
{
    if (cfg.regions.size() != 2) {
        throw std::runtime_error(
            "initialise_phi_from_regions: sharp-interface setup currently expects exactly 2 regions"
        );
    }

    const Region<DIM>& r0 = cfg.regions[0];
    const Region<DIM>& r1 = cfg.regions[1];

    if (r0.material_id == r1.material_id) {
        throw std::runtime_error(
            "initialise_phi_from_regions: GFM/SM sharp-interface mode requires two distinct materials"
        );
    }

    int total_cells = 1;
    for (int d = 0; d < DIM; ++d) {
        if (N[d] <= 0) {
            throw std::runtime_error("initialise_phi_from_regions: invalid grid size");
        }
        total_cells *= N[d];
    }

    std::array<double, DIM> dx{};
    for (int d = 0; d < DIM; ++d) {
        dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / N[d];
    }

    std::vector<double> phi(total_cells, 1.0);
    std::array<int, DIM> idx{};

    for (int linear = 0; linear < total_cells; ++linear) {
        int tmp = linear;
        for (int d = DIM - 1; d >= 0; --d) {
            idx[d] = tmp % N[d];
            tmp /= N[d];
        }

        const auto x = compute_cell_center<DIM>(idx, cfg.domain_min, dx);

        if (point_in_region<DIM>(x, r0)) {
            phi[linear] = -1.0;
        }
        else if (point_in_region<DIM>(x, r1)) {
            phi[linear] = 1.0;
        }
        else {
            throw std::runtime_error(
                "initialise_phi_from_regions: cell not covered by either sharp-interface region"
            );
        }
    }

    return phi;
}


// [3] Build initial level-set list
template<int DIM>
inline std::vector<std::vector<double>> initialise_phi_list(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N
)
{
    std::vector<std::vector<double>> phi_list;

    if (!cfg.use_level_set) {
        return phi_list;
    }

    if (cfg.initial_condition != "regions") {
        throw std::runtime_error(
            "initialise_phi_list: current sharp-interface level set initialisation only supports region ICs"
        );
    }

    phi_list.push_back(initialise_phi_from_regions<DIM>(cfg, N));
    return phi_list;
}


// [4] Build solver context
template<int DIM>
inline SolverContext<DIM> build_solver_context(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N,
    const std::vector<int>& material_id,
    const std::vector<EOSParams>& material_params,
    const std::vector<std::vector<double>>& phi_list
)
{
    SolverContext<DIM> ctx;

    ctx.N = N;

    for (int d = 0; d < DIM; ++d) {
        ctx.dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / N[d];
    }

    ctx.cfl = cfg.cfl;
    ctx.dt_max = 1e-3;

    ctx.material_id = material_id;
    ctx.material_params = material_params;

    ctx.initialise_level_set_grid();
    ctx.phi_list = phi_list;

    ctx.reinit_enabled = cfg.use_level_set;
    ctx.reinit_frequency = (cfg.reinit_interval > 0) ? cfg.reinit_interval : 5;
    ctx.reinit_iterations = 5;

    return ctx;
}


// [5] Run sharp-interface Euler solver path
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

    std::vector<std::vector<double>> phi_list = initialise_phi_list<DIM>(cfg, N);

    SolverContext<DIM> ctx = build_solver_context<DIM>(
        cfg,
        N,
        material_id,
        material_params,
        phi_list
    );

    double time = 0.0;
    int step = 0;

    while (time < cfg.tfinal - 1e-14) {
        ctx.dt_max = cfg.tfinal - time;

        StepResult<DIM> result;

        if (cfg.interface_method == "SM") {
            if (!ctx.phi_list.empty()) {
                throw std::runtime_error("SM mode should not carry a level set");
            }

            throw std::runtime_error(
                "SM dispatch not yet separated from GFM path. "
                "Either implement advance_one_step_sm or run GFM mode only."
            );
        }
        else if (cfg.interface_method == "GFM") {
            result = advance_one_step<DIM, EOS>(U, ctx);
        }
        else {
            throw std::runtime_error("run_sharp_interface_case: invalid sharp-interface method");
        }

        if (result.dt <= 0.0) {
            throw std::runtime_error("Non-positive timestep");
        }

        U = result.U_new;
        ctx.phi_list = result.phi_list_new;

        if (!ctx.phi_list.empty()) {
            assign_material_ids_from_phi<DIM>(
                ctx.phi_list,
                ctx.material_id,
                ctx.level_set_grid
            );
        }

        if (ctx.reinit_enabled &&
            !ctx.phi_list.empty() &&
            ctx.reinit_frequency > 0 &&
            step > 0 &&
            step % ctx.reinit_frequency == 0) {
            for (int k = 0; k < ctx.n_interfaces(); ++k) {
                ctx.phi_list[k] = reinitialise_phi<DIM>(
                    ctx.phi_list[k],
                    ctx.level_set_grid,
                    ctx.reinit_iterations
                );
            }

            assign_material_ids_from_phi<DIM>(
                ctx.phi_list,
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


// [6] Run diffuse-interface solver path
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


// [7] MAIN
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


