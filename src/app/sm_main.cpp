#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include "src/app/dimension.hpp"
#include "src/io/config.hpp"
#include "src/io/config_loader.hpp"
#include "src/setup/initial_conditions.hpp"
#include "src/euler/solver/advance_step.hpp"
#include "src/euler/solver/solver_context.hpp"
#include "src/io/write_csv.hpp"
#include "src/io/compute_exact_solution.hpp"
#include "src/io/write_exact_csv.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"

// Single Material Multidimensional Compressible Euler Flow Solver


// EOS
using EOS = IdealGasEOS;



// directory helper
inline void ensure_directory(const std::string& dir)
{
    if (!dir.empty()) {
        std::filesystem::create_directories(dir);
    }
}



// build EOS params
std::vector<EOSParams> build_material_params(
    const std::vector<MaterialConfig>& materials
)
{
    std::vector<EOSParams> params(materials.size());

    for (const auto& m : materials) {

        if (m.id < 0 || m.id >= static_cast<int>(materials.size())) {
            throw std::runtime_error("Invalid material id");
        }

        if (m.type == "ideal_gas") {
            if (!m.params.count("gamma")) {
                throw std::runtime_error("Missing gamma in material");
            }
            params[m.id].gamma = m.params.at("gamma");
        }
        else {
            throw std::runtime_error("Unsupported EOS type");
        }
    }

    return params;
}



// MAIN
int main(int argc, char** argv)
{
    try {

        if (argc < 2) {
            throw std::runtime_error("Usage: ./gfm_main <config_file>");
        }

        const std::string config_file = argv[1];

        const Config<DIM_> cfg = load_config<DIM_>(config_file);
        std::cout << "Regions parsed: " << cfg.regions.size() << "\n";


        
        // [2] Materials
        const auto material_params = build_material_params(cfg.materials);

        
        // [3] Loop over resolutions
        for (const auto& N : cfg.N_list) {

            std::cout << "Running N = ";
            for (int d = 0; d < DIM_; ++d) {
                std::cout << N[d] << " ";
            }
            std::cout << "\n";

            
            // [3.1] Initialise
            std::vector<Conserved<DIM_>> U;
            std::vector<int> material_id;

            initialise_from_config<DIM_, EOS>(
                U,
                material_id,
                cfg,
                N
            );

            
            // [3.2] Solver context
            SolverContext<DIM_> ctx;

            ctx.N = N;

            for (int d = 0; d < DIM_; ++d) {
                ctx.dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / N[d];
            }

            ctx.cfl = cfg.cfl;

            ctx.material_id = material_id;
            ctx.material_params = material_params;
            ctx.background_material_id = material_id.empty() ? 0 : material_id[0];

            ctx.initialise_level_set_grid();
            ctx.initialise_boundary_conditions();

            ctx.advect_level_set = false;
            ctx.reassign_material_from_phi = false;
            ctx.reinit_enabled = false;
            ctx.reinit_iterations = 5;

            
            // [3.3] Time loop
            double time = 0.0;
            int step = 0;

            while (time < cfg.tfinal - 1e-14) {

                ctx.dt_max = cfg.tfinal - time;

                auto result = advance_one_step<DIM_, EOS>(
                    U,
                    ctx
                );

                U = result.U_new;

                time += result.dt;
                step++;

                if (result.dt <= 0.0) {
                    throw std::runtime_error("Non-positive timestep");
                }

                if (step % 25 == 0 || time >= cfg.tfinal - 1e-14) {
                    std::cout << "step=" << step
                            << " time=" << time
                            << " dt=" << result.dt << "\n";
                }
            }

            
            // [3.4] Write numerical
            std::string filename = cfg.output_prefix;

            if (!cfg.output_dir.empty()) {
                filename = cfg.output_dir + "/" + filename + "/" + filename;
            }

            for (int d = 0; d < DIM_; ++d) {
                filename += "_N" + std::to_string(N[d]);
            }

            filename += ".csv";

            std::cout << "Writing to: " << filename << "\n";

            std::filesystem::path p(filename);
            ensure_directory(p.parent_path().string());

            write_csv<DIM_, EOS>(
                filename,
                U,
                ctx.material_id,
                N,
                cfg.domain_min,
                cfg.domain_max,
                material_params
            );

            std::cout << "Written: " << filename << "\n";

            
            #if APP_DIM == 1 // Loads this codeonly for 1D simulations
            
            // [3.5] Exact solution
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

        std::cout << "Finished.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}


// Run example: ./sm_main configs/toro/test1.txt
