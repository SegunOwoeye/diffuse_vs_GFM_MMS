#include <iostream>
#include <stdexcept>
#include <string>

#include "src/app/dimension.hpp"
#include "src/io/config.hpp"
#include "src/io/config_loader.hpp"
#include "src/euler/eos.hpp"

#include "src/app/material/material_builder.hpp"
#include "src/app/solver/run_sharp.hpp"
#include "src/app/solver/run_dim.hpp"

// Multimaterial / Multidimensional Compressible Euler Flow Solver

using EOS = IdealGasEOS;


// [0] Main
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
                run_sharp_interface_case<DIM_, EOS>(cfg, N, material_params);
            }
            else if (cfg.interface_method == "DIM") {
                run_dim_case<DIM_, EOS>(cfg, N, material_params);
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