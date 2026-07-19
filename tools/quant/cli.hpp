#pragma once

// Header-only implementation units for the quantitative validation runner.
// Command-line parser for the quantitative suite executable.

#include "report_builders.hpp"

namespace quant {

inline std::vector<std::string> split_cli_list(const std::string& value)
{
    std::vector<std::string> out;
    for (const auto& part : split(value, ',')) {
        const std::string item = trim(part);
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

Args parse_args(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
            ++i;
            return argv[i];
        };
        auto collect_values = [&](std::vector<std::string>& out) {
            while (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                ++i;
                const auto values = split_cli_list(argv[i]);
                out.insert(out.end(), values.begin(), values.end());
            }
        };

        if (key == "--preset") args.preset = next();
        else if (key == "--dry-run") args.dry_run = true;
        else if (key == "--overwrite") args.overwrite = true;
        else if (key == "--resume") args.resume = true;
        else if (key == "--collect-only") args.collect_only = true;
        else if (key == "--all-core") args.all_core = true;
        else if (key == "--result-root") args.result_root = next();
        else if (key == "--case") {
            const auto values = split_cli_list(next());
            args.cases.insert(args.cases.end(), values.begin(), values.end());
        }
        else if (key == "--cases") collect_values(args.cases);
        else if (key == "--method") {
            const auto values = split_cli_list(next());
            args.methods.insert(args.methods.end(), values.begin(), values.end());
        }
        else if (key == "--methods") collect_values(args.methods);
        else if (key == "--resolutions") {
            std::vector<std::string> values;
            collect_values(values);
            for (const auto& value : values) args.resolutions.push_back(parse_resolution(value));
        }
        else if (key == "--sensitivity") args.sensitivity = next();
        else if (key == "--scaling") args.scaling = next();
        else if (key == "--omp-threads") args.omp_threads = std::stoi(next());
        else if (key == "--scaling-threads") {
            args.scaling_threads.clear();
            for (const auto& value : split_cli_list(next())) {
                const int threads = std::stoi(value);
                if (threads <= 0) {
                    throw std::runtime_error("scaling thread counts must be positive");
                }
                args.scaling_threads.push_back(threads);
            }
            if (args.scaling_threads.empty()) {
                throw std::runtime_error("--scaling-threads requires at least one value");
            }
        }
        else if (key == "--timeout-seconds") args.timeout_seconds = std::stoi(next());
        else if (key == "--conservation-interval") args.conservation_interval = std::stoi(next());
        else if (key == "--benchmark-mode") args.benchmark_mode = next();
        else if (key == "--benchmark-repeats") args.benchmark_repeats = std::stoi(next());
        else if (key == "--benchmark-warmups") args.benchmark_warmups = std::stoi(next());
        else if (key == "--help" || key == "-h") {
            std::cout
                << "Usage: tools/quant_suite [options]\n"
                << "  --preset quick|full\n"
                << "  --dry-run\n"
                << "  --resume (skip runs with successful metadata)\n"
                << "  --collect-only (rebuild summaries from existing run output)\n"
                << "  --all-core (Toro1, FedkiwD2 1D, oblique FedkiwD2, shock bubble)\n"
                << "  --case/--cases toro|toro_1d|explosion2d|explosion3d|fedkiw|planar|oblique|shock_bubble|bubble_reinit|water_air_bubble|gorsse_tc9|he2023_three_material|he2023_three_material_1d|he2023_triple_point|gorsse_tc9_3d|bubble_zero_velocity|bubble_zero_velocity_physical_flow|bubble_zero_velocity_input_mean_star|bubble_zero_velocity_zero_star|bubble_static|bubble3d [case]\n"
                << "  --method/--methods DIM SIM common\n"
                << "  --resolutions 100 200 400, or 100,200,400, or 325x45,650x89\n"
                << "  --sensitivity dim_interface_thickness|dim_interface_thickness_bubble|sim_weno2_reinit_interval|sim_weno2_reinit_interval_bubble\n"
                << "      dim_interface_thickness varies interface_thickness at fixed interface_sharpness_alpha=2\n"
                << "      sim_weno2_reinit_interval varies reinit_interval with WENO2 spatial derivatives and SSP-RK2 pseudo-time integration\n"
                << "      dim_epsilon, dim_alpha, and sim_reinit legacy modes remain supported\n"
                << "      full and legacy sim_reinit modes include reference extras\n"
                << "      *_bubble modes run helium bubble only\n"
                << "  --scaling openmp_threads (2D helium bubble DIM/SIM at 1,2,4,8,16,32 threads)\n"
                << "  --scaling-threads 1,2,4,8,16,32\n"
                << "  --omp-threads N\n"
                << "  --benchmark-mode NAME (default standard; use clean for controlled timing)\n"
                << "  --benchmark-warmups N (default 0)\n"
                << "  --benchmark-repeats N measured repeats (default 1)\n"
                << "  --timeout-seconds N (0 disables timeout; default disabled)\n"
                << "  --conservation-interval N (default 25)\n"
                << "  --result-root results/quantitative/name\n";
            std::exit(0);
        }
        else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.result_root.empty()) {
        args.result_root = fs::path("results") / "quantitative" / now_stamp();
    }
    return args;
}

} // namespace quant
