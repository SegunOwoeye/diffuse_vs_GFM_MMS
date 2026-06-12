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
                << "  --collect-only (rebuild summaries from existing run output)\n"
                << "  --all-core (Toro1, FedkiwD2 1D, oblique FedkiwD2, shock bubble)\n"
                << "  --case/--cases toro|toro_1d|explosion2d|explosion3d|fedkiw|planar|oblique|shock_bubble|bubble3d [case]\n"
                << "  --method/--methods DIM SIM common (comma lists accepted)\n"
                << "  --resolutions 100 200 400, or 100,200,400, or 325x45,650x89\n"
                << "  --sensitivity dim_epsilon|sim_reinit (DIM: 1D D2 + bubble; SIM: oblique + bubble)\n"
                << "  --scaling openmp_threads (bubble DIM/SIM at 1,2,4,8,16 threads)\n"
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
