#pragma once

// Header-only implementation units for the quantitative validation runner.
// Command-line parser for the quantitative suite executable.

#include "report_builders.hpp"

namespace quant {

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
                out.push_back(argv[i]);
            }
        };

        if (key == "--preset") args.preset = next();
        else if (key == "--dry-run") args.dry_run = true;
        else if (key == "--overwrite") args.overwrite = true;
        else if (key == "--collect-only") args.collect_only = true;
        else if (key == "--result-root") args.result_root = next();
        else if (key == "--cases") collect_values(args.cases);
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
                << "  --cases toro|toro_1d|explosion2d|explosion3d|fedkiw|planar|shock_bubble [case]\n"
                << "  --methods DIM SIM common\n"
                << "  --resolutions 100 200 400 or 200x200\n"
                << "  --sensitivity dim_epsilon|sim_reinit\n"
                << "  --scaling openmp_threads\n"
                << "  --omp-threads N\n"
                << "  --benchmark-mode NAME (default standard; use clean for controlled timing)\n"
                << "  --benchmark-warmups N (default 0)\n"
                << "  --benchmark-repeats N measured repeats (default 1)\n"
                << "  --timeout-seconds N (0 disables timeout; default 900)\n"
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
