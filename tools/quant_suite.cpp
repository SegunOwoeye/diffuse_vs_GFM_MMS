#include "quant/cli.hpp"

int main(int argc, char** argv)
{
    try {
        const quant::Args args = quant::parse_args(argc, argv);
        const auto runs = quant::build_runs(args);
        fs::create_directories(args.result_root / "logs");
        fs::create_directories(args.result_root / "raw");
        fs::create_directories(args.result_root / "runs");
        fs::create_directories(args.result_root / "figures");
        quant::write_manifest(args.result_root / "manifest.json", runs);

        std::cout << "{\n"
                  << "  \"result_root\": \"" << quant::json_escape(args.result_root.string()) << "\",\n"
                  << "  \"dry_run\": " << (args.dry_run ? "true" : "false") << ",\n"
                  << "  \"run_count\": " << runs.size() << "\n"
                  << "}\n";
        for (const auto& run : runs) {
            std::cout << "[run] " << run.run_id << " "
                      << run.case_def.config_path.string() << " N="
                      << quant::resolution_label(run.resolution) << " OMP="
                      << run.omp_threads << " repeat="
                      << run.repeat_id << " warmup="
                      << (run.warmup ? "true" : "false")
                      << " benchmark_mode=" << run.benchmark_mode << "\n";
        }

        std::vector<bool> successes;
        for (const auto& run : runs) {
            if (args.collect_only) {
                const fs::path metadata = args.result_root / "runs" / run.run_id / "metadata.json";
                std::string text = fs::exists(metadata) ? quant::read_file(metadata) : "";
                successes.push_back(text.find("\"success\": true") != std::string::npos);
            }
            else {
                successes.push_back(quant::run_one(run, args));
            }
        }
        if (!args.dry_run) {
            quant::collect(args.result_root, runs, successes);
        }
        return std::all_of(successes.begin(), successes.end(), [](bool ok) { return ok; }) ? 0 : 1;
    }
    catch (const std::exception& exc) {
        std::cerr << "quant_suite error: " << exc.what() << "\n";
        return 1;
    }
}
