#include "quant/cli.hpp"

namespace {

std::string format_duration(double seconds)
{
    if (!std::isfinite(seconds) || seconds < 0.0) {
        return "estimating";
    }

    const long long rounded = static_cast<long long>(seconds + 0.5);
    const long long hours = rounded / 3600;
    const long long minutes = (rounded % 3600) / 60;
    const long long secs = rounded % 60;

    std::ostringstream out;
    if (hours > 0) {
        out << hours << "h ";
    }
    if (hours > 0 || minutes > 0) {
        out << minutes << "m ";
    }
    out << secs << "s";
    return out.str();
}

std::string progress_bar(std::size_t completed, std::size_t total)
{
    constexpr std::size_t width = 28;
    if (total == 0) {
        return "[" + std::string(width, '-') + "]";
    }

    const std::size_t filled = std::min(width, completed * width / total);
    return "[" + std::string(filled, '#') + std::string(width - filled, '-') + "]";
}

void print_progress(
    std::size_t completed,
    std::size_t total,
    const std::chrono::steady_clock::time_point& start,
    const std::string& label
)
{
    const auto now = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(now - start).count();
    const double average =
        completed > 0 ? elapsed / static_cast<double>(completed) : std::numeric_limits<double>::quiet_NaN();
    const double remaining =
        completed > 0 && total >= completed
            ? average * static_cast<double>(total - completed)
            : std::numeric_limits<double>::quiet_NaN();
    const double estimated_total =
        completed > 0 ? elapsed + remaining : std::numeric_limits<double>::quiet_NaN();

    std::cout << "[progress] "
              << progress_bar(completed, total) << " "
              << completed << "/" << total
              << " elapsed=" << format_duration(elapsed)
              << " eta=" << format_duration(remaining)
              << " est_total=" << format_duration(estimated_total);
    if (!label.empty()) {
        std::cout << " " << label;
    }
    std::cout << "\n";
}

} // namespace

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
        const auto suite_start = std::chrono::steady_clock::now();
        print_progress(0, runs.size(), suite_start, "starting");
        for (std::size_t index = 0; index < runs.size(); ++index) {
            const auto& run = runs[index];
            print_progress(index, runs.size(), suite_start, "running=" + run.run_id);
            if (args.collect_only) {
                const fs::path metadata = args.result_root / "runs" / run.run_id / "metadata.json";
                std::string text = fs::exists(metadata) ? quant::read_file(metadata) : "";
                successes.push_back(text.find("\"success\": true") != std::string::npos);
            }
            else {
                successes.push_back(quant::run_one(run, args));
            }
            print_progress(index + 1, runs.size(), suite_start, "completed=" + run.run_id);
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
