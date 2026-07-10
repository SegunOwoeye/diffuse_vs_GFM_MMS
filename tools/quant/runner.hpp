#pragma once

// Header-only implementation units for the quantitative validation runner.
// Execution layer: generated configs, build commands, per-run metadata, and manifests.

#include "case_registry.hpp"

namespace quant {

std::string generated_config_text(const RunSpec& run, const fs::path& raw_dir)
{
    std::stringstream input(read_file(run.case_def.config_path));
    std::ostringstream output;
    std::string line;
    bool wrote_n = false;
    std::set<std::string> wrote;

    auto replacement_for = [&](const std::string& key) -> std::optional<std::string> {
        if (key == "N") return render_resolution(run.resolution);
        if (key == "output_dir") return raw_dir.generic_string();
        if (key == "output_prefix") return run.output_prefix;
        const auto it = run.overrides.find(key);
        if (it != run.overrides.end()) return it->second;
        return std::nullopt;
    };

    while (std::getline(input, line)) {
        const std::string stripped = trim(line);
        const auto pos = stripped.find('=');
        const std::string key = (pos == std::string::npos) ? "" : trim(stripped.substr(0, pos));
        if (key == "N") {
            if (!wrote_n) {
                output << "N = " << render_resolution(run.resolution) << "\n";
                wrote_n = true;
                wrote.insert("N");
            }
            continue;
        }
        const auto replacement = replacement_for(key);
        if (replacement.has_value()) {
            output << key << " = " << replacement.value() << "\n";
            wrote.insert(key);
            continue;
        }
        output << line << "\n";
    }

    for (const auto& key : {"N", "output_dir", "output_prefix"}) {
        if (!wrote.count(key)) {
            output << key << " = " << replacement_for(key).value() << "\n";
        }
    }
    for (const auto& item : run.overrides) {
        if (!wrote.count(item.first)) {
            output << item.first << " = " << item.second << "\n";
        }
    }
    return output.str();
}

void ensure_executable(const CaseDef& def)
{
    static std::set<std::string> compiled_this_invocation;

    if (compiled_this_invocation.count(def.executable_path)) {
        return;
    }

    std::cout << "[compile] " << def.executable_path << "\n";
    const int rc = std::system(join_command(def.compile_command).c_str());
    if (rc != 0) {
        throw std::runtime_error("Compile failed for " + def.executable_path);
    }
    compiled_this_invocation.insert(def.executable_path);
}

std::string executable_for(const CaseDef& def)
{
#ifdef _WIN32
    if (fs::exists(def.executable_path + ".exe")) {
        return def.executable_path + ".exe";
    }
#endif
    return "./" + def.executable_path;
}


void write_manifest(const fs::path& path, const std::vector<RunSpec>& runs)
{
    std::ostringstream json;
    json << "{\n  \"run_count\": " << runs.size() << ",\n  \"runs\": [\n";
    for (std::size_t i = 0; i < runs.size(); ++i) {
        const auto& run = runs[i];
        if (i > 0) json << ",\n";
        json << "    {\"run_id\": \"" << json_escape(run.run_id)
             << "\", \"group\": \"" << run.case_def.group
             << "\", \"case\": \"" << run.case_def.name
             << "\", \"method\": \"" << run.case_def.method
             << "\", \"resolution\": \"" << resolution_label(run.resolution)
             << "\", \"omp_threads\": " << run.omp_threads
             << ", \"repeat_id\": " << run.repeat_id
             << ", \"warmup\": " << (run.warmup ? "true" : "false")
             << ", \"benchmark_mode\": \"" << json_escape(run.benchmark_mode) << "\"}";
    }
    json << "\n  ]\n}\n";
    write_file(path, json.str());
}

std::map<std::string, std::string> base_row(const RunSpec& run, bool success)
{
    return {
        {"run_id", run.run_id},
        {"group", run.case_def.group},
        {"case", run.case_def.name},
        {"case_label", run.case_def.label},
        {"method", run.case_def.method},
        {"resolution", resolution_label(run.resolution)},
        {"N", resolution_label(run.resolution)},
        {"omp_threads", std::to_string(run.omp_threads)},
        {"repeat_id", std::to_string(run.repeat_id)},
        {"warmup", run.warmup ? "true" : "false"},
        {"benchmark_mode", run.benchmark_mode},
        {"success", success ? "true" : "false"},
        {"sensitivity", run.sensitivity},
        {"scaling", run.scaling},
        {"parameter_name", run.parameter_name},
        {"parameter_value", run.parameter_value},
    };
}

bool run_one(const RunSpec& run, const Args& args)
{
    const fs::path result_root = args.result_root;
    const bool dry_run = args.dry_run;
    const bool overwrite = args.overwrite;
    const fs::path run_dir = result_root / "runs" / run.run_id;
    const fs::path raw_dir = result_root / "raw" / run.run_id;
    const fs::path log_dir = result_root / "logs" / run.run_id;
    if (overwrite) {
        fs::remove_all(run_dir);
        fs::remove_all(raw_dir);
        fs::remove_all(log_dir);
    }
    fs::create_directories(run_dir);
    fs::create_directories(raw_dir);
    fs::create_directories(log_dir);

    const fs::path generated_config = run_dir / "generated_config.txt";
    write_file(generated_config, generated_config_text(run, raw_dir));

    const fs::path case_dir = raw_dir / run.output_prefix;
    const fs::path solution = case_dir / (run.output_prefix + resolution_suffix(run.resolution) + ".csv");
    const fs::path runtime = case_dir / (run.output_prefix + resolution_suffix(run.resolution) + "_runtime.txt");
    const fs::path conservation = case_dir / (run.output_prefix + resolution_suffix(run.resolution) + "_conservation.csv");

    auto write_metadata = [&](bool success, int rc, double wall, const std::string& command, const std::string& status) {
        std::ostringstream metadata;
        metadata << "{\n"
                 << "  \"run_id\": \"" << json_escape(run.run_id) << "\",\n"
                 << "  \"case_name\": \"" << run.case_def.name << "\",\n"
                 << "  \"case_label\": \"" << run.case_def.label << "\",\n"
                 << "  \"group\": \"" << run.case_def.group << "\",\n"
                 << "  \"method\": \"" << run.case_def.method << "\",\n"
                 << "  \"resolution\": \"" << resolution_label(run.resolution) << "\",\n"
                 << "  \"omp_threads\": " << run.omp_threads << ",\n"
                 << "  \"repeat_id\": " << run.repeat_id << ",\n"
                 << "  \"warmup\": " << (run.warmup ? "true" : "false") << ",\n"
                 << "  \"benchmark_mode\": \"" << json_escape(run.benchmark_mode) << "\",\n"
                 << "  \"omp_schedule\": \"dynamic\",\n"
                 << "  \"conservation_interval\": " << args.conservation_interval << ",\n"
                 << "  \"timeout_seconds\": " << args.timeout_seconds << ",\n"
                 << "  \"sensitivity\": \"" << run.sensitivity << "\",\n"
                 << "  \"scaling\": \"" << run.scaling << "\",\n"
                 << "  \"parameter_name\": \"" << run.parameter_name << "\",\n"
                 << "  \"parameter_value\": \"" << run.parameter_value << "\",\n"
                 << "  \"command\": \"" << json_escape(command) << "\",\n"
                 << "  \"status\": \"" << status << "\",\n"
                 << "  \"success\": " << (success ? "true" : "false") << ",\n"
                 << "  \"returncode\": " << rc << ",\n"
                 << "  \"runtime_seconds\": " << wall << ",\n"
                 << "  \"solution\": \"" << json_escape(solution.string()) << "\",\n"
                 << "  \"runtime_report\": \"" << json_escape(runtime.string()) << "\",\n"
                 << "  \"conservation_report\": \"" << json_escape(conservation.string()) << "\"\n"
                 << "}\n";
        write_file(run_dir / "metadata.json", metadata.str());
    };

    if (dry_run) {
        write_metadata(true, 0, 0.0, "", "dry_run");
        return true;
    }

    ensure_executable(run.case_def);
    std::string command =
        "env OMP_NUM_THREADS=" + std::to_string(run.omp_threads) +
        " OMP_SCHEDULE=dynamic SOLVER_CONSERVATION=1 SOLVER_CONSERVATION_INTERVAL=" +
        std::to_string(args.conservation_interval) + " " +
        shell_quote(executable_for(run.case_def)) + " " +
        shell_quote(generated_config.string()) +
        " > " + shell_quote((log_dir / "stdout.txt").string()) +
        " 2> " + shell_quote((log_dir / "stderr.txt").string());
    if (args.timeout_seconds > 0) {
        command = "timeout " + std::to_string(args.timeout_seconds) + "s " + command;
    }

    write_metadata(false, -1, 0.0, command, "started");

    const auto start = std::chrono::steady_clock::now();
    const int rc = std::system(command.c_str());
    const auto end = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(end - start).count();
    const bool success = (rc == 0);
    std::string status = success ? "completed" : "failed";
    if (args.timeout_seconds > 0 && rc != 0 && wall >= static_cast<double>(args.timeout_seconds) - 1.0) {
        status = "timeout";
    }
    write_metadata(success, rc, wall, command, status);
    std::cout << "[done] " << run.run_id << " status=" << status << " seconds=" << wall << "\n";
    return success;
}

} // namespace quant
