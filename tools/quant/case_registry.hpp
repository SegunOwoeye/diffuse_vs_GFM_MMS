#pragma once

// Header-only implementation units for the quantitative validation runner.
// Case catalogue and expansion of presets, methods, resolutions, scaling, and repeats.

#include "utils.hpp"

namespace quant {

bool contains_method(const std::vector<std::string>& methods, const std::string& method)
{
    if (methods.empty()) {
        return true;
    }
    for (const auto& item : methods) {
        if (method_normalized(item) == method) {
            return true;
        }
    }
    return false;
}

std::vector<CaseDef> case_registry()
{
    const std::vector<std::string> sm1 = {
        "g++", "-std=c++17", "-O2", "-fopenmp", "-I.", "-DAPP_DIM=1",
        "src/app/sm_main.cpp", "-o", "sm_main_1d"
    };
    const std::vector<std::string> sm2 = {
        "g++", "-std=c++17", "-O2", "-fopenmp", "-I.", "-DAPP_DIM=2",
        "src/app/sm_main.cpp", "-o", "sm_main_2d"
    };
    const std::vector<std::string> sm3 = {
        "g++", "-std=c++17", "-O2", "-fopenmp", "-I.", "-DAPP_DIM=3",
        "src/app/sm_main.cpp", "-o", "sm_main_3d"
    };
    const std::vector<std::string> mm1 = {
        "g++", "-std=c++17", "-O2", "-fopenmp", "-I.", "-DAPP_DIM=1",
        "src/app/multimaterial_main.cpp", "-o", "mm_main_1d"
    };
    const std::vector<std::string> mm2 = {
        "g++", "-std=c++17", "-O2", "-fopenmp", "-I.", "-DAPP_DIM=2",
        "src/app/multimaterial_main.cpp", "-o", "mm_main_2d"
    };

    std::vector<CaseDef> cases;
    for (int i = 1; i <= 5; ++i) {
        cases.push_back({
            "toro_1d", "test" + std::to_string(i), "toro" + std::to_string(i),
            "common", 1, "sm_1d", "sm_main_1d", sm1,
            "configs/toro/test" + std::to_string(i) + ".txt",
            {{100}, {200}, {400}, {800}}
        });
    }
    cases.push_back({
        "toro_explosion_2d", "explosion1", "explosion1", "common", 2,
        "sm_2d", "sm_main_2d", sm2, "configs/toro/explosion1.txt",
        {{100, 100}, {200, 200}, {400, 400}}
    });
    cases.push_back({
        "toro_explosion_3d", "explosion2", "explosion2", "common", 3,
        "sm_3d", "sm_main_3d", sm3, "configs/toro/explosion2.txt",
        {{100, 100, 100}, {200, 200, 200}, {400, 400, 400}}
    });

    const std::vector<std::string> labels = {"FedkiwA", "FedkiwB", "FedkiwC", "FedkiwD1", "FedkiwD2"};
    for (int i = 1; i <= 5; ++i) {
        cases.push_back({
            "fedkiw_1d", "test" + std::to_string(i), labels[i - 1], "SIM", 1,
            "mm_1d", "mm_main_1d", mm1,
            "configs/GFM/MM_1D_validation/test" + std::to_string(i) + ".txt",
            {{100}, {200}, {400}, {800}}
        });
        cases.push_back({
            "fedkiw_1d", "test" + std::to_string(i), labels[i - 1], "DIM", 1,
            "mm_1d", "mm_main_1d", mm1,
            "configs/DIM/MM_1D_validation/test" + std::to_string(i) + ".txt",
            {{100}, {200}, {400}, {800}}
        });
        cases.push_back({
            "fedkiw_2d_planar", "test" + std::to_string(i), labels[i - 1], "SIM", 2,
            "mm_2d", "mm_main_2d", mm2,
            "configs/GFM/MM_2D_validation/test" + std::to_string(i) + ".txt",
            {{100, 100}, {200, 200}, {400, 400}}
        });
        cases.push_back({
            "fedkiw_2d_planar", "test" + std::to_string(i), labels[i - 1], "DIM", 2,
            "mm_2d", "mm_main_2d", mm2,
            "configs/DIM/MM_2D_validation/test" + std::to_string(i) + ".txt",
            {{100, 100}, {200, 200}, {400, 400}}
        });
    }

    cases.push_back({
        "shock_bubble_2d", "test6", "helium_bubble_2d", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/GFM/MM_2D_validation/test6.txt",
        {{1300, 178}}
    });
    cases.push_back({
        "shock_bubble_2d", "test6", "helium_bubble_2d", "DIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/DIM/MM_2D_validation/test6.txt",
        {{1300, 178}}
    });

    return cases;
}

std::set<std::string> groups_for_case_alias(const std::vector<std::string>& aliases)
{
    if (aliases.empty()) {
        return {"toro_1d", "fedkiw_1d"};
    }

    std::set<std::string> groups;
    for (const auto& raw : aliases) {
        const std::string alias = lower(raw);
        if (alias == "toro") {
            groups.insert("toro_1d");
            groups.insert("toro_explosion_2d");
            groups.insert("toro_explosion_3d");
        }
        else if (alias == "toro_no3d") {
            groups.insert("toro_1d");
            groups.insert("toro_explosion_2d");
        }
        else if (alias == "toro_1d") groups.insert("toro_1d");
        else if (alias == "explosion" || alias == "explosion2d") groups.insert("toro_explosion_2d");
        else if (alias == "explosion3d" || alias == "toro_3d") groups.insert("toro_explosion_3d");
        else if (alias == "fedkiw" || alias == "fedkiw_1d") groups.insert("fedkiw_1d");
        else if (alias == "planar" || alias == "fedkiw_2d") groups.insert("fedkiw_2d_planar");
        else if (alias == "shock_bubble" || alias == "bubble") groups.insert("shock_bubble_2d");
    }
    return groups;
}

bool is_specific_case_filter(const std::string& value)
{
    const std::string alias = lower(value);
    static const std::set<std::string> broad = {
        "toro", "toro_no3d", "toro_1d", "explosion", "explosion2d", "explosion3d",
        "toro_3d", "fedkiw", "fedkiw_1d", "planar", "fedkiw_2d", "shock_bubble", "bubble"
    };
    return !broad.count(alias);
}

bool case_allowed(const CaseDef& def, const Args& args, const std::set<std::string>& groups)
{
    if (!groups.empty() && !groups.count(def.group)) {
        return false;
    }
    if (args.cases.empty() && args.preset == "quick") {
        if (def.name != "test1" && def.name != "test5") {
            return false;
        }
    }
    bool has_specific = false;
    bool matched = false;
    for (const auto& item : args.cases) {
        if (!is_specific_case_filter(item)) {
            continue;
        }
        has_specific = true;
        const std::string value = lower(item);
        if (lower(def.name) == value || lower(def.label) == value) {
            matched = true;
        }
    }
    return !has_specific || matched;
}

std::vector<std::vector<int>> preset_resolutions(const Args& args, const CaseDef& def)
{
    if (!args.resolutions.empty()) {
        std::vector<std::vector<int>> out;
        for (auto resolution : args.resolutions) {
            if (resolution.size() == 1 && def.dimension > 1) {
                resolution = std::vector<int>(def.dimension, resolution.front());
            }
            out.push_back(resolution);
        }
        return out;
    }

    if (args.preset == "quick") {
        if (!args.cases.empty()) {
            return def.default_resolutions;
        }
        if (def.dimension == 1) {
            return {{100}, {200}};
        }
        if (def.group == "shock_bubble_2d") {
            return {};
        }
        return {{100, 100}};
    }
    return def.default_resolutions;
}

std::string method_prefix(const std::string& method)
{
    if (method == "DIM") return "dim";
    if (method == "SIM") return "gfm";
    return "common";
}

std::string sanitize_value(std::string value)
{
    for (char& c : value) {
        if (c == '.') c = 'p';
        else if (c == '-') c = 'm';
        else if (c == '=' || c == ';' || c == ' ') c = '_';
    }
    return value;
}

std::string make_run_id(
    const CaseDef& def,
    const std::vector<int>& resolution,
    int omp_threads,
    const std::string& parameter_name = "",
    const std::string& parameter_value = ""
)
{
    std::ostringstream out;
    out << def.group << "__" << def.name << "__" << def.label << "__"
        << def.method << "__" << resolution_label(resolution);
    if (parameter_name == "omp_threads") {
        out << "__omp" << omp_threads;
    }
    if (!parameter_name.empty()) {
        out << "__" << parameter_name << "_" << sanitize_value(parameter_value);
    }
    return out.str();
}

std::vector<RunSpec> build_normal_runs(const Args& args)
{
    std::vector<RunSpec> runs;
    const auto cases = case_registry();
    const auto groups = groups_for_case_alias(args.cases);

    for (const auto& def : cases) {
        if (!case_allowed(def, args, groups) || !contains_method(args.methods, def.method)) {
            continue;
        }
        for (const auto& resolution : preset_resolutions(args, def)) {
            RunSpec run;
            run.case_def = def;
            run.resolution = resolution;
            run.omp_threads = args.omp_threads;
            run.output_prefix =
                "quant_" + method_prefix(def.method) + "_" + def.label + "_" + resolution_label(resolution);
            if (def.method == "common") {
                run.output_prefix = "quant_" + def.label + "_" + resolution_label(resolution);
            }
            run.run_id = make_run_id(def, resolution, args.omp_threads);
            runs.push_back(run);
        }
    }
    return runs;
}

std::optional<CaseDef> find_case(const std::string& group, const std::string& name, const std::string& method)
{
    for (const auto& def : case_registry()) {
        if (def.group == group && def.name == name && def.method == method) {
            return def;
        }
    }
    return std::nullopt;
}

std::vector<RunSpec> build_sensitivity_runs(const Args& args)
{
    std::vector<RunSpec> runs;
    if (args.sensitivity == "dim_epsilon") {
        const auto def = find_case("fedkiw_1d", "test5", "DIM").value();
        for (double value : {0.005, 0.01, 0.02, 0.04}) {
            RunSpec run;
            run.case_def = def;
            run.resolution = {400};
            run.omp_threads = args.omp_threads;
            run.sensitivity = args.sensitivity;
            run.parameter_name = "epsilon_alpha";
            std::ostringstream text;
            text << value;
            run.parameter_value = text.str();
            run.overrides["interface_thickness"] = run.parameter_value;
            run.output_prefix = "quant_dim_FedkiwD2_epsilon_alpha_" + sanitize_value(run.parameter_value);
            run.run_id = make_run_id(def, run.resolution, args.omp_threads, run.parameter_name, run.parameter_value);
            runs.push_back(run);
        }
    }
    else if (args.sensitivity == "sim_reinit") {
        const auto def = find_case("fedkiw_1d", "test5", "SIM").value();
        for (int interval : {1, 5, 10, 20}) {
            for (int iterations : {2, 5, 10}) {
                RunSpec run;
                run.case_def = def;
                run.resolution = {400};
                run.omp_threads = args.omp_threads;
                run.sensitivity = args.sensitivity;
                run.parameter_name = "reinit";
                run.parameter_value =
                    "reinit_interval=" + std::to_string(interval) +
                    ";reinit_iterations=" + std::to_string(iterations);
                run.overrides["reinit_interval"] = std::to_string(interval);
                run.overrides["reinit_iterations"] = std::to_string(iterations);
                run.output_prefix =
                    "quant_gfm_FedkiwD2_reinit_" + std::to_string(interval) + "_" + std::to_string(iterations);
                run.run_id = make_run_id(def, run.resolution, args.omp_threads, run.parameter_name, run.parameter_value);
                runs.push_back(run);
            }
        }
    }
    return runs;
}

std::vector<RunSpec> build_scaling_runs(const Args& args)
{
    std::vector<RunSpec> runs;
    if (args.scaling != "openmp_threads") {
        return runs;
    }

    const std::vector<std::tuple<std::string, std::string, std::string, std::vector<int>>> targets = {
        {"fedkiw_1d", "test5", "DIM", {400}},
        {"fedkiw_1d", "test5", "SIM", {400}},
        {"fedkiw_2d_planar", "test5", "DIM", {200, 200}},
        {"fedkiw_2d_planar", "test5", "SIM", {200, 200}},
    };
    for (const auto& target : targets) {
        const auto def = find_case(std::get<0>(target), std::get<1>(target), std::get<2>(target)).value();
        for (int threads : {1, 2, 4, 6}) {
            RunSpec run;
            run.case_def = def;
            run.resolution = std::get<3>(target);
            run.omp_threads = threads;
            run.scaling = args.scaling;
            run.parameter_name = "omp_threads";
            run.parameter_value = std::to_string(threads);
            run.output_prefix =
                "quant_scaling_" + method_prefix(def.method) + "_" + def.label +
                "_" + resolution_label(run.resolution) + "_omp" + std::to_string(threads);
            run.run_id = make_run_id(def, run.resolution, threads, run.parameter_name, run.parameter_value);
            runs.push_back(run);
        }
    }
    return runs;
}

std::vector<RunSpec> apply_benchmark_repeats(
    const std::vector<RunSpec>& base_runs,
    const Args& args
)
{
    const int warmups = std::max(0, args.benchmark_warmups);
    const int repeats = std::max(1, args.benchmark_repeats);
    const int total = warmups + repeats;
    if (total <= 1 && args.benchmark_mode == "standard") {
        std::vector<RunSpec> out = base_runs;
        for (auto& run : out) {
            run.repeat_id = 0;
            run.warmup = false;
            run.benchmark_mode = args.benchmark_mode;
        }
        return out;
    }

    std::vector<RunSpec> out;
    for (const auto& base : base_runs) {
        for (int repeat = 0; repeat < total; ++repeat) {
            RunSpec run = base;
            run.repeat_id = repeat;
            run.warmup = repeat < warmups;
            run.benchmark_mode = args.benchmark_mode;
            run.run_id += "__repeat" + std::to_string(repeat);
            if (run.warmup) {
                run.run_id += "__warmup";
            }
            run.output_prefix += "_repeat" + std::to_string(repeat);
            if (run.warmup) {
                run.output_prefix += "_warmup";
            }
            out.push_back(run);
        }
    }
    return out;
}

std::vector<RunSpec> build_runs(const Args& args)
{
    std::vector<RunSpec> runs;
    if (!args.sensitivity.empty()) {
        runs = build_sensitivity_runs(args);
    }
    else if (!args.scaling.empty()) {
        runs = build_scaling_runs(args);
    }
    else {
        runs = build_normal_runs(args);
    }
    return apply_benchmark_repeats(runs, args);
}

} // namespace quant
