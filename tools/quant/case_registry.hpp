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
    const std::vector<std::string> mm3 = {
        "g++", "-std=c++17", "-O2", "-fopenmp", "-I.", "-DAPP_DIM=3",
        "src/app/multimaterial_main.cpp", "-o", "mm_main_3d"
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
        {{50, 50, 50}, {100, 100, 100}, {200, 200, 200}}
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
        cases.push_back({
            "fedkiw_2d_oblique", "test" + std::to_string(i), labels[i - 1] + "45", "SIM", 2,
            "mm_2d", "mm_main_2d", mm2,
            "configs/GFM/MM_2D_validation/test" + std::to_string(i) + "_oblique45.txt",
            {{100, 100}, {200, 200}, {400, 400}}
        });
        cases.push_back({
            "fedkiw_2d_oblique", "test" + std::to_string(i), labels[i - 1] + "45", "DIM", 2,
            "mm_2d", "mm_main_2d", mm2,
            "configs/DIM/MM_2D_validation/test" + std::to_string(i) + "_oblique45.txt",
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
        "shock_bubble_2d_reinit", "test6_reinit5", "helium_bubble_2d_reinit5", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/GFM/bubble_collapse/comparison/helium_bubble_2d_reinit5_tvd.txt",
        {{1300, 178}}
    });
    cases.push_back({
        "shock_bubble_2d_reinit", "test6_reinit10", "helium_bubble_2d_reinit10", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/GFM/bubble_collapse/comparison/helium_bubble_2d_reinit10_tvd.txt",
        {{1300, 178}}
    });
    cases.push_back({
        "shock_bubble_2d_reinit", "test6_reinit10_iter1", "helium_bubble_2d_reinit10_iter1", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/GFM/bubble_collapse/comparison/helium_bubble_2d_reinit10_iter1_tvd.txt",
        {{1300, 178}}
    });
    cases.push_back({
        "shock_bubble_2d_reinit", "test6_reinit10_iter2", "helium_bubble_2d_reinit10_iter2", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/GFM/bubble_collapse/comparison/helium_bubble_2d_reinit10_iter2_tvd.txt",
        {{1300, 178}}
    });
    cases.push_back({
        "shock_bubble_2d_reinit", "test6_redistance10_iter2", "helium_bubble_2d_redistance10_iter2", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/GFM/bubble_collapse/comparison/helium_bubble_2d_redistance10_iter2_tvd.txt",
        {{1300, 178}}
    });
    cases.push_back({
        "water_air_bubble_2d", "case2", "practical_case2_water_air_bubble_2d", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/GFM/bubble_collapse/practical_case2_water_air_bubble_2d.txt",
        {{201, 201}, {401, 401}, {601, 601}}
    });
    cases.push_back({
        "water_air_bubble_2d", "case2", "practical_case2_water_air_bubble_2d", "DIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/DIM/bubble_collapse/practical_case2_water_air_bubble_2d.txt",
        {{201, 201}, {401, 401}, {601, 601}}
    });
    cases.push_back({
        "gorsse_tc9_water_air_bubble_2d", "tc9", "gorsse_tc9_water_air_bubble_2d", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/GFM/bubble_collapse/gorsse_tc9_water_air_bubble_2d.txt",
        {{240, 200}, {480, 400}}
    });
    cases.push_back({
        "gorsse_tc9_water_air_bubble_2d", "tc9", "gorsse_tc9_water_air_bubble_2d", "DIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/DIM/bubble_collapse/gorsse_tc9_water_air_bubble_2d.txt",
        {{240, 200}, {480, 400}}
    });
    cases.push_back({
        "he2023_three_material_1d", "three_material_1d", "he2023_three_material_1d", "SIM", 1,
        "mm_1d", "mm_main_1d", mm1,
        "configs/diagnostics/he2023_three_material_1d_gfm_lowres.txt",
        {{100}, {200}, {400}, {800}, {2000}}
    });
    cases.push_back({
        "he2023_three_material_1d", "three_material_1d", "he2023_three_material_1d", "DIM", 1,
        "mm_1d", "mm_main_1d", mm1,
        "configs/diagnostics/he2023_three_material_1d_dim_lowres.txt",
        {{100}, {200}, {400}, {800}, {2000}}
    });
    cases.push_back({
        "he2023_three_material_triple_point_2d", "triple_point", "he2023_three_material_triple_point_2d", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/diagnostics/he2023_three_material_triple_point_2d_gfm_lowres_mcrs.txt",
        {{1400, 600}}
    });
    cases.push_back({
        "he2023_three_material_triple_point_2d", "triple_point", "he2023_three_material_triple_point_2d", "DIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/diagnostics/he2023_three_material_triple_point_2d_dim_lowres.txt",
        {{1400, 600}}
    });
    cases.push_back({
        "applsci_three_material_translation_2d", "three_material_translation", "applsci_2021_three_material_translation", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/GFM/MM_2D_validation/applsci_2021_three_material_translation.txt",
        {{200, 100}, {400, 200}}
    });
    cases.push_back({
        "applsci_three_material_translation_2d", "three_material_translation", "applsci_2021_three_material_translation", "DIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/DIM/MM_2D_validation/applsci_2021_three_material_translation.txt",
        {{200, 100}, {400, 200}}
    });
    cases.push_back({
        "gorsse_tc9_water_air_bubble_3d", "tc9", "gorsse_tc9_water_air_bubble_3d", "SIM", 3,
        "mm_3d", "mm_main_3d", mm3,
        "configs/GFM/bubble_collapse/gorsse_tc9_water_air_bubble_3d.txt",
        {{240, 200, 200}}
    });
    cases.push_back({
        "gorsse_tc9_water_air_bubble_3d", "tc9", "gorsse_tc9_water_air_bubble_3d", "DIM", 3,
        "mm_3d", "mm_main_3d", mm3,
        "configs/DIM/bubble_collapse/gorsse_tc9_water_air_bubble_3d.txt",
        {{240, 200, 200}}
    });
    cases.push_back({
        "shock_bubble_2d", "test6", "helium_bubble_2d", "DIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/DIM/MM_2D_validation/test6.txt",
        {{1300, 178}}
    });
    cases.push_back({
        "shock_bubble_2d_zero_velocity", "test6_zero_velocity", "helium_bubble_2d_zero_velocity", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/GFM/bubble_collapse/comparison/helium_bubble_2d_zero_velocity_tvd.txt",
        {{325, 45}, {650, 89}, {1300, 178}}
    });
    cases.push_back({
        "shock_bubble_2d_zero_velocity_physical_flow", "test6_zero_velocity_physical_flow", "helium_bubble_2d_zero_velocity_physical_flow", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/GFM/bubble_collapse/comparison/helium_bubble_2d_zero_velocity_physical_flow_tvd.txt",
        {{325, 45}, {650, 89}, {1300, 178}}
    });
    cases.push_back({
        "shock_bubble_2d_zero_velocity_input_mean_star", "test6_zero_velocity_input_mean_star", "helium_bubble_2d_zero_velocity_input_mean_star", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/GFM/bubble_collapse/comparison/helium_bubble_2d_zero_velocity_input_mean_star_tvd.txt",
        {{325, 45}, {650, 89}, {1300, 178}}
    });
    cases.push_back({
        "shock_bubble_2d_zero_velocity_zero_star", "test6_zero_velocity_zero_star", "helium_bubble_2d_zero_velocity_zero_star", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/GFM/bubble_collapse/comparison/helium_bubble_2d_zero_velocity_zero_star_tvd.txt",
        {{325, 45}, {650, 89}, {1300, 178}}
    });
    cases.push_back({
        "shock_bubble_2d_static_equilibrium", "static_normal_speed", "helium_bubble_2d_static_normal_speed", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/GFM/bubble_collapse/comparison/helium_bubble_2d_static_normal_speed_tvd.txt",
        {{325, 45}, {650, 89}, {1300, 178}}
    });
    cases.push_back({
        "shock_bubble_2d_static_equilibrium", "static_flow", "helium_bubble_2d_static_flow", "SIM", 2,
        "mm_2d", "mm_main_2d", mm2,
        "configs/GFM/bubble_collapse/comparison/helium_bubble_2d_static_flow_tvd.txt",
        {{325, 45}, {650, 89}, {1300, 178}}
    });
    cases.push_back({
        "shock_bubble_3d", "test6", "helium_bubble_3d", "SIM", 3,
        "mm_3d", "mm_main_3d", mm3,
        "configs/GFM/MM_3D_validation/test6.txt",
        {{325, 45, 45}}
    });
    cases.push_back({
        "shock_bubble_3d", "test6", "helium_bubble_3d", "DIM", 3,
        "mm_3d", "mm_main_3d", mm3,
        "configs/DIM/MM_3D_validation/test6.txt",
        {{325, 45, 45}}
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
        else if (alias == "fedkiw_d2" || alias == "fedkiwd2") groups.insert("fedkiw_1d");
        else if (alias == "planar" || alias == "fedkiw_2d") groups.insert("fedkiw_2d_planar");
        else if (alias == "oblique" || alias == "fedkiw_2d_oblique" || alias == "oblique45" || alias == "oblique_mm") groups.insert("fedkiw_2d_oblique");
        else if (alias == "shock_bubble" || alias == "bubble") groups.insert("shock_bubble_2d");
        else if (alias == "bubble_reinit" || alias == "shock_bubble_reinit" || alias == "rgfm_bubble_reinit") groups.insert("shock_bubble_2d_reinit");
        else if (alias == "water_air_bubble" || alias == "water_air_bubble_2d" || alias == "practical_case2") groups.insert("water_air_bubble_2d");
        else if (alias == "gorsse_tc9" || alias == "tc9_water_air_bubble" || alias == "water_air_bubble_gorsse") groups.insert("gorsse_tc9_water_air_bubble_2d");
        else if (alias == "he2023_three_material" || alias == "hu2023_three_material" || alias == "three_material" || alias == "three_material_he2023") {
            groups.insert("he2023_three_material_1d");
            groups.insert("he2023_three_material_triple_point_2d");
        }
        else if (alias == "he2023_three_material_1d" || alias == "hu2023_three_material_1d" || alias == "three_material_1d") groups.insert("he2023_three_material_1d");
        else if (alias == "he2023_triple_point" || alias == "hu2023_triple_point" || alias == "he2023_three_material_2d" || alias == "three_material_2d") groups.insert("he2023_three_material_triple_point_2d");
        else if (alias == "applsci_three_material" || alias == "three_material_shock_bubble") groups.insert("he2023_three_material_triple_point_2d");
        else if (alias == "applsci_translation" || alias == "three_material_translation" || alias == "coated_translation") groups.insert("applsci_three_material_translation_2d");
        else if (alias == "gorsse_tc9_3d" || alias == "tc9_water_air_bubble_3d" || alias == "water_air_bubble_gorsse_3d") groups.insert("gorsse_tc9_water_air_bubble_3d");
        else if (alias == "bubble_zero_velocity" || alias == "shock_bubble_zero_velocity" || alias == "rgfm_bubble_zero_velocity") groups.insert("shock_bubble_2d_zero_velocity");
        else if (alias == "bubble_zero_velocity_physical_flow" || alias == "rgfm_bubble_zero_velocity_physical_flow") groups.insert("shock_bubble_2d_zero_velocity_physical_flow");
        else if (alias == "bubble_zero_velocity_input_mean_star" || alias == "rgfm_bubble_zero_velocity_input_mean_star") groups.insert("shock_bubble_2d_zero_velocity_input_mean_star");
        else if (alias == "bubble_zero_velocity_zero_star" || alias == "rgfm_bubble_zero_velocity_zero_star") groups.insert("shock_bubble_2d_zero_velocity_zero_star");
        else if (alias == "bubble_static" || alias == "bubble_static_equilibrium" || alias == "rgfm_bubble_static") groups.insert("shock_bubble_2d_static_equilibrium");
        else if (alias == "shock_bubble3d" || alias == "shock_bubble_3d" || alias == "bubble3d") groups.insert("shock_bubble_3d");
    }
    return groups;
}

bool is_specific_case_filter(const std::string& value)
{
    const std::string alias = lower(value);
    static const std::set<std::string> broad = {
        "toro", "toro_no3d", "toro_1d", "explosion", "explosion2d", "explosion3d",
        "toro_3d", "fedkiw", "fedkiw_1d", "planar", "fedkiw_2d", "oblique",
        "fedkiw_2d_oblique", "oblique45", "oblique_mm", "shock_bubble", "bubble",
        "bubble_reinit", "shock_bubble_reinit", "rgfm_bubble_reinit",
        "water_air_bubble", "water_air_bubble_2d", "practical_case2",
        "gorsse_tc9", "tc9_water_air_bubble", "water_air_bubble_gorsse",
        "he2023_three_material", "hu2023_three_material", "three_material",
        "three_material_he2023", "he2023_three_material_1d", "hu2023_three_material_1d",
        "three_material_1d", "he2023_triple_point", "hu2023_triple_point",
        "he2023_three_material_2d", "three_material_2d",
        "applsci_three_material", "three_material_shock_bubble",
        "applsci_translation", "three_material_translation", "coated_translation",
        "gorsse_tc9_3d", "tc9_water_air_bubble_3d", "water_air_bubble_gorsse_3d",
        "bubble_zero_velocity", "shock_bubble_zero_velocity", "rgfm_bubble_zero_velocity",
        "bubble_zero_velocity_physical_flow", "rgfm_bubble_zero_velocity_physical_flow",
        "bubble_zero_velocity_input_mean_star", "rgfm_bubble_zero_velocity_input_mean_star",
        "bubble_zero_velocity_zero_star", "rgfm_bubble_zero_velocity_zero_star",
        "bubble_static", "bubble_static_equilibrium", "rgfm_bubble_static",
        "shock_bubble3d", "shock_bubble_3d", "bubble3d"
    };
    return !broad.count(alias);
}

std::string case_filter_key(std::string value)
{
    value = lower(value);
    std::string key;
    for (char c : value) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            key.push_back(c);
        }
    }
    return key;
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
        const std::string value = case_filter_key(item);
        const std::string label = case_filter_key(def.label);
        const std::string name = case_filter_key(def.name);
        if (name == value || label == value || label == value + "45") {
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
        if (def.group == "shock_bubble_2d" || def.group == "shock_bubble_3d") {
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
            if (def.group == "he2023_three_material_1d" &&
                def.method == "DIM" &&
                !resolution.empty()) {
                std::ostringstream width;
                width << (1.0 / static_cast<double>(resolution.front()));
                run.overrides["interface_thickness"] = width.str();
            }
            run.run_id = make_run_id(def, resolution, args.omp_threads);
            runs.push_back(run);
        }
    }
    return runs;
}

std::vector<std::vector<int>> core_resolutions_for(const Args& args, const CaseDef& def)
{
    if (!args.resolutions.empty()) {
        return preset_resolutions(args, def);
    }
    if (def.group == "shock_bubble_2d") {
        return {{325, 45}, {650, 89}, {1300, 178}};
    }
    if (def.dimension == 2) {
        return {{100, 100}, {200, 200}, {400, 400}};
    }
    return {{100}, {200}, {400}};
}

bool is_core_case(const CaseDef& def)
{
    if (def.group == "toro_1d") {
        return def.name == "test1" && def.method == "common";
    }
    if (def.group == "fedkiw_1d") {
        return def.name == "test5";
    }
    if (def.group == "fedkiw_2d_oblique") {
        return def.name == "test5";
    }
    if (def.group == "shock_bubble_2d") {
        return true;
    }
    return false;
}

std::vector<RunSpec> build_core_runs(const Args& args)
{
    std::vector<RunSpec> runs;
    for (const auto& def : case_registry()) {
        if (!is_core_case(def) || !contains_method(args.methods, def.method)) {
            continue;
        }
        for (const auto& resolution : core_resolutions_for(args, def)) {
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
        const auto add_dim_epsilon_run = [&](const CaseDef& def,
                                             const std::vector<int>& resolution,
                                             int dx_units,
                                             double physical_width) {
            RunSpec run;
            run.case_def = def;
            run.resolution = resolution;
            run.omp_threads = args.omp_threads;
            run.sensitivity = args.sensitivity;
            run.parameter_name = "epsilon_alpha_dx";
            run.parameter_value = std::to_string(dx_units) + "dx";
            std::ostringstream width;
            width << physical_width;
            run.overrides["interface_thickness"] = width.str();
            run.output_prefix =
                "quant_dim_" + def.label + "_epsilon_alpha_" + std::to_string(dx_units) + "dx";
            run.run_id = make_run_id(def, run.resolution, args.omp_threads, run.parameter_name, run.parameter_value);
            runs.push_back(run);
        };

        const auto fedkiw = find_case("fedkiw_1d", "test5", "DIM").value();
        for (int dx_units : {1, 2, 3, 4, 6}) {
            add_dim_epsilon_run(fedkiw, {400}, dx_units, static_cast<double>(dx_units) / 400.0);
        }

        const auto bubble = find_case("shock_bubble_2d", "test6", "DIM").value();
        for (int dx_units : {1, 2, 3, 4, 6}) {
            add_dim_epsilon_run(bubble, {1300, 178}, dx_units, 0.25 * static_cast<double>(dx_units));
        }
    }
    else if (args.sensitivity == "dim_alpha") {
        const auto add_dim_alpha_run = [&](const CaseDef& def,
                                           const std::vector<int>& resolution,
                                           const std::string& alpha_label,
                                           double alpha) {
            RunSpec run;
            run.case_def = def;
            run.resolution = resolution;
            run.omp_threads = args.omp_threads;
            run.sensitivity = args.sensitivity;
            run.parameter_name = "tanh_alpha";
            run.parameter_value = alpha_label;
            std::ostringstream value;
            value << alpha;
            run.overrides["interface_sharpness_alpha"] = value.str();
            run.output_prefix =
                "quant_dim_" + def.label + "_tanh_alpha_" + alpha_label;
            run.run_id = make_run_id(def, run.resolution, args.omp_threads, run.parameter_name, run.parameter_value);
            runs.push_back(run);
        };

        const auto fedkiw = find_case("fedkiw_1d", "test5", "DIM").value();
        const auto bubble = find_case("shock_bubble_2d", "test6", "DIM").value();

        for (const auto& item : std::vector<std::pair<std::string, double>>{
                 {"0p5", 0.5},
                 {"1", 1.0},
                 {"2", 2.0},
                 {"4", 4.0},
                 {"8", 8.0}}) {
            add_dim_alpha_run(fedkiw, {400}, item.first, item.second);
            add_dim_alpha_run(bubble, {1300, 178}, item.first, item.second);
        }
    }
    else if (args.sensitivity == "sim_reinit") {
        const auto add_sim_reinit_run = [&](const CaseDef& def,
                                            const std::vector<int>& resolution,
                                            const std::string& interval_label,
                                            int interval) {
            RunSpec run;
            run.case_def = def;
            run.resolution = resolution;
            run.omp_threads = args.omp_threads;
            run.sensitivity = args.sensitivity;
            run.parameter_name = "reinit_interval";
            run.parameter_value = interval_label;
            run.overrides["reinit_interval"] = std::to_string(interval);
            run.output_prefix =
                "quant_gfm_" + def.label + "_reinit_" + sanitize_value(interval_label);
            run.run_id = make_run_id(def, run.resolution, args.omp_threads, run.parameter_name, run.parameter_value);
            runs.push_back(run);
        };

        const auto oblique = find_case("fedkiw_2d_oblique", "test5", "SIM").value();
        const auto bubble = find_case("shock_bubble_2d", "test6", "SIM").value();
        for (const auto& item : std::vector<std::pair<std::string, int>>{
                 {"1", 1}, {"2", 2}, {"5", 5}, {"10", 10}, {"20", 20}, {"never", 0}
             }) {
            add_sim_reinit_run(oblique, {200, 200}, item.first, item.second);
            add_sim_reinit_run(bubble, {1300, 178}, item.first, item.second);
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
        {"shock_bubble_2d", "test6", "DIM", {1300, 178}},
        {"shock_bubble_2d", "test6", "SIM", {1300, 178}},
    };
    for (const auto& target : targets) {
        const auto def = find_case(std::get<0>(target), std::get<1>(target), std::get<2>(target)).value();
        for (int threads : {1, 2, 4, 8, 16}) {
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
    else if (args.all_core) {
        runs = build_core_runs(args);
    }
    else {
        runs = build_normal_runs(args);
    }
    return apply_benchmark_repeats(runs, args);
}

} // namespace quant
