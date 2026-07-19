#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "tools/quant/cli.hpp"

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_sim_only_scaling()
{
    quant::Args args;
    args.scaling = "openmp_threads";
    args.methods = {"SIM"};
    args.resolutions = {{12, 4}};
    args.scaling_threads = {1, 3};

    const auto runs = quant::build_runs(args);
    require(runs.size() == 2, "SIM-only scaling should create one run per requested thread count");
    require(runs[0].case_def.method == "SIM", "first SIM-only scaling run must use SIM");
    require(runs[1].case_def.method == "SIM", "second SIM-only scaling run must use SIM");
    require(runs[0].omp_threads == 1 && runs[1].omp_threads == 3,
            "scaling must preserve the requested thread list");
    require(runs[0].timing_only && runs[1].timing_only,
            "scaling runs must be marked timing-only");
}

void test_one_dimensional_reference_ladders()
{
    const std::vector<std::vector<int>> exact_ladder = {
        {100}, {200}, {400}, {800}
    };
    for (const std::string& method : {"SIM", "DIM"}) {
        const auto fedkiw = quant::find_case("fedkiw_1d", "test5", method);
        require(fedkiw.has_value(),
                "Fedkiw D2 must exist for both multimaterial methods");
        require(fedkiw->default_resolutions == exact_ladder,
                "Fedkiw D2 must use the 100, 200, 400, 800 ladder");

        quant::RunSpec run;
        run.case_def = fedkiw.value();
        const auto reference = quant::exact_reference_path(run, "tmp/not_used");
        require(reference.has_value(),
                "Fedkiw D2 must resolve a generated exact reference");
        require(reference->filename() == "test5_exact_raw.csv",
                "Fedkiw D2 must use the raw two-material Riemann solution");
        const auto fields = quant::exact_reference_fields(reference.value());
        for (const std::string& field : {"rho", "u0", "p", "e"}) {
            require(fields.count(field) == 1,
                    "Fedkiw exact reference must contain every reported field");
            require(fields.at(field).first.size() == 2000,
                    "Fedkiw exact reference must contain 2000 samples");
        }
    }

    const std::vector<std::vector<int>> numerical_reference_ladder = {
        {100}, {200}, {400}, {800}, {2000}
    };
    for (const std::string& method : {"SIM", "DIM"}) {
        const auto he = quant::find_case(
            "he2023_three_material_1d",
            "three_material_1d",
            method
        );
        require(he.has_value(),
                "He three-material 1D must exist for both methods");
        require(he->default_resolutions == numerical_reference_ladder,
                "He three-material 1D must use N=2000 as its finest reference");
    }
}

void test_three_material_self_reference_is_method_specific()
{
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("quant_three_material_reference_" + quant::now_stamp());
    std::vector<quant::RunSpec> runs;
    std::vector<bool> successes;

    for (const std::string& method : {"SIM", "DIM"}) {
        const auto definition = quant::find_case(
            "he2023_three_material_1d",
            "three_material_1d",
            method
        );
        require(definition.has_value(),
                "three-material reference test requires both methods");
        for (const int resolution : {100, 2000}) {
            quant::RunSpec run;
            run.case_def = definition.value();
            run.resolution = {resolution};
            run.output_prefix =
                "reference_test_" + method + "_" + std::to_string(resolution);
            run.run_id =
                "three_material_reference_" + method + "_" +
                std::to_string(resolution);
            const std::filesystem::path case_dir =
                quant::run_case_dir(root, run);
            std::filesystem::create_directories(case_dir);
            const std::filesystem::path solution =
                case_dir /
                (run.output_prefix + quant::resolution_suffix(run.resolution) + ".csv");
            const double offset =
                method == "SIM" ? 0.0 : 0.1;
            quant::write_file(
                solution,
                "x0,rho,u0,p,e\n"
                "0.25," + std::to_string(1.0 + offset) + ",0.0,1.0,2.5\n"
                "0.75," + std::to_string(0.8 + offset) + ",0.1,0.9,2.3\n"
            );
            runs.push_back(run);
            successes.push_back(true);
        }
    }

    const auto rows =
        quant::he2023_self_reference_error_rows(root, runs, successes);
    std::filesystem::remove_all(root);
    require(rows.size() == 24,
            "each method must compare four fields and three norms against N=2000");
    for (const auto& row : rows) {
        const std::string method = row.at("method");
        require(row.at("N") == "100",
                "N=2000 reference runs must not be reported as errors");
        require(
            row.at("reference").find("reference_test_" + method + "_2000") !=
                std::string::npos,
            "each method must use its own N=2000 numerical reference"
        );
    }
}

void test_production_3d_resolution()
{
    const auto sim = quant::find_case("shock_bubble_3d", "test6", "SIM");
    const auto dim = quant::find_case("shock_bubble_3d", "test6", "DIM");
    require(sim.has_value() && dim.has_value(), "3D helium cases must exist for SIM and DIM");
    require(sim->default_resolutions == std::vector<std::vector<int>>{{325, 45, 45}},
            "SIM 3D helium resolution must be 325x45x45");
    require(dim->default_resolutions == std::vector<std::vector<int>>{{325, 45, 45}},
            "DIM 3D helium resolution must be 325x45x45");

    quant::Args args;
    args.cases = {"bubble3d"};
    args.methods = {"SIM"};
    const auto runs = quant::build_runs(args);
    require(runs.size() == 1, "3D helium selection should create one SIM run");
}

void test_production_3d_sparse_output_times()
{
    quant::Args helium_args;
    helium_args.cases = {"bubble3d"};
    helium_args.methods = {"SIM", "DIM"};
    const auto helium_runs = quant::build_runs(helium_args);
    require(helium_runs.size() == 2,
            "3D helium selection should create SIM and DIM runs");
    for (const auto& run : helium_runs) {
        require(run.overrides.count("output_times") == 1,
                "3D helium production runs must override the output schedule");
        require(run.overrides.at("output_times") == "[70.550, 141.1]",
                "3D helium production runs must retain only middle and final snapshots");
        const std::string generated = quant::generated_config_text(run, "tmp/raw");
        require(generated.find("output_times = [70.550, 141.1]") != std::string::npos,
                "generated 3D helium configuration must contain the sparse output schedule");
    }

    quant::Args gorsse_args;
    gorsse_args.cases = {"gorsse_tc9_3d"};
    gorsse_args.methods = {"SIM", "DIM"};
    const auto gorsse_runs = quant::build_runs(gorsse_args);
    require(gorsse_runs.size() == 2,
            "3D Gorsse selection should create SIM and DIM runs");
    for (const auto& run : gorsse_runs) {
        require(run.overrides.count("output_times") == 1,
                "3D Gorsse production runs must override the output schedule");
        require(run.overrides.at("output_times") == "[3.01e-4, 5.0e-4]",
                "3D Gorsse production runs must retain only middle and final snapshots");
        const std::string generated = quant::generated_config_text(run, "tmp/raw");
        require(generated.find("output_times = [3.01e-4, 5.0e-4]") != std::string::npos,
                "generated 3D Gorsse configuration must contain the sparse output schedule");
    }
}

void test_timing_config_disables_snapshots()
{
    quant::Args args;
    args.scaling = "openmp_threads";
    args.methods = {"SIM"};
    args.resolutions = {{12, 4}};
    args.scaling_threads = {1};
    const auto runs = quant::build_runs(args);
    require(runs.size() == 1, "timing configuration test requires one run");

    const std::string generated = quant::generated_config_text(runs.front(), "tmp/raw");
    require(generated.find("output_times = []") != std::string::npos,
            "timing-only generated configuration must disable scheduled snapshots");
}

void test_sim_production_uses_weno2_reinitialisation()
{
    quant::Args args;
    args.cases = {"bubble"};
    args.methods = {"SIM"};
    const auto runs = quant::build_runs(args);
    require(runs.size() == 1,
            "SIM WENO2 production check requires one helium bubble run");

    const auto& run = runs.front();
    require(run.overrides.at("level_set_spatial_derivative") == "weno2",
            "SIM production runs must use WENO2 level-set derivatives");
    require(run.overrides.at("level_set_reinit_method") == "sussman",
            "SIM production runs must use Hamilton-Jacobi reinitialisation");
    require(run.overrides.at("reinit_iterations") == "10",
            "SIM production runs must use ten pseudo-time iterations");

    const std::string generated = quant::generated_config_text(run, "tmp/raw");
    require(generated.find("level_set_spatial_derivative = weno2") != std::string::npos,
            "generated SIM configuration must select WENO2 derivatives");
    require(generated.find("level_set_reinit_method = sussman") != std::string::npos,
            "generated SIM configuration must select PDE reinitialisation");
    require(generated.find("reinit_iterations = 10") != std::string::npos,
            "generated SIM configuration must contain ten pseudo-time iterations");
}

void test_dim_interface_thickness_sensitivity()
{
    quant::Args args;
    args.sensitivity = "dim_interface_thickness_bubble";
    args.methods = {"DIM"};
    const auto runs = quant::build_runs(args);
    require(runs.size() == 5,
            "DIM interface-thickness sensitivity must create five helium bubble runs");

    const std::vector<std::string> expected_parameters = {
        "1dx", "2dx", "3dx", "4dx", "6dx"
    };
    const std::vector<std::string> expected_thicknesses = {
        "0.25", "0.5", "0.75", "1", "1.5"
    };

    for (std::size_t index = 0; index < runs.size(); ++index) {
        const auto& run = runs[index];
        require(run.parameter_name == "interface_thickness_dx",
                "DIM thickness runs must identify interface thickness as the varied parameter");
        require(run.parameter_value == expected_parameters[index],
                "DIM thickness runs must retain the requested cell-width label");
        require(run.overrides.at("interface_thickness") == expected_thicknesses[index],
                "DIM thickness runs must convert cell widths to physical bubble-grid widths");
        require(run.overrides.at("interface_sharpness_alpha") == "2",
                "DIM thickness runs must hold tanh sharpness alpha fixed at two");
        require(run.overrides.at("output_times") == "[141.1]",
                "DIM thickness runs must retain only the final helium snapshot");

        const std::string generated = quant::generated_config_text(run, "tmp/raw");
        require(
            generated.find(
                "interface_thickness = " + expected_thicknesses[index]
            ) != std::string::npos,
            "generated DIM thickness configuration must contain the varied thickness"
        );
        require(generated.find("interface_sharpness_alpha = 2") != std::string::npos,
                "generated DIM thickness configuration must contain fixed tanh alpha");
    }
}

void test_sim_weno2_reinit_interval_sensitivity()
{
    quant::Args args;
    args.sensitivity = "sim_weno2_reinit_interval_bubble";
    args.methods = {"SIM"};
    const auto runs = quant::build_runs(args);
    require(runs.size() == 6,
            "SIM WENO2 reinitialisation sensitivity must create six helium bubble runs");

    const std::vector<std::string> expected_parameters = {
        "5", "10", "20", "50", "100", "never"
    };
    const std::vector<std::string> expected_intervals = {
        "5", "10", "20", "50", "100", "0"
    };

    for (std::size_t index = 0; index < runs.size(); ++index) {
        const auto& run = runs[index];
        require(run.resolution == std::vector<int>({1300, 178}),
                "SIM WENO2 sensitivity must retain the production resolution by default");
        require(run.parameter_name == "weno2_reinit_interval",
                "SIM WENO2 reinitialisation runs must identify the varied interval");
        require(run.parameter_value == expected_parameters[index],
                "SIM WENO2 reinitialisation runs must retain the requested interval label");
        require(run.overrides.at("reinit_interval") == expected_intervals[index],
                "SIM WENO2 reinitialisation runs must set the requested interval");
        require(run.overrides.at("level_set_reinit_method") == "sussman",
                "SIM WENO2 reinitialisation runs must use the Hamilton-Jacobi PDE");
        require(run.overrides.at("level_set_spatial_derivative") == "weno2",
                "SIM WENO2 reinitialisation runs must select WENO2 derivatives");
        require(run.overrides.at("reinit_iterations") == "10",
                "SIM WENO2 reinitialisation runs must use ten SSP-RK2 pseudo-time steps");
        require(run.overrides.at("output_times") == "[141.1]",
                "SIM WENO2 reinitialisation runs must retain only the final snapshot");

        const std::string generated = quant::generated_config_text(run, "tmp/raw");
        require(
            generated.find(
                "reinit_interval = " + expected_intervals[index]
            ) != std::string::npos,
            "generated SIM WENO2 configuration must contain the varied interval"
        );
        require(generated.find("level_set_reinit_method = sussman") != std::string::npos,
                "generated SIM WENO2 configuration must select PDE reinitialisation");
        require(generated.find("level_set_spatial_derivative = weno2") != std::string::npos,
                "generated SIM WENO2 configuration must select WENO2 derivatives");
        require(generated.find("reinit_iterations = 10") != std::string::npos,
                "generated SIM WENO2 configuration must contain ten SSP-RK2 steps");
    }

    quant::Args low_resolution_args;
    low_resolution_args.sensitivity = "sim_weno2_reinit_interval_bubble";
    low_resolution_args.methods = {"SIM"};
    low_resolution_args.resolutions = {{325, 45}};
    const auto low_resolution_runs = quant::build_runs(low_resolution_args);
    require(low_resolution_runs.size() == 6,
            "low-resolution SIM WENO2 sensitivity must retain all six intervals");
    for (const auto& run : low_resolution_runs) {
        require(run.resolution == std::vector<int>({325, 45}),
                "SIM WENO2 sensitivity must honor the requested low resolution");
    }
}

void test_final_time_solution_selection()
{
    quant::Args args;
    args.cases = {"he2023_three_material_1d"};
    args.methods = {"SIM"};
    args.resolutions = {{100}};
    const auto runs = quant::build_runs(args);
    require(runs.size() == 1, "final-time selection test requires one He 1D run");
    const auto& run = runs.front();

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / ("quant_runner_tests_" + quant::now_stamp());
    const std::filesystem::path case_dir = quant::run_case_dir(root, run);
    std::filesystem::create_directories(case_dir);
    const std::string suffix = quant::resolution_suffix(run.resolution) + ".csv";
    const std::filesystem::path early =
        case_dir / (run.output_prefix + "_t5p000000em02" + suffix);
    const std::filesystem::path final =
        case_dir / (run.output_prefix + "_t2p000000em01" + suffix);
    quant::write_file(early, "early\n");
    quant::write_file(final, "final\n");

    const std::filesystem::path selected = quant::run_solution_path(root, run);
    std::filesystem::remove_all(root);
    require(selected.filename() == final.filename(),
            "solution discovery must select the greatest physical output time");
}

} // namespace

int main()
{
    try {
        test_sim_only_scaling();
        test_one_dimensional_reference_ladders();
        test_three_material_self_reference_is_method_specific();
        test_production_3d_resolution();
        test_production_3d_sparse_output_times();
        test_timing_config_disables_snapshots();
        test_sim_production_uses_weno2_reinitialisation();
        test_dim_interface_thickness_sensitivity();
        test_sim_weno2_reinit_interval_sensitivity();
        test_final_time_solution_selection();
        std::cout << "quant runner tests passed\n";
        return 0;
    }
    catch (const std::exception& exc) {
        std::cerr << "quant runner tests failed: " << exc.what() << "\n";
        return 1;
    }
}
