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
    require(!runs.front().overrides.count("output_times"),
            "production 3D runs must retain the configured output-time series");
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
        test_production_3d_resolution();
        test_timing_config_disables_snapshots();
        test_final_time_solution_selection();
        std::cout << "quant runner tests passed\n";
        return 0;
    }
    catch (const std::exception& exc) {
        std::cerr << "quant runner tests failed: " << exc.what() << "\n";
        return 1;
    }
}
