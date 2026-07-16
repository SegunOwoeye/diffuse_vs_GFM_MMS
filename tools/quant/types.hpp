#pragma once

// Header-only implementation units for the quantitative validation runner.
// Shared data structures for cases, run instances, and command-line options.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

namespace quant {

struct CaseDef {
    std::string group;
    std::string name;
    std::string label;
    std::string method;
    int dimension = 1;
    std::string executable_key;
    std::string executable_path;
    std::vector<std::string> compile_command;
    fs::path config_path;
    std::vector<std::vector<int>> default_resolutions;
};

struct RunSpec {
    std::string run_id;
    CaseDef case_def;
    std::vector<int> resolution;
    std::string output_prefix;
    int omp_threads = 6;
    int repeat_id = 0;
    bool warmup = false;
    bool timing_only = false;
    std::string benchmark_mode = "standard";
    std::string sensitivity;
    std::string scaling;
    std::string parameter_name;
    std::string parameter_value;
    std::map<std::string, std::string> overrides;
};

struct Args {
    std::string preset = "quick";
    bool dry_run = false;
    bool overwrite = false;
    bool resume = false;
    bool collect_only = false;
    bool all_core = false;
    fs::path result_root;
    std::vector<std::string> cases;
    std::vector<std::string> methods;
    std::vector<std::vector<int>> resolutions;
    std::string sensitivity;
    std::string scaling;
    int omp_threads = 6;
    std::vector<int> scaling_threads = {1, 2, 4, 8, 16, 32};
    int timeout_seconds = 0;
    int conservation_interval = 25;
    int benchmark_repeats = 1;
    int benchmark_warmups = 0;
    std::string benchmark_mode = "standard";
};

} // namespace quant
