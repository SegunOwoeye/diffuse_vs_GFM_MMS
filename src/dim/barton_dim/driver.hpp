#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "src/dim/barton_dim/initial_conditions.hpp"
#include "src/dim/barton_dim/output.hpp"
#include "src/dim/barton_dim/solver.hpp"

namespace dim::barton_dim {

inline bool is_barton_dim_config(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open configuration file: " + filename);
    std::string line;
    while (std::getline(file, line)) {
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        line = solid::text::trim(line);
        if (line.empty()) continue;
        const auto [key, value] = solid::text::split_key_value(line);
        if (key == "model") return value == "barton_dim";
    }
    return false;
}

inline std::string time_suffix(double time)
{
    std::ostringstream stream;
    stream << "_t" << std::scientific << std::setprecision(6) << time;
    std::string suffix = stream.str();
    for (char& character : suffix) {
        if (character == '+') character = 'p';
        else if (character == '-') character = 'm';
    }
    return suffix;
}

template<int DIM>
inline int run_case(const std::string& config_file)
{
    const Config<DIM> config = load_config<DIM>(config_file);
    std::vector<State<DIM>> states = initialise(config);
    std::string base = config.output_prefix;
    if (!config.output_dir.empty()) {
        base = config.output_dir + "/" + config.output_prefix;
        if (config.output_layout == "rgfm") base += "/" + config.output_prefix;
    }
    std::size_t next_output = 0;
    double time = 0.0;
    int steps = 0;
    while (time < config.final_time - 1.0e-14) {
        while (next_output < config.output_times.size() && config.output_times[next_output] < time - 1.0e-14) {
            ++next_output;
        }
        const double output_time = next_output < config.output_times.size()
            ? config.output_times[next_output]
            : config.final_time;
        const double maximum_dt = std::min(config.final_time - time, output_time - time);
        if (maximum_dt <= 1.0e-14) {
            write_csv(base + time_suffix(output_time) + ".csv", states, config, time);
            ++next_output;
            continue;
        }
        const double dt = advance_one_timestep(states, config, maximum_dt);
        if (!std::isfinite(dt) || dt <= 0.0) throw std::runtime_error("Barton-DIM timestep collapsed");
        time += dt;
        ++steps;
        if (config.progress_interval > 0 && steps % config.progress_interval == 0) {
            std::cout << "Barton-DIM progress: steps=" << steps << ", time=" << time
                      << ", dt=" << dt << std::endl;
        }
        if (next_output < config.output_times.size() && time >= config.output_times[next_output] - 1.0e-13) {
            write_csv(base + time_suffix(config.output_times[next_output]) + ".csv", states, config, time);
            ++next_output;
        }
        if (steps > 1000000) throw std::runtime_error("Barton-DIM exceeded one million timesteps");
    }
    if (next_output == 0 || config.output_times.empty() ||
        std::abs(config.output_times.back() - config.final_time) > 1.0e-13) {
        write_csv(base + time_suffix(config.final_time) + ".csv", states, config, time);
    }
    const std::filesystem::path runtime_path(base + "_runtime.txt");
    if (!runtime_path.parent_path().empty()) std::filesystem::create_directories(runtime_path.parent_path());
    std::ofstream runtime(runtime_path);
    if (!runtime) throw std::runtime_error("Cannot write Barton-DIM runtime report");
    runtime << "model = barton_dim\n";
    runtime << "dimension = " << DIM << "\n";
    runtime << "steps = " << steps << "\n";
    runtime << "final_time = " << time << "\n";
    runtime << "cells = " << states.size() << "\n";
    runtime << "output_format = " << config.output_format << "\n";
    runtime << "output_layout = " << config.output_layout << "\n";
    std::cout << "Barton-DIM finished: dimension=" << DIM << ", steps=" << steps
              << ", final_time=" << time << "\n";
    return 0;
}

} // namespace dim::barton_dim
