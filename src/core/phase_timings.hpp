#pragma once

// [0] Lightweight solver phase timings
struct SolverPhaseTimings {
    double cfl_seconds = 0.0;
    double sweep_seconds = 0.0;
    double boundary_seconds = 0.0;
    double level_set_seconds = 0.0;
    double material_transfer_seconds = 0.0;
    double conservation_seconds = 0.0;
    double output_seconds = 0.0;

    inline void add_step(const SolverPhaseTimings& other)
    {
        cfl_seconds += other.cfl_seconds;
        sweep_seconds += other.sweep_seconds;
        boundary_seconds += other.boundary_seconds;
        level_set_seconds += other.level_set_seconds;
        material_transfer_seconds += other.material_transfer_seconds;
    }
};
