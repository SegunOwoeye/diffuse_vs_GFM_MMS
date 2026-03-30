// Storing quantities derived from conserved variables

#pragma once
#include <array>

template<int DIM>
struct ThermoState {
    double rho;
    std::array<double, DIM> vel;
    double p;
    double c;
};