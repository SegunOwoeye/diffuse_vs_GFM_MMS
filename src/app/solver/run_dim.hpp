#pragma once

#include <array>
#include <stdexcept>
#include <vector>

#include "src/euler/eos_params.hpp"
#include "src/io/config.hpp"


// [0] Diffuse-interface driver placeholder
template<int DIM, typename EOS>
inline void run_dim_case(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N,
    const std::vector<EOSParams>& material_params
)
{
    (void)cfg;
    (void)N;
    (void)material_params;

    throw std::runtime_error(
        "DIM mode selected, but diffuse-interface solver path is not implemented in this driver yet"
    );
}

