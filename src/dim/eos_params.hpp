#pragma once

#include <array>


// [0] Per-material EOS parameters
template<int NMAT>
struct EOSParams {

    std::array<double, NMAT> gamma{};

    // stiffened gas support 
    std::array<double, NMAT> p_inf{};

};