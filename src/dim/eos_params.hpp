#pragma once

#include <array>
#include <stdexcept>
#include <vector>

namespace dim {

struct EOSParams {
    std::vector<double> gamma{};

    int nmat() const
    {
        return static_cast<int>(gamma.size());
    }

    void validate() const
    {
        if (gamma.empty()) {
            throw std::runtime_error("dim::EOSParams: no materials configured");
        }

        
    }
};

} // namespace dim
