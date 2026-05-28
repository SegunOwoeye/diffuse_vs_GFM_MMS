#pragma once

#include <array>
#include <stdexcept>
#include <vector>

#include "src/euler/eos_params.hpp"

namespace dim {

struct EOSParams {
    std::vector<::EOSParams> material{};

    int nmat() const
    {
        return static_cast<int>(material.size());
    }

    void validate() const
    {
        if (material.empty()) {
            throw std::runtime_error("dim::EOSParams: no materials configured");
        }

        for (const auto& params : material) {
            if (params.gamma <= 1.0) {
                throw std::runtime_error("dim::EOSParams: gamma must be > 1");
            }
        }
    }
};

} // namespace dim
