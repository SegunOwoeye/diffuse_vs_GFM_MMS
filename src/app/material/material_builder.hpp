#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "src/io/config.hpp"
#include "src/euler/eos_params.hpp"


// [0] Build EOS parameter list from material config
inline std::vector<EOSParams> build_material_params(
    const std::vector<MaterialConfig>& materials
)
{
    std::vector<EOSParams> params(materials.size());

    for (const auto& m : materials) {
        if (m.id < 0 || m.id >= static_cast<int>(materials.size())) {
            throw std::runtime_error("build_material_params: invalid material id");
        }

        if (m.type == "ideal_gas") {
            if (!m.params.count("gamma")) {
                throw std::runtime_error("build_material_params: missing gamma in material");
            }

            params[m.id].gamma = m.params.at("gamma");
        }
        else {
            throw std::runtime_error("build_material_params: unsupported EOS type: " + m.type);
        }
    }

    return params;
}

