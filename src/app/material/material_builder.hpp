#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "src/dim/eos_params.hpp"
#include "src/euler/eos_params.hpp"
#include "src/io/config.hpp"

// [1] Build GFM EOS parameter list from material config
inline std::vector<EOSParams> build_material_params(
    const std::vector<MaterialConfig>& materials
)
{
    std::vector<EOSParams> params(materials.size());

    for (const auto& m : materials) {
        if (m.id < 0 || m.id >= static_cast<int>(materials.size())) {
            throw std::runtime_error("build_material_params: invalid material id");
        }

        if (m.type != "ideal_gas" && m.type != "stiffened_gas") {
            throw std::runtime_error("build_material_params: unsupported EOS type: " + m.type);
        }

        if (!m.params.count("gamma")) {
            throw std::runtime_error("build_material_params: missing gamma in material");
        }

        params[m.id].gamma = m.params.at("gamma");
        if (m.type == "stiffened_gas") {
            if (!m.params.count("p_inf")) {
                throw std::runtime_error("build_material_params: missing p_inf in stiffened gas material");
            }
            params[m.id].p_inf = m.params.at("p_inf");
        }
    }

    return params;
}

// [2] Build DIM EOS parameter list from material config
inline dim::EOSParams build_dim_material_params(
    const std::vector<MaterialConfig>& materials
)
{
    dim::EOSParams params{};
    params.gamma.assign(materials.size(), 0.0);

    for (const auto& m : materials) {
        if (m.id < 0 || m.id >= static_cast<int>(materials.size())) {
            throw std::runtime_error("build_dim_material_params: invalid material id");
        }

        if (m.type != "ideal_gas") {
            throw std::runtime_error("build_dim_material_params: unsupported EOS type: " + m.type);
        }

        if (!m.params.count("gamma")) {
            throw std::runtime_error("build_dim_material_params: missing gamma in material");
        }

        params.gamma[m.id] = m.params.at("gamma");
    }

    params.validate();
    return params;
}
