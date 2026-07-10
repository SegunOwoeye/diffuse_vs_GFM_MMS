#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "src/dim/eos_params.hpp"
#include "src/euler/eos_params.hpp"
#include "src/io/config.hpp"

inline EOSParams build_one_material_params(
    const MaterialConfig& m
)
{
    EOSParams params{};
    params.kind = eos_kind_from_string(m.type);

    if (!m.params.count("gamma")) {
        throw std::runtime_error("build_material_params: missing gamma in material");
    }

    params.gamma = m.params.at("gamma");

    if (m.type == "stiffened_gas") {
        if (!m.params.count("p_inf")) {
            throw std::runtime_error("build_material_params: missing p_inf in stiffened gas material");
        }
        params.p_inf = m.params.at("p_inf");
    }

    if (params.kind == EOSKind::noble_abel) {
        if (m.params.count("b")) {
            params.covolume = m.params.at("b");
        }
        else if (m.params.count("covolume")) {
            params.covolume = m.params.at("covolume");
        }
    }

    if (params.kind == EOSKind::peng_robinson) {
        if (m.params.count("R")) {
            params.gas_constant = m.params.at("R");
        }
        if (m.params.count("cv")) {
            params.cv = m.params.at("cv");
        }
        if (m.params.count("Tc")) {
            params.critical_temperature = m.params.at("Tc");
        }
        else if (m.params.count("critical_temperature")) {
            params.critical_temperature = m.params.at("critical_temperature");
        }
        if (m.params.count("pc")) {
            params.critical_pressure = m.params.at("pc");
        }
        else if (m.params.count("critical_pressure")) {
            params.critical_pressure = m.params.at("critical_pressure");
        }
        if (m.params.count("omega")) {
            params.acentric_factor = m.params.at("omega");
        }
        else if (m.params.count("acentric_factor")) {
            params.acentric_factor = m.params.at("acentric_factor");
        }
        if (m.params.count("T_ref")) {
            params.reference_temperature = m.params.at("T_ref");
        }
        else if (m.params.count("reference_temperature")) {
            params.reference_temperature = m.params.at("reference_temperature");
        }
        if (m.params.count("a")) {
            params.pr_a = m.params.at("a");
        }
        if (m.params.count("b")) {
            params.pr_b = m.params.at("b");
        }
        if (params.critical_temperature > 0.0 && params.critical_pressure > 0.0) {
            const double R = params.gas_constant;
            const double Tc = params.critical_temperature;
            const double pc = params.critical_pressure;

            if (!m.params.count("a")) {
                params.pr_a = 0.457236 * R * R * Tc * Tc / pc;
            }
            if (!m.params.count("b")) {
                params.pr_b = 0.077796 * R * Tc / pc;
            }
        }
        if (params.pr_a <= 0.0 || params.pr_b <= 0.0 ||
            params.critical_temperature <= 0.0) {
            throw std::runtime_error(
                "build_material_params: peng_robinson requires Tc and either pc or explicit a,b"
            );
        }
    }

    if (params.kind == EOSKind::tait) {
        if (!m.params.count("B") && !m.params.count("tait_B")) {
            throw std::runtime_error("build_material_params: missing B/tait_B in Tait material");
        }
        if (!m.params.count("rho0")) {
            throw std::runtime_error("build_material_params: missing rho0 in Tait material");
        }

        params.tait_B = m.params.count("B") ? m.params.at("B") : m.params.at("tait_B");
        params.rho0 = m.params.at("rho0");
        if (m.params.count("p0")) {
            params.p0 = m.params.at("p0");
        }
    }

    if (params.kind == EOSKind::mie_gruneisen) {
        if (!m.params.count("rho_ref")) {
            throw std::runtime_error("build_material_params: missing rho_ref in Mie-Gruneisen material");
        }
        if (!m.params.count("A1") || !m.params.count("A2") ||
            !m.params.count("E1") || !m.params.count("E2")) {
            throw std::runtime_error("build_material_params: missing A1/A2/E1/E2 in Mie-Gruneisen material");
        }

        params.rho_ref = m.params.at("rho_ref");
        params.mie_A1 = m.params.at("A1");
        params.mie_A2 = m.params.at("A2");
        params.mie_E1 = m.params.at("E1");
        params.mie_E2 = m.params.at("E2");
    }

    return params;
}

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

        params[m.id] = build_one_material_params(m);
    }

    return params;
}

// [2] Build DIM EOS parameter list from material config
inline dim::EOSParams build_dim_material_params(
    const std::vector<MaterialConfig>& materials
)
{
    dim::EOSParams params{};
    params.material.assign(materials.size(), EOSParams{});

    for (const auto& m : materials) {
        if (m.id < 0 || m.id >= static_cast<int>(materials.size())) {
            throw std::runtime_error("build_dim_material_params: invalid material id");
        }

        params.material[m.id] = build_one_material_params(m);
    }

    params.validate();
    return params;
}
