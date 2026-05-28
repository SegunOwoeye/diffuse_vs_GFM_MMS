#pragma once

#include <stdexcept>
#include <string>


// [0] Equation of State Type
enum class EOSKind {
    ideal_gas,
    stiffened_gas,
    noble_abel,
    peng_robinson,
    tait
};


inline EOSKind eos_kind_from_string(const std::string& type)
{
    if (type == "ideal_gas") {
        return EOSKind::ideal_gas;
    }
    if (type == "stiffened_gas") {
        return EOSKind::stiffened_gas;
    }
    if (type == "real_gas" || type == "noble_abel" || type == "noble_abel_gas") {
        return EOSKind::noble_abel;
    }
    if (type == "peng_robinson" || type == "peng-robinson") {
        return EOSKind::peng_robinson;
    }
    if (type == "tait") {
        return EOSKind::tait;
    }

    throw std::runtime_error("Unsupported EOS type: " + type);
}


// [0] Equation of State Parameters
struct EOSParams {

    EOSKind kind = EOSKind::ideal_gas;

    double gamma = 1.4;

    // Used only for stiffened gas
    double p_inf = 0.0;

    // Noble-Abel / real gas co-volume. Units are volume per mass.
    double covolume = 0.0;

    // Peng-Robinson pure-fluid parameters. R, cv, a, and b are mass-specific.
    double gas_constant = 287.0;
    double cv = 718.0;
    double critical_temperature = 0.0;
    double critical_pressure = 0.0;
    double acentric_factor = 0.0;
    double reference_temperature = 0.0;
    double pr_a = 0.0;
    double pr_b = 0.0;

    // Tait liquid parameters: p = p0 + B*((rho/rho0)^gamma - 1).
    double tait_B = 0.0;
    double rho0 = 1.0;
    double p0 = 0.0;

};
