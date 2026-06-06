#pragma once

// Deterministic Johnson-Cook damage checks for the Barton tensor solid path.

#include "src/solid/elastoplastic/barton/plasticity.hpp"

namespace solid::barton {

inline TensorPrim3D make_damage_validation_primitive(
    double equivalent_stress,
    double triaxiality,
    double temperature,
    double rho)
{
    TensorPrim3D P{};
    P.rho = rho;
    P.T = temperature;
    const double mean = triaxiality * equivalent_stress;
    P.sigma = {
        mean + 2.0 * equivalent_stress / 3.0,
        0.0,
        0.0,
        0.0,
        mean - equivalent_stress / 3.0,
        0.0,
        0.0,
        0.0,
        mean - equivalent_stress / 3.0
    };
    return P;
}

inline double johnson_cook_expected_failure_strain(
    const TensorMaterial& mat,
    double triaxiality,
    double plastic_strain_rate,
    double temperature)
{
    const double rate_ratio = std::max(
        plastic_strain_rate / std::max(mat.reference_plastic_strain_rate, 1.0e-30),
        1.0e-12);
    const double homologous_temperature = std::clamp(
        (temperature - mat.T0) / std::max(mat.melt_temperature - mat.T0, 1.0),
        0.0,
        1.0);
    return std::max(
        (mat.jc_D1 + mat.jc_D2 * std::exp(mat.jc_D3 * triaxiality)) *
            (1.0 + mat.jc_D4 * std::log(rate_ratio)) *
            (1.0 + mat.jc_D5 * homologous_temperature),
        1.0e-8);
}

inline bool close_relative(double actual, double expected, double tolerance)
{
    const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
    return std::abs(actual - expected) <= tolerance * scale;
}

inline int validate_johnson_cook_damage()
{
    TensorMaterial mat{};
    mat.damage_enabled = true;
    mat.rho0 = 8900.0;
    mat.T0 = 300.0;
    mat.melt_temperature = 1356.0;
    mat.jc_D1 = 0.54;
    mat.jc_D2 = 4.89;
    mat.jc_D3 = -3.03;
    mat.jc_D4 = 0.014;
    mat.jc_D5 = 1.12;
    mat.reference_plastic_strain_rate = 1.0;
    mat.failed_damage = 1.0;

    struct Case {
        double triaxiality;
        double rate;
        double temperature;
        double eqps_increment;
        double dt;
    };

    const std::array<Case, 3> cases{{
        {0.0, 1.0, 300.0, 0.543, 0.543},
        {1.0 / 3.0, 10.0, 828.0, 0.010, 0.001},
        {-0.5, 0.1, 1356.0, 0.050, 0.500}
    }};

    bool ok = true;
    std::cout << "[SOLID][JC] Johnson-Cook damage validation\n";
    for (std::size_t i = 0; i < cases.size(); ++i) {
        const auto& c = cases[i];
        const TensorPrim3D P = make_damage_validation_primitive(
            1.0e9,
            c.triaxiality,
            c.temperature,
            mat.rho0);
        const double expected_eps_f = johnson_cook_expected_failure_strain(
            mat,
            c.triaxiality,
            c.rate,
            c.temperature);
        const double actual_eps_f = johnson_cook_failure_strain(P, mat, c.rate);
        const double expected_damage_increment = c.eqps_increment / expected_eps_f;

        TensorState3D U{};
        U.rho = mat.rho0;
        accumulate_johnson_cook_damage(U, P, mat, c.dt, c.eqps_increment);
        const double actual_damage = U.rhoDamage / U.rho;
        const double actual_eqps = U.rhoEqps / U.rho;
        const bool case_ok =
            close_relative(actual_eps_f, expected_eps_f, 1.0e-12) &&
            close_relative(actual_damage, expected_damage_increment, 1.0e-12) &&
            close_relative(actual_eqps, c.eqps_increment, 1.0e-12);
        ok = ok && case_ok;
        std::cout << "  case " << (i + 1)
                  << ": eps_f=" << actual_eps_f
                  << ", damage=" << actual_damage
                  << ", eqps=" << actual_eqps
                  << (case_ok ? " [ok]\n" : " [FAIL]\n");
    }

    const TensorPrim3D Pcap = make_damage_validation_primitive(1.0e9, 1.0, 1200.0, mat.rho0);
    TensorState3D Ucap{};
    Ucap.rho = mat.rho0;
    for (int n = 0; n < 1000; ++n) {
        accumulate_johnson_cook_damage(Ucap, Pcap, mat, 1.0e-6, 0.05);
    }
    const double capped_damage = Ucap.rhoDamage / Ucap.rho;
    const bool cap_ok = close_relative(capped_damage, mat.failed_damage, 1.0e-12);
    ok = ok && cap_ok;
    std::cout << "  cap: damage=" << capped_damage
              << (cap_ok ? " [ok]\n" : " [FAIL]\n");

    if (!ok) {
        std::cerr << "[SOLID][JC] Johnson-Cook damage validation failed\n";
        return 1;
    }
    std::cout << "[SOLID][JC] Johnson-Cook damage validation passed\n";
    return 0;
}

} // namespace solid::barton
