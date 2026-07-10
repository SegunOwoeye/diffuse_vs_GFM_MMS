#pragma once

#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "src/euler/state.hpp"
#include "src/euler/eos_params.hpp"
#include "src/math/numerical_safety.hpp"


template<int DIM>
inline double internal_energy_from_conserved(
    const Conserved<DIM>& U,
    double rho
)
{
    double mom2 = 0.0;
    for (int d = 0; d < DIM; ++d) {
        mom2 += U.mom[d] * U.mom[d];
    }

    const double kineticE = 0.5 * safe_div(mom2, rho);
    return std::max(safe_div(U.E - kineticE, rho), 0.0);
}


template<int DIM>
inline double raw_internal_energy_from_conserved(
    const Conserved<DIM>& U,
    double rho
)
{
    double mom2 = 0.0;
    for (int d = 0; d < DIM; ++d) {
        mom2 += U.mom[d] * U.mom[d];
    }

    const double kineticE = 0.5 * safe_div(mom2, rho);
    return safe_div(U.E - kineticE, rho);
}


inline double tait_pressure_from_density(
    double rho,
    const EOSParams& params
)
{
    rho = clamp_min(rho);
    const double rho0 = clamp_min(params.rho0);
    const double B = clamp_min(params.tait_B);
    return params.p0 + B * (std::pow(rho / rho0, params.gamma) - 1.0);
}


inline double tait_internal_energy_from_density(
    double rho,
    const EOSParams& params
)
{
    rho = clamp_min(rho);
    const double rho0 = clamp_min(params.rho0);
    const double B = clamp_min(params.tait_B);
    const double g = params.gamma;

    const double compression =
        B / (rho0 * (g - 1.0)) *
        (std::pow(rho / rho0, g - 1.0) - 1.0);
    const double reference_shift =
        (params.p0 - B) * (1.0 / rho0 - 1.0 / rho);

    return std::max(compression + reference_shift, 1e-12);
}

inline double mie_gruneisen_power_term(
    double r,
    double exponent,
    double coefficient,
    double rho_ref
)
{
    const double denom = rho_ref * safe_denom(exponent - 1.0, 1e-14);
    return coefficient / denom * (std::pow(clamp_min(r), exponent - 1.0) - 1.0);
}


inline double mie_gruneisen_cold_energy(
    double rho,
    const EOSParams& params
)
{
    rho = clamp_min(rho);
    const double rho_ref = clamp_min(params.rho_ref);
    const double r = rho / rho_ref;
    return mie_gruneisen_power_term(r, params.mie_E1, params.mie_A1, rho_ref)
        - mie_gruneisen_power_term(r, params.mie_E2, params.mie_A2, rho_ref);
}


inline double mie_gruneisen_cold_pressure(
    double rho,
    const EOSParams& params
)
{
    rho = clamp_min(rho);
    const double rho_ref = clamp_min(params.rho_ref);
    const double r = rho / rho_ref;
    return params.mie_A1 * std::pow(r, params.mie_E1)
        - params.mie_A2 * std::pow(r, params.mie_E2);
}


inline double mie_gruneisen_pressure_from_rho_e(
    double rho,
    double e,
    const EOSParams& params
)
{
    rho = clamp_min(rho);
    const double thermal_e = e - mie_gruneisen_cold_energy(rho, params);
    return (params.gamma - 1.0) * rho * thermal_e +
        mie_gruneisen_cold_pressure(rho, params);
}


inline double peng_robinson_kappa(const EOSParams& params)
{
    const double omega = params.acentric_factor;
    return 0.37464 + 1.54226 * omega - 0.26992 * omega * omega;
}


inline double peng_robinson_reduced_sqrt_temperature(
    double T,
    const EOSParams& params
)
{
    return std::sqrt(clamp_min(T) / clamp_min(params.critical_temperature));
}


inline double peng_robinson_alpha(
    double T,
    const EOSParams& params
)
{
    const double kappa = peng_robinson_kappa(params);
    const double sqrt_Tr = peng_robinson_reduced_sqrt_temperature(T, params);
    const double q = 1.0 + kappa * (1.0 - sqrt_Tr);
    return q * q;
}


inline double peng_robinson_a(
    double T,
    const EOSParams& params
)
{
    return clamp_min(params.pr_a) * peng_robinson_alpha(T, params);
}


inline double peng_robinson_da_dT(
    double T,
    const EOSParams& params
)
{
    T = clamp_min(T);
    const double kappa = peng_robinson_kappa(params);
    const double sqrt_Tr = peng_robinson_reduced_sqrt_temperature(T, params);
    const double q = 1.0 + kappa * (1.0 - sqrt_Tr);
    return clamp_min(params.pr_a) * (-kappa * q * sqrt_Tr / T);
}


inline double peng_robinson_d2a_dT2(
    double T,
    const EOSParams& params
)
{
    T = clamp_min(T);
    const double kappa = peng_robinson_kappa(params);
    const double sqrt_Tr = peng_robinson_reduced_sqrt_temperature(T, params);
    const double q = 1.0 + kappa * (1.0 - sqrt_Tr);
    return clamp_min(params.pr_a) *
        (kappa * (kappa * sqrt_Tr * sqrt_Tr + q * sqrt_Tr) / (2.0 * T * T));
}


inline double peng_robinson_K1(
    double rho,
    const EOSParams& params
)
{
    rho = clamp_min(rho);
    const double v = 1.0 / rho;
    const double b = clamp_min(params.pr_b, 1e-14);
    const double sqrt2 = std::sqrt(2.0);
    const double numerator = safe_denom(v + (1.0 - sqrt2) * b, 1e-14);
    const double denominator = safe_denom(v + (1.0 + sqrt2) * b, 1e-14);
    return std::log(numerator / denominator) / (2.0 * sqrt2 * b);
}


inline double peng_robinson_pressure_from_rho_T(
    double rho,
    double T,
    const EOSParams& params
)
{
    rho = clamp_min(rho);
    T = clamp_min(T);

    const double R = clamp_min(params.gas_constant);
    const double v = 1.0 / rho;
    const double b = clamp_min(params.pr_b, 1e-14);
    const double a = peng_robinson_a(T, params);

    const double repulsive = R * T / safe_denom(v - b, 1e-14);
    const double attractive_denom = safe_denom(v * v + 2.0 * b * v - b * b, 1e-14);
    return repulsive - a / attractive_denom;
}


inline double peng_robinson_internal_energy_from_rho_T(
    double rho,
    double T,
    const EOSParams& params
)
{
    rho = clamp_min(rho);
    T = clamp_min(T);

    const double cv = clamp_min(params.cv);
    const double T_ref = std::max(params.reference_temperature, 0.0);
    const double e0 = cv * (T - T_ref);
    const double K1 = peng_robinson_K1(rho, params);
    const double a = peng_robinson_a(T, params);
    const double da_dT = peng_robinson_da_dT(T, params);

    return e0 + K1 * (a - T * da_dT);
}


inline double peng_robinson_de_dT_rho(
    double rho,
    double T,
    const EOSParams& params
)
{
    return clamp_min(params.cv) -
        peng_robinson_K1(rho, params) * T * peng_robinson_d2a_dT2(T, params);
}


inline double peng_robinson_temperature_from_rho_e(
    double rho,
    double e,
    const EOSParams& params
)
{
    rho = clamp_min(rho);
    e = std::max(e, -1e20);

    double lo = 1e-6;
    double hi = std::max(
        2.0 * clamp_min(params.critical_temperature),
        std::abs(e) / clamp_min(params.cv) + std::max(params.reference_temperature, 0.0) + 10.0
    );

    auto residual = [&](double T) {
        return peng_robinson_internal_energy_from_rho_T(rho, T, params) - e;
    };

    double flo = residual(lo);
    double fhi = residual(hi);

    for (int n = 0; n < 100 && fhi < 0.0; ++n) {
        hi *= 2.0;
        fhi = residual(hi);
    }

    if (!(flo <= 0.0 && fhi >= 0.0)) {
        return clamp_min(std::abs(e) / clamp_min(params.cv) + std::max(params.reference_temperature, 0.0));
    }

    double T = 0.5 * (lo + hi);
    for (int n = 0; n < 80; ++n) {
        const double f = residual(T);
        if (std::abs(f) <= 1e-10 * std::max(1.0, std::abs(e))) {
            return clamp_min(T);
        }

        const double dedT = safe_denom(peng_robinson_de_dT_rho(rho, T, params), 1e-14);
        double candidate = T - f / dedT;

        if (!(candidate > lo && candidate < hi) || !std::isfinite(candidate)) {
            candidate = 0.5 * (lo + hi);
        }

        if (f > 0.0) {
            hi = T;
        }
        else {
            lo = T;
        }

        T = candidate;
    }

    return clamp_min(T);
}


inline double peng_robinson_temperature_from_rho_p(
    double rho,
    double p,
    const EOSParams& params
)
{
    rho = clamp_min(rho);
    p = clamp_min(p);

    double lo = 1e-6;
    double hi = std::max(
        2.0 * clamp_min(params.critical_temperature),
        p * std::max(1.0 / rho - params.pr_b, 1e-12) / clamp_min(params.gas_constant) + 10.0
    );

    auto residual = [&](double T) {
        return peng_robinson_pressure_from_rho_T(rho, T, params) - p;
    };

    double flo = residual(lo);
    double fhi = residual(hi);

    for (int n = 0; n < 100 && fhi < 0.0; ++n) {
        hi *= 2.0;
        fhi = residual(hi);
    }

    if (!(flo <= 0.0 && fhi >= 0.0)) {
        return clamp_min(p * std::max(1.0 / rho - params.pr_b, 1e-12) /
            clamp_min(params.gas_constant));
    }

    for (int n = 0; n < 80; ++n) {
        const double mid = 0.5 * (lo + hi);
        const double fmid = residual(mid);

        if (std::abs(fmid) <= 1e-10 * std::max(1.0, p)) {
            return clamp_min(mid);
        }

        if (fmid > 0.0) {
            hi = mid;
        }
        else {
            lo = mid;
        }
    }

    return clamp_min(0.5 * (lo + hi));
}


inline double peng_robinson_pressure_from_rho_e(
    double rho,
    double e,
    const EOSParams& params
)
{
    const double T = peng_robinson_temperature_from_rho_e(rho, e, params);
    return peng_robinson_pressure_from_rho_T(rho, T, params);
}


inline double peng_robinson_internal_energy(
    double rho,
    double p,
    const EOSParams& params
)
{
    const double T = peng_robinson_temperature_from_rho_p(rho, p, params);
    return peng_robinson_internal_energy_from_rho_T(rho, T, params);
}


inline double peng_robinson_sound_speed_from_rho_T(
    double rho,
    double T,
    const EOSParams& params
)
{
    rho = clamp_min(rho);
    T = clamp_min(T);

    const double R = clamp_min(params.gas_constant);
    const double v = 1.0 / rho;
    const double b = clamp_min(params.pr_b, 1e-14);
    const double a = peng_robinson_a(T, params);
    const double da_dT = peng_robinson_da_dT(T, params);
    const double D = safe_denom(v * v + 2.0 * b * v - b * b, 1e-14);
    const double dpdT_v = R / safe_denom(v - b, 1e-14) - da_dT / D;
    const double dpdv_T =
        -R * T / safe_denom((v - b) * (v - b), 1e-14) +
        2.0 * a * (v + b) / safe_denom(D * D, 1e-14);

    const double cv = clamp_min(peng_robinson_de_dT_rho(rho, T, params));
    const double cp = cv - T * dpdT_v * dpdT_v / safe_denom(dpdv_T, 1e-14);
    const double gamma = clamp_min(cp / cv);
    const double c2 = -gamma * v * v * dpdv_T;

    return std::sqrt(clamp_min(c2));
}


inline double numerical_sound_speed_from_rho_e(
    double rho,
    double e,
    const EOSParams& params,
    double (*pressure_rho_e)(double, double, const EOSParams&)
)
{
    rho = clamp_min(rho);
    e = std::max(e, 0.0);

    const double drho = std::max(1e-7 * rho, 1e-8);
    const double de = std::max(1e-7 * std::max(e, 1.0), 1e-8);

    const double p_plus_rho = pressure_rho_e(rho + drho, e, params);
    const double p_minus_rho = pressure_rho_e(std::max(rho - drho, 1e-12), e, params);
    const double dp_drho_e = (p_plus_rho - p_minus_rho) /
        ((rho + drho) - std::max(rho - drho, 1e-12));

    const double p_plus_e = pressure_rho_e(rho, e + de, params);
    const double p_minus_e = pressure_rho_e(rho, std::max(e - de, 0.0), params);
    const double dp_de_rho = (p_plus_e - p_minus_e) /
        ((e + de) - std::max(e - de, 0.0));

    const double p = pressure_rho_e(rho, e, params);
    const double c2 = dp_drho_e + dp_de_rho * safe_div(p, rho * rho);
    return std::sqrt(clamp_min(c2));
}


// [1] Ideal Gas EOS
struct IdealGasEOS {
    // Pressure
    template<int DIM>
    static double pressure(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = clamp_min(U.rho);

        double mom2 = 0.0;
        for (int d = 0; d < DIM; ++d)
            mom2 += U.mom[d] * U.mom[d];

        const double kineticE = 0.5 * safe_div(mom2, rho);

        const double p_raw = (params.gamma - 1.0) * (U.E - kineticE);

        return clamp_min(p_raw);
    }

    // Speed of Sound
    template<int DIM>
    static double sound_speed(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = clamp_min(U.rho);
        const double p = pressure(U, params);

        return std::sqrt(params.gamma * safe_div(p, rho));
    }

    // Internal energy
    static double internal_energy(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        return safe_div(p, ((params.gamma - 1.0) * rho));
    }

    // Combined Pressure and sound speed calculation
    template<int DIM>
    static inline void compute_p_c(
        const Conserved<DIM>& U,
        const EOSParams& params,
        double& p,
        double& c
    )
    {
        const double rho = clamp_min(U.rho);

        double mom2 = 0.0;
        for (int d = 0; d < DIM; ++d)
            mom2 += U.mom[d] * U.mom[d];

        const double kineticE = 0.5 * safe_div(mom2, rho);

        const double p_raw = (params.gamma - 1.0) * (U.E - kineticE);
        p = clamp_min(p_raw);

        c = std::sqrt(params.gamma * safe_div(p, rho));
    }

    /* 
        Recover ghost density from p* using EOS-consistent closure
        (preserves thermodynamic branch instead of fixing density)
    */

    // Entropy invariant (p / rho^gamma)
    static double entropy_invariant(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        rho = clamp_min(rho);
        p = clamp_min(p);

        return safe_div(p, std::pow(rho, params.gamma));
    }


    // Recover density from (p, invariant)
    static double density_from_p_invariant(
        double p,
        double K,
        const EOSParams& params
    )
    {
        p = clamp_min(p);
        K = clamp_min(K);

        return std::pow(safe_div(p, K), safe_div(1.0, params.gamma));
    }
};



// [2] Stiffened Gas EOS
struct StiffenedGasEOS {
    // Pressure
    template<int DIM>
    static double pressure(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = clamp_min(U.rho);

        double mom2 = 0.0;
        for (int d = 0; d < DIM; ++d)
            mom2 += U.mom[d] * U.mom[d];

        const double kineticE = 0.5 * safe_div(mom2,rho);

        const double p_raw = (params.gamma - 1.0) * (U.E - kineticE)
            - params.gamma * params.p_inf;

        return clamp_min(p_raw);
    }

    // Speed of Sound
    template<int DIM>
    static double sound_speed(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = clamp_min(U.rho);
        const double p = pressure(U, params);

        return std::sqrt(params.gamma * safe_div((p + params.p_inf), rho));
    }


    // Internal energy
    static double internal_energy(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        return safe_div((p + params.gamma * params.p_inf),
            ((params.gamma - 1.0) * rho));
    }

    // Combined Pressure and sound speed calculation
    template<int DIM>
    static inline void compute_p_c(
        const Conserved<DIM>& U,
        const EOSParams& params,
        double& p,
        double& c
    )
    {
        const double rho = clamp_min(U.rho);

        double mom2 = 0.0;
        for (int d = 0; d < DIM; ++d)
            mom2 += U.mom[d] * U.mom[d];

        const double kineticE = 0.5 * safe_div(mom2, rho);

        const double p_raw = (params.gamma - 1.0) * (U.E - kineticE)
            - params.gamma * params.p_inf;

        p = clamp_min(p_raw);

        c = std::sqrt(params.gamma * safe_div((p + params.p_inf), rho));
    }


    /* 
        Recover ghost density from p* using EOS-consistent closure
        (preserves thermodynamic branch instead of fixing density)
    */

    // Entropy invariant ((p + p_inf) / rho^gamma)
    static double entropy_invariant(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        rho = clamp_min(rho);
        p = clamp_min(p);

        return safe_div((p + params.p_inf), std::pow(rho, params.gamma));
    }

    // Recover density from (p, invariant)
    static double density_from_p_invariant(
        double p,
        double K,
        const EOSParams& params
    )
    {
        p = clamp_min(p);
        K = clamp_min(K);

        return std::pow(safe_div(p + params.p_inf, K),
            safe_div(1.0, params.gamma)
        );
    }






};


// [3] Noble-Abel / co-volume real gas EOS
struct NobleAbelGasEOS {
    template<int DIM>
    static double pressure(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = clamp_min(U.rho);
        const double e = internal_energy_from_conserved<DIM>(U, rho);
        const double denom = safe_denom(1.0 - params.covolume * rho, 1e-14);
        return clamp_min((params.gamma - 1.0) * rho * e / denom);
    }

    template<int DIM>
    static double sound_speed(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = clamp_min(U.rho);
        const double p = pressure<DIM>(U, params);
        const double denom = safe_denom(rho * (1.0 - params.covolume * rho), 1e-14);
        return std::sqrt(clamp_min(params.gamma * p / denom));
    }

    static double internal_energy(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        rho = clamp_min(rho);
        p = clamp_min(p);
        return p * (1.0 - params.covolume * rho) /
            ((params.gamma - 1.0) * rho);
    }

    template<int DIM>
    static inline void compute_p_c(
        const Conserved<DIM>& U,
        const EOSParams& params,
        double& p,
        double& c
    )
    {
        p = pressure<DIM>(U, params);
        c = sound_speed<DIM>(U, params);
    }

    static double entropy_invariant(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        rho = clamp_min(rho);
        p = clamp_min(p);
        const double specific_volume = safe_denom(1.0 / rho - params.covolume, 1e-14);
        return p * std::pow(specific_volume, params.gamma);
    }

    static double density_from_p_invariant(
        double p,
        double K,
        const EOSParams& params
    )
    {
        p = clamp_min(p);
        K = clamp_min(K);
        const double specific_volume = std::pow(K / p, 1.0 / params.gamma);
        return 1.0 / safe_denom(specific_volume + params.covolume, 1e-14);
    }
};


// [4] Peng-Robinson EOS with calorically-perfect internal-energy closure
struct PengRobinsonEOS {
    template<int DIM>
    static double pressure(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = clamp_min(U.rho);
        const double e = raw_internal_energy_from_conserved<DIM>(U, rho);
        return clamp_min(peng_robinson_pressure_from_rho_e(rho, e, params));
    }

    template<int DIM>
    static double sound_speed(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = clamp_min(U.rho);
        const double e = raw_internal_energy_from_conserved<DIM>(U, rho);
        const double T = peng_robinson_temperature_from_rho_e(rho, e, params);
        return peng_robinson_sound_speed_from_rho_T(rho, T, params);
    }

    static double internal_energy(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        return peng_robinson_internal_energy(rho, p, params);
    }

    template<int DIM>
    static inline void compute_p_c(
        const Conserved<DIM>& U,
        const EOSParams& params,
        double& p,
        double& c
    )
    {
        p = pressure<DIM>(U, params);
        c = sound_speed<DIM>(U, params);
    }

    static double entropy_invariant(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        rho = clamp_min(rho);
        p = clamp_min(p);
        return safe_div(p, std::pow(rho, params.gamma));
    }

    static double density_from_p_invariant(
        double p,
        double K,
        const EOSParams& params
    )
    {
        p = clamp_min(p);
        K = clamp_min(K);
        return std::pow(safe_div(p, K), safe_div(1.0, params.gamma));
    }
};


// [5] Tait barotropic liquid EOS
struct TaitEOS {
    template<int DIM>
    static double pressure(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        return clamp_min(tait_pressure_from_density(clamp_min(U.rho), params));
    }

    template<int DIM>
    static double sound_speed(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = clamp_min(U.rho);
        const double p = pressure<DIM>(U, params);
        return std::sqrt(clamp_min(params.gamma * safe_div(p - params.p0 + params.tait_B, rho)));
    }

    static double internal_energy(
        double rho,
        double,
        const EOSParams& params
    )
    {
        return tait_internal_energy_from_density(rho, params);
    }

    template<int DIM>
    static inline void compute_p_c(
        const Conserved<DIM>& U,
        const EOSParams& params,
        double& p,
        double& c
    )
    {
        p = pressure<DIM>(U, params);
        c = sound_speed<DIM>(U, params);
    }

    static double entropy_invariant(
        double,
        double p,
        const EOSParams&
    )
    {
        return clamp_min(p);
    }

    static double density_from_p_invariant(
        double p,
        double,
        const EOSParams& params
    )
    {
        const double B = clamp_min(params.tait_B);
        const double ratio = std::max((p - params.p0) / B + 1.0, 1e-12);
        return clamp_min(params.rho0) * std::pow(ratio, 1.0 / params.gamma);
    }
};


// [6] Mie-Gruneisen fluid EOS
struct MieGruneisenEOS {
    template<int DIM>
    static double pressure(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = clamp_min(U.rho);
        const double e = raw_internal_energy_from_conserved<DIM>(U, rho);
        return clamp_min(mie_gruneisen_pressure_from_rho_e(rho, e, params));
    }

    template<int DIM>
    static double sound_speed(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        const double rho = clamp_min(U.rho);
        const double e = raw_internal_energy_from_conserved<DIM>(U, rho);
        return numerical_sound_speed_from_rho_e(
            rho,
            e,
            params,
            mie_gruneisen_pressure_from_rho_e
        );
    }

    static double internal_energy(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        rho = clamp_min(rho);
        p = clamp_min(p);
        const double thermal_pressure = p - mie_gruneisen_cold_pressure(rho, params);
        return safe_div(thermal_pressure, (params.gamma - 1.0) * rho) +
            mie_gruneisen_cold_energy(rho, params);
    }

    template<int DIM>
    static inline void compute_p_c(
        const Conserved<DIM>& U,
        const EOSParams& params,
        double& p,
        double& c
    )
    {
        p = pressure<DIM>(U, params);
        c = sound_speed<DIM>(U, params);
    }

    static double entropy_invariant(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        rho = clamp_min(rho);
        p = clamp_min(p);
        const double thermal_pressure =
            clamp_min(p - mie_gruneisen_cold_pressure(rho, params));
        return safe_div(thermal_pressure, std::pow(rho, params.gamma));
    }

    static double density_from_p_invariant(
        double p,
        double K,
        const EOSParams& params
    )
    {
        p = clamp_min(p);
        K = clamp_min(K);
        double lo = 1.0e-6 * clamp_min(params.rho_ref);
        double hi = 100.0 * clamp_min(params.rho_ref);

        auto residual = [&](double rho) {
            return K * std::pow(clamp_min(rho), params.gamma) +
                mie_gruneisen_cold_pressure(rho, params) - p;
        };

        while (residual(lo) > 0.0 && lo > 1.0e-12 * clamp_min(params.rho_ref)) {
            lo *= 0.5;
        }

        while (residual(hi) < 0.0 && hi < 1.0e6 * clamp_min(params.rho_ref)) {
            hi *= 2.0;
        }

        for (int iter = 0; iter < 100; ++iter) {
            const double mid = 0.5 * (lo + hi);
            if (residual(mid) < 0.0) {
                lo = mid;
            }
            else {
                hi = mid;
            }
        }

        return clamp_min(0.5 * (lo + hi));
    }
};


// [7] Runtime-dispatched EOS used when different materials have different EOS kinds
struct MaterialEOS {
    template<int DIM>
    static double pressure(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        switch (params.kind) {
            case EOSKind::ideal_gas:
                return IdealGasEOS::pressure<DIM>(U, params);
            case EOSKind::stiffened_gas:
                return StiffenedGasEOS::pressure<DIM>(U, params);
            case EOSKind::noble_abel:
                return NobleAbelGasEOS::pressure<DIM>(U, params);
            case EOSKind::peng_robinson:
                return PengRobinsonEOS::pressure<DIM>(U, params);
            case EOSKind::tait:
                return TaitEOS::pressure<DIM>(U, params);
            case EOSKind::mie_gruneisen:
                return MieGruneisenEOS::pressure<DIM>(U, params);
        }

        throw std::runtime_error("MaterialEOS::pressure: unsupported EOS kind");
    }

    template<int DIM>
    static double sound_speed(
        const Conserved<DIM>& U,
        const EOSParams& params
    )
    {
        switch (params.kind) {
            case EOSKind::ideal_gas:
                return IdealGasEOS::sound_speed<DIM>(U, params);
            case EOSKind::stiffened_gas:
                return StiffenedGasEOS::sound_speed<DIM>(U, params);
            case EOSKind::noble_abel:
                return NobleAbelGasEOS::sound_speed<DIM>(U, params);
            case EOSKind::peng_robinson:
                return PengRobinsonEOS::sound_speed<DIM>(U, params);
            case EOSKind::tait:
                return TaitEOS::sound_speed<DIM>(U, params);
            case EOSKind::mie_gruneisen:
                return MieGruneisenEOS::sound_speed<DIM>(U, params);
        }

        throw std::runtime_error("MaterialEOS::sound_speed: unsupported EOS kind");
    }

    static double internal_energy(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        switch (params.kind) {
            case EOSKind::ideal_gas:
                return IdealGasEOS::internal_energy(rho, p, params);
            case EOSKind::stiffened_gas:
                return StiffenedGasEOS::internal_energy(rho, p, params);
            case EOSKind::noble_abel:
                return NobleAbelGasEOS::internal_energy(rho, p, params);
            case EOSKind::peng_robinson:
                return PengRobinsonEOS::internal_energy(rho, p, params);
            case EOSKind::tait:
                return TaitEOS::internal_energy(rho, p, params);
            case EOSKind::mie_gruneisen:
                return MieGruneisenEOS::internal_energy(rho, p, params);
        }

        throw std::runtime_error("MaterialEOS::internal_energy: unsupported EOS kind");
    }

    template<int DIM>
    static inline void compute_p_c(
        const Conserved<DIM>& U,
        const EOSParams& params,
        double& p,
        double& c
    )
    {
        p = pressure<DIM>(U, params);
        c = sound_speed<DIM>(U, params);
    }

    static double entropy_invariant(
        double rho,
        double p,
        const EOSParams& params
    )
    {
        switch (params.kind) {
            case EOSKind::ideal_gas:
                return IdealGasEOS::entropy_invariant(rho, p, params);
            case EOSKind::stiffened_gas:
                return StiffenedGasEOS::entropy_invariant(rho, p, params);
            case EOSKind::noble_abel:
                return NobleAbelGasEOS::entropy_invariant(rho, p, params);
            case EOSKind::peng_robinson:
                return PengRobinsonEOS::entropy_invariant(rho, p, params);
            case EOSKind::tait:
                return TaitEOS::entropy_invariant(rho, p, params);
            case EOSKind::mie_gruneisen:
                return MieGruneisenEOS::entropy_invariant(rho, p, params);
        }

        throw std::runtime_error("MaterialEOS::entropy_invariant: unsupported EOS kind");
    }

    static double density_from_p_invariant(
        double p,
        double K,
        const EOSParams& params
    )
    {
        switch (params.kind) {
            case EOSKind::ideal_gas:
                return IdealGasEOS::density_from_p_invariant(p, K, params);
            case EOSKind::stiffened_gas:
                return StiffenedGasEOS::density_from_p_invariant(p, K, params);
            case EOSKind::noble_abel:
                return NobleAbelGasEOS::density_from_p_invariant(p, K, params);
            case EOSKind::peng_robinson:
                return PengRobinsonEOS::density_from_p_invariant(p, K, params);
            case EOSKind::tait:
                return TaitEOS::density_from_p_invariant(p, K, params);
            case EOSKind::mie_gruneisen:
                return MieGruneisenEOS::density_from_p_invariant(p, K, params);
        }

        throw std::runtime_error("MaterialEOS::density_from_p_invariant: unsupported EOS kind");
    }
};


inline bool eos_has_stiffened_gas_wave_curve(const EOSParams& params)
{
    return params.kind == EOSKind::ideal_gas ||
           params.kind == EOSKind::stiffened_gas;
}




