#pragma once

// Barton 1D and tensor plastic relaxation updates.

#include "src/solid/elastoplastic/barton/flux.hpp"

// 1D rate-dependent plastic relaxation.

// Rate-dependent plastic relaxation used by the 1D Barton aluminium flyer-plate tests.

namespace solid::barton {

inline double relaxation_time(double sigmaI, const Material& mat)
{
    sigmaI = std::max(std::abs(sigmaI), 1.0);
    const double log_tau =
        std::log(std::max(mat.tau0, 1.0e-30)) +
        mat.relaxation_n *
            (std::log(std::max(mat.sigma0, 1.0)) - std::log(sigmaI));
    return std::exp(std::clamp(log_tau, -40.0, 40.0));
}

inline double implicit_relaxation_scale(double trial_sigmaI, const Material& mat, double dt)
{
    trial_sigmaI = std::max(trial_sigmaI, 1.0);
    if (trial_sigmaI <= mat.sigma0 || dt <= 0.0) {
        return 1.0;
    }

    auto residual = [&](double scale) {
        const double relaxed_sigma = std::max(scale * trial_sigmaI, 1.0);
        const double tau = relaxation_time(relaxed_sigma, mat);
        return scale - 1.0 / (1.0 + dt / tau);
    };

    double lo = 0.0;
    double hi = 1.0;
    for (int iter = 0; iter < 80; ++iter) {
        const double mid = 0.5 * (lo + hi);
        if (residual(mid) <= 0.0) {
            lo = mid;
        }
        else {
            hi = mid;
        }
    }
    return std::clamp(0.5 * (lo + hi), 0.0, 1.0);
}

inline void apply_plastic_relaxation(State& U, const Material& mat, double dt)
{
    const double rho = clamp_min(U.rho, 1.0e-12);
    double F11 = U.rhoF11 / rho;
    double F22 = U.rhoF22 / rho;
    double F33 = U.rhoF33 / rho;

    const Primitive before = cons_to_prim(U, mat);
    if (before.sigmaI <= mat.sigma0 || before.sigmaI <= 1.0) {
        return;
    }

    const double J = clamp_min(F11 * F22 * F33, 1.0e-18);
    const double J13 = std::pow(J, 1.0 / 3.0);
    const double scale = implicit_relaxation_scale(before.sigmaI, mat, dt);
    const double l11 = std::log(clamp_min(F11 / J13, 1.0e-18));
    const double l22 = std::log(clamp_min(F22 / J13, 1.0e-18));
    const double l33 = std::log(clamp_min(F33 / J13, 1.0e-18));
    F11 = J13 * std::exp(scale * l11);
    F22 = J13 * std::exp(scale * l22);
    F33 = J13 * std::exp(scale * l33);

    U.rhoF11 = rho * clamp_min(F11, 1.0e-10);
    U.rhoF22 = rho * clamp_min(F22, 1.0e-10);
    U.rhoF33 = rho * clamp_min(F33, 1.0e-10);
    U.rhoPlastic += rho * (1.0 - scale);
    enforce(U, mat);
}

} // namespace solid::barton

// Tensor plastic relaxation.

// Material plastic relaxation model used by the Barton radial tensor solver.

namespace solid::barton {

template<int DIM>
inline double material_equivalent_stress(const TensorPrim<DIM>& P)
{
    const double mean = (P.sigma[0] + P.sigma[4] + P.sigma[8]) / 3.0;
    std::array<double, 9> dev = P.sigma;
    dev[0] -= mean;
    dev[4] -= mean;
    dev[8] -= mean;
    double j2 = 0.0;
    for (double value : dev) {
        j2 += value * value;
    }
    return std::sqrt(std::max(1.5 * j2, 0.0));
}

inline double material_relaxation_time(double sigmaI, const TensorMaterial& mat)
{
    sigmaI = std::max(std::abs(sigmaI), 1.0);
    const double log_tau =
        std::log(std::max(mat.tau0, 1.0e-30)) +
        mat.relaxation_n *
            (std::log(std::max(mat.sigma0, 1.0)) - std::log(sigmaI));
    return std::exp(std::clamp(log_tau, -40.0, 40.0));
}

template<int DIM>
inline double johnson_cook_failure_strain(
    const TensorPrim<DIM>& P,
    const TensorMaterial& mat,
    double plastic_strain_rate)
{
    const double sigma_eq = std::max(material_equivalent_stress(P), 1.0);
    const double mean_stress = (P.sigma[0] + P.sigma[4] + P.sigma[8]) / 3.0;
    const double triaxiality = std::clamp(mean_stress / sigma_eq, -5.0, 5.0);
    const double rate_ratio = std::max(
        plastic_strain_rate / std::max(mat.reference_plastic_strain_rate, 1.0e-30),
        1.0e-12);
    const double homologous_temperature = std::clamp(
        (P.T - mat.T0) / std::max(mat.melt_temperature - mat.T0, 1.0),
        0.0,
        1.0);

    const double stress_factor = mat.jc_D1 + mat.jc_D2 * std::exp(mat.jc_D3 * triaxiality);
    const double rate_factor = 1.0 + mat.jc_D4 * std::log(rate_ratio);
    const double temperature_factor = 1.0 + mat.jc_D5 * homologous_temperature;
    return std::max(stress_factor * rate_factor * temperature_factor, 1.0e-8);
}

template<int DIM>
inline void accumulate_johnson_cook_damage(
    TensorState<DIM>& U,
    const TensorPrim<DIM>& before,
    const TensorMaterial& mat,
    double dt,
    double eqps_increment)
{
    if (eqps_increment <= 0.0 || dt <= 0.0) {
        return;
    }

    const double rho = clamp_min(U.rho, 1.0e-12);
    const double old_eqps = std::max(finite_or(U.rhoEqps / rho, 0.0), 0.0);
    U.rhoEqps = rho * (old_eqps + eqps_increment);

    if (!mat.damage_enabled) {
        return;
    }

    const double rate = eqps_increment / dt;
    const double failure_strain = johnson_cook_failure_strain(before, mat, rate);
    const double old_damage = std::clamp(finite_or(U.rhoDamage / rho, 0.0), 0.0, mat.failed_damage);
    const double next_damage = std::min(mat.failed_damage, old_damage + eqps_increment / failure_strain);
    U.rhoDamage = rho * next_damage;
}

inline void apply_tensor_plastic_relaxation(TensorState2D& U, const TensorMaterial& mat, double dt)
{
    if (dt <= 0.0) {
        return;
    }
    TensorPrim2D P = tensor_prim(U, mat);
    const double sigmaI = material_equivalent_stress(P);
    if (sigmaI <= mat.sigma0 || sigmaI <= 1.0) {
        return;
    }

    const double detF = det3(P.F);
    if (!std::isfinite(detF) || std::abs(detF) < 1.0e-14) {
        return;
    }
    const double j13 = std::cbrt(std::abs(detF));
    std::array<double, 9> Fvol{j13, 0.0, 0.0, 0.0, j13, 0.0, 0.0, 0.0, j13};

    auto candidate = [&](double scale) {
        std::array<double, 9> F{};
        for (int q = 0; q < 9; ++q) {
            F[q] = Fvol[q] + scale * (P.F[q] - Fvol[q]);
        }
        const double d = det3(F);
        if (std::isfinite(d) && std::abs(d) > 1.0e-14) {
            const double fix = std::cbrt(std::abs(detF / d));
            for (double& value : F) {
                value *= fix;
            }
        }
        return F;
    };
    auto equiv_for_scale = [&](double scale) {
        const auto F = candidate(scale);
        TensorState2D trial = U;
        for (int q = 0; q < 9; ++q) {
            trial.rhoF[q] = trial.rho * F[q];
        }
        return material_equivalent_stress(tensor_prim(trial, mat));
    };

    double yield_scale = 0.0;
    if (equiv_for_scale(0.0) > mat.sigma0) {
        yield_scale = 0.0;
    }
    else {
        double lo = 0.0;
        double hi = 1.0;
        for (int iter = 0; iter < 60; ++iter) {
            const double mid = 0.5 * (lo + hi);
            if (equiv_for_scale(mid) <= mat.sigma0) {
                lo = mid;
            }
            else {
                hi = mid;
            }
        }
        yield_scale = lo;
    }

    const double tau = material_relaxation_time(sigmaI, mat);
    const double time_scale = 1.0 / (1.0 + dt / std::max(tau, 1.0e-30));
    const double final_scale = std::clamp(std::max(yield_scale, time_scale), 0.0, 1.0);
    const double final_sigmaI = equiv_for_scale(final_scale);
    const double shear_modulus = std::max(P.rho * mat.b0 * mat.b0, 1.0);
    const double eqps_increment = std::max(sigmaI - final_sigmaI, 0.0) / (3.0 * shear_modulus);
    const std::array<double, 9> newF = candidate(final_scale);
    for (int q = 0; q < 9; ++q) {
        U.rhoF[q] = U.rho * newF[q];
    }
    accumulate_johnson_cook_damage(U, P, mat, dt, eqps_increment);
    enforce_tensor(U, mat);
}

inline void apply_tensor_plastic_relaxation(TensorState3D& U, const TensorMaterial& mat, double dt)
{
    if (dt <= 0.0) {
        return;
    }
    TensorPrim3D P = tensor_prim(U, mat);
    const double sigmaI = material_equivalent_stress(P);
    if (sigmaI <= mat.sigma0 || sigmaI <= 1.0) {
        return;
    }

    const double detF = det3(P.F);
    if (!std::isfinite(detF) || std::abs(detF) < 1.0e-14) {
        return;
    }
    const double j13 = std::cbrt(std::abs(detF));
    std::array<double, 9> Fvol{j13, 0.0, 0.0, 0.0, j13, 0.0, 0.0, 0.0, j13};

    auto candidate = [&](double scale) {
        std::array<double, 9> F{};
        for (int q = 0; q < 9; ++q) {
            F[q] = Fvol[q] + scale * (P.F[q] - Fvol[q]);
        }
        const double d = det3(F);
        if (std::isfinite(d) && std::abs(d) > 1.0e-14) {
            const double fix = std::cbrt(std::abs(detF / d));
            for (double& value : F) {
                value *= fix;
            }
        }
        return F;
    };
    auto equiv_for_scale = [&](double scale) {
        const auto F = candidate(scale);
        TensorState3D trial = U;
        for (int q = 0; q < 9; ++q) {
            trial.rhoF[q] = trial.rho * F[q];
        }
        return material_equivalent_stress(tensor_prim(trial, mat));
    };

    double yield_scale = 0.0;
    if (equiv_for_scale(0.0) <= mat.sigma0) {
        double lo = 0.0;
        double hi = 1.0;
        for (int iter = 0; iter < 60; ++iter) {
            const double mid = 0.5 * (lo + hi);
            if (equiv_for_scale(mid) <= mat.sigma0) {
                lo = mid;
            }
            else {
                hi = mid;
            }
        }
        yield_scale = lo;
    }

    const double tau = material_relaxation_time(sigmaI, mat);
    const double time_scale = 1.0 / (1.0 + dt / std::max(tau, 1.0e-30));
    const double final_scale = std::clamp(std::max(yield_scale, time_scale), 0.0, 1.0);
    const double final_sigmaI = equiv_for_scale(final_scale);
    const double shear_modulus = std::max(P.rho * mat.b0 * mat.b0, 1.0);
    const double eqps_increment = std::max(sigmaI - final_sigmaI, 0.0) / (3.0 * shear_modulus);
    const std::array<double, 9> newF = candidate(final_scale);
    for (int q = 0; q < 9; ++q) {
        U.rhoF[q] = U.rho * newF[q];
    }
    accumulate_johnson_cook_damage(U, P, mat, dt, eqps_increment);
    enforce_tensor(U, mat);
}

inline void apply_tensor_plastic_relaxation(
    std::vector<TensorState2D>& U,
    const TensorMaterial& mat,
    double dt)
{
    const int count = static_cast<int>(U.size());
#pragma omp parallel for schedule(static) if(count > 1024)
    for (int i = 0; i < count; ++i) {
        apply_tensor_plastic_relaxation(U[i], mat, dt);
    }
}

inline void apply_tensor_plastic_relaxation(
    std::vector<TensorState3D>& U,
    const TensorMaterial& mat,
    double dt)
{
    const int count = static_cast<int>(U.size());
#pragma omp parallel for schedule(static) if(count > 1024)
    for (int i = 0; i < count; ++i) {
        apply_tensor_plastic_relaxation(U[i], mat, dt);
    }
}

inline TensorState2D tensor_reflect_x(const TensorState2D& U)
{
    TensorState2D G = U;
    G.mom[0] = -G.mom[0];
    G.rhoF[1] = -G.rhoF[1];
    G.rhoF[3] = -G.rhoF[3];
    return G;
}

inline TensorState2D tensor_reflect_y(const TensorState2D& U)
{
    TensorState2D G = U;
    G.mom[1] = -G.mom[1];
    G.rhoF[1] = -G.rhoF[1];
    G.rhoF[3] = -G.rhoF[3];
    return G;
}

inline TensorState3D tensor_reflect_x(const TensorState3D& U)
{
    TensorState3D G = U;
    G.mom[0] = -G.mom[0];
    G.rhoF[1] = -G.rhoF[1];
    G.rhoF[2] = -G.rhoF[2];
    G.rhoF[3] = -G.rhoF[3];
    G.rhoF[6] = -G.rhoF[6];
    return G;
}

inline TensorState3D tensor_reflect_y(const TensorState3D& U)
{
    TensorState3D G = U;
    G.mom[1] = -G.mom[1];
    G.rhoF[1] = -G.rhoF[1];
    G.rhoF[3] = -G.rhoF[3];
    G.rhoF[5] = -G.rhoF[5];
    G.rhoF[7] = -G.rhoF[7];
    return G;
}

inline TensorState3D tensor_reflect_z(const TensorState3D& U)
{
    TensorState3D G = U;
    G.mom[2] = -G.mom[2];
    G.rhoF[2] = -G.rhoF[2];
    G.rhoF[5] = -G.rhoF[5];
    G.rhoF[6] = -G.rhoF[6];
    G.rhoF[7] = -G.rhoF[7];
    return G;
}

} // namespace solid::barton
