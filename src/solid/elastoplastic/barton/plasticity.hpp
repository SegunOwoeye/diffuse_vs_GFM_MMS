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

inline double material_equivalent_stress(const TensorPrim2D& P)
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
    const std::array<double, 9> newF = candidate(final_scale);
    for (int q = 0; q < 9; ++q) {
        U.rhoF[q] = U.rho * newF[q];
    }
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

} // namespace solid::barton
