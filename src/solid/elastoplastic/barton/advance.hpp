#pragma once

// Barton finite-volume advance routines for 1D and tensor multidimensional paths.

#include "src/solid/elastoplastic/barton/plasticity.hpp"

// 1D finite-volume step.

// Single-step finite-volume update for the 1D Barton plate-impact solver.

namespace solid::barton {

inline double max_signal_speed(const std::vector<State>& U, const Material& mat)
{
    double max_speed = 1.0;
    for (const auto& cell : U) {
        const Primitive P = cons_to_prim(cell, mat);
        max_speed = std::max(max_speed, std::abs(P.u) + P.wave_speed);
    }
    return max_speed;
}

inline double advance(std::vector<State>& U, const Material& mat, double dx, double cfl, double max_dt,
                      const BoundaryConditions& bc)
{
    const int n = static_cast<int>(U.size());
    const double dt = std::min(max_dt, cfl * dx / max_signal_speed(U, mat));

    std::vector<State> extended(n + 2);
    for (int i = 0; i < n; ++i) {
        extended[i + 1] = U[i];
    }
    extended[0] = ghost(U.front(), bc, true, mat);
    extended[n + 1] = ghost(U.back(), bc, false, mat);

    std::vector<State> left_cell(n + 2);
    std::vector<State> right_cell(n + 2);
    std::vector<State> half_cell(n + 2);
    half_cell[0] = extended[0];
    half_cell[n + 1] = extended[n + 1];

    for (int i = 1; i <= n; ++i) {
        const StateSlope slope = limited_slope(extended[i - 1], extended[i], extended[i + 1]);
        left_cell[i] = add_slope(extended[i], slope, -0.5);
        right_cell[i] = add_slope(extended[i], slope, 0.5);
        enforce(left_cell[i], mat);
        enforce(right_cell[i], mat);
        half_cell[i] = hancock_predict_cell(extended[i], left_cell[i], right_cell[i], mat, dt, dx);
    }
    half_cell[0] = ghost(half_cell[1], bc, true, mat);
    half_cell[n + 1] = ghost(half_cell[n], bc, false, mat);

    std::vector<State> left_half(n + 2);
    std::vector<State> right_half(n + 2);
    for (int i = 1; i <= n; ++i) {
        const StateSlope slope = limited_slope(half_cell[i - 1], half_cell[i], half_cell[i + 1]);
        left_half[i] = add_slope(half_cell[i], slope, -0.5);
        right_half[i] = add_slope(half_cell[i], slope, 0.5);
        enforce(left_half[i], mat);
        enforce(right_half[i], mat);
    }

    std::vector<State> face_flux(n + 1);
    std::vector<double> face_rhoF11(n + 1, 0.0);
    face_flux[0] = rusanov_flux(half_cell[0], left_half[1], mat);
    face_rhoF11[0] = 0.5 * (half_cell[0].rhoF11 + left_half[1].rhoF11);
    for (int i = 1; i < n; ++i) {
        face_flux[i] = rusanov_flux(right_half[i], left_half[i + 1], mat);
        face_rhoF11[i] = 0.5 * (right_half[i].rhoF11 + left_half[i + 1].rhoF11);
    }
    face_flux[n] = rusanov_flux(right_half[n], half_cell[n + 1], mat);
    face_rhoF11[n] = 0.5 * (right_half[n].rhoF11 + half_cell[n + 1].rhoF11);

    std::vector<State> next = U;
    for (int i = 0; i < n; ++i) {
        next[i] = U[i] - (dt / dx) * (face_flux[i + 1] - face_flux[i]);
    }

    for (int i = 0; i < n; ++i) {
        const Primitive P = cons_to_prim(U[i], mat);
        const double beta = (face_rhoF11[i + 1] - face_rhoF11[i]) / dx;
        next[i].rhoF11 -= dt * P.u * beta;
        enforce(next[i], mat);
        apply_plastic_relaxation(next[i], mat, dt);
    }

    U.swap(next);
    return dt;
}

} // namespace solid::barton

// Tensor compatibility and plastic source updates.

// Compatibility and plasticity source update for the 2D tensor formulation.

namespace solid::barton {

inline double tensor_dt(const std::vector<TensorState2D>& U, const TensorMaterial& mat,
                        double dx, double dy, double cfl, double max_dt)
{
    double sx = 1.0;
    double sy = 1.0;
    for (const auto& cell : U) {
        const TensorPrim2D P = tensor_prim(cell, mat);
        sx = std::max(sx, std::abs(P.vel[0]) + P.wave_speed);
        sy = std::max(sy, std::abs(P.vel[1]) + P.wave_speed);
    }
    return std::min(max_dt, cfl * std::min(dx / sx, dy / sy));
}

inline void apply_tensor_compatibility_and_plasticity(
    std::vector<TensorState2D>& next,
    const std::vector<TensorState2D>& old,
    const TensorMaterial& mat,
    int nx,
    int ny,
    double dx,
    double dy,
    double dt)
{
    auto sample = [&](int i, int j) {
        i = std::clamp(i, 0, nx - 1);
        j = std::clamp(j, 0, ny - 1);
        return old[hidx(i, j, nx)];
    };
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            TensorState2D& U = next[hidx(i, j, nx)];
            const TensorPrim2D Pold = tensor_prim(old[hidx(i, j, nx)], mat);
            std::array<double, 3> beta{};
            for (int col = 0; col < 3; ++col) {
                beta[col] =
                    (sample(i + 1, j).rhoF[3 * 0 + col] - sample(i - 1, j).rhoF[3 * 0 + col]) / (2.0 * dx) +
                    (sample(i, j + 1).rhoF[3 * 1 + col] - sample(i, j - 1).rhoF[3 * 1 + col]) / (2.0 * dy);
            }
            const std::array<double, 3> vel{Pold.vel[0], Pold.vel[1], 0.0};
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    U.rhoF[3 * row + col] -= dt * vel[row] * beta[col];
                }
            }

            TensorPrim2D P = tensor_prim(U, mat);
            const double mean = (P.sigma[0] + P.sigma[4] + P.sigma[8]) / 3.0;
            std::array<double, 9> dev = P.sigma;
            dev[0] -= mean;
            dev[4] -= mean;
            dev[8] -= mean;
            double sigmaI = 0.0;
            for (double value : dev) sigmaI += value * value;
            sigmaI = std::sqrt(1.5 * sigmaI);
            const double tau = relaxation_time(sigmaI, {mat.rho0, mat.b0 * mat.b0 * mat.rho0, mat.sigma0, mat.tau0, mat.relaxation_n});
            const double a = std::clamp(dt / (2.0 * std::max(mat.b0 * mat.b0 * P.rho, 1.0) * tau), 0.0, 0.2);
            if (sigmaI > mat.sigma0 && a > 0.0) {
                auto F = P.F;
                std::array<double, 9> newF = F;
                for (int row = 0; row < 3; ++row) {
                    for (int col = 0; col < 3; ++col) {
                        double corr = 0.0;
                        for (int k = 0; k < 3; ++k) {
                            corr += dev[3 * row + k] * F[3 * k + col];
                        }
                        newF[3 * row + col] -= a * corr;
                    }
                }
                const double old_det = det3(F);
                const double new_det = det3(newF);
                if (std::isfinite(new_det) && std::abs(new_det) > 1.0e-14) {
                    const double fix = std::cbrt(std::abs(old_det / new_det));
                    for (double& value : newF) value *= fix;
                }
                for (int q = 0; q < 9; ++q) U.rhoF[q] = U.rho * newF[q];
            }
            enforce_tensor(U, mat);
        }
    }
}

} // namespace solid::barton

// Tensor directional sweeps.

// Directionally split MUSCL-Hancock sweeps for the Barton tensor multidimensional tensor solver.

namespace solid::barton {

inline TensorState2D hancock_predict_tensor_cell(
    const TensorState2D& centre,
    const TensorState2D& left_face,
    const TensorState2D& right_face,
    const TensorMaterial& mat,
    double dt,
    double dx,
    int dir)
{
    const TensorState2D FL = tensor_flux(left_face, mat, dir);
    const TensorState2D FR = tensor_flux(right_face, mat, dir);
    TensorState2D predicted = centre - (0.5 * dt / dx) * (FR - FL);
    enforce_tensor(predicted, mat);
    return predicted;
}

inline double tensor_split_dt(const std::vector<TensorState2D>& U, const TensorMaterial& mat,
                              double dx, double dy, double cfl, double max_dt)
{
    double sx = 1.0;
    double sy = 1.0;
    const int count = static_cast<int>(U.size());
#pragma omp parallel for reduction(max:sx,sy) schedule(static) if(count > 1024)
    for (int i = 0; i < count; ++i) {
        const TensorPrim2D P = tensor_prim(U[i], mat);
        sx = std::max(sx, std::abs(P.vel[0]) + P.wave_speed);
        sy = std::max(sy, std::abs(P.vel[1]) + P.wave_speed);
    }
    const double denom = sx / dx + sy / dy;
    return std::min(max_dt, cfl / std::max(denom, 1.0e-30));
}

inline double tensor_flattening_scale(
    const TensorState2D& left,
    const TensorState2D& centre,
    const TensorState2D& right,
    const TensorMaterial& mat)
{
    const TensorPrim2D PL = tensor_prim(left, mat);
    const TensorPrim2D PC = tensor_prim(centre, mat);
    const TensorPrim2D PR = tensor_prim(right, mat);
    double jump = std::abs(PR.p - PL.p);
    for (int q = 0; q < 9; ++q) {
        jump = std::max(jump, std::abs(PR.sigma[q] - PL.sigma[q]));
    }
    const double modulus = std::max(
        material_bulk_modulus(PC.rho, PC.T, mat) + 4.0 * PC.rho * mat.b0 * mat.b0 / 3.0,
        1.0e6);
    const double z = jump / modulus;
    constexpr double z0 = 0.05;
    constexpr double z1 = 0.20;
    if (z <= z0) {
        return 1.0;
    }
    if (z >= z1) {
        return 0.0;
    }
    return 1.0 - (z - z0) / (z1 - z0);
}

inline TensorState2D tensor_sweep_ghost(const TensorState2D& U, int dir, bool lo)
{
    (void)lo;
    return dir == 0 ? tensor_reflect_x(U) : tensor_reflect_y(U);
}

inline void advance_tensor_sweep(
    std::vector<TensorState2D>& U,
    const TensorMaterial& mat,
    int nx,
    int ny,
    double h,
    double dt,
    int dir)
{
    const int lines = dir == 0 ? ny : nx;
    const int n = dir == 0 ? nx : ny;
    auto index = [&](int a, int line) {
        return dir == 0 ? hidx(a, line, nx) : hidx(line, a, nx);
    };

#pragma omp parallel for schedule(dynamic, 1) if(lines * n > 4096)
    for (int line = 0; line < lines; ++line) {
        std::vector<TensorState2D> ext(n + 2);
        for (int a = 0; a < n; ++a) {
            ext[a + 1] = U[index(a, line)];
        }
        ext[0] = tensor_sweep_ghost(ext[1], dir, true);
        ext[n + 1] = ext[n];

        std::vector<TensorState2D> left_cell(n + 2);
        std::vector<TensorState2D> right_cell(n + 2);
        std::vector<TensorState2D> half_cell(n + 2);
        half_cell[0] = ext[0];
        half_cell[n + 1] = ext[n + 1];
        for (int a = 1; a <= n; ++a) {
            const TensorSlope2D slope = scale_tensor_slope(
                limited_tensor_slope(ext[a - 1], ext[a], ext[a + 1]),
                tensor_flattening_scale(ext[a - 1], ext[a], ext[a + 1], mat));
            left_cell[a] = add_tensor_slope(ext[a], slope, -0.5);
            right_cell[a] = add_tensor_slope(ext[a], slope, 0.5);
            enforce_tensor(left_cell[a], mat);
            enforce_tensor(right_cell[a], mat);
            half_cell[a] = hancock_predict_tensor_cell(ext[a], left_cell[a], right_cell[a], mat, dt, h, dir);
        }
        half_cell[0] = tensor_sweep_ghost(half_cell[1], dir, true);
        half_cell[n + 1] = half_cell[n];

        std::vector<TensorState2D> left_half(n + 2);
        std::vector<TensorState2D> right_half(n + 2);
        for (int a = 1; a <= n; ++a) {
            const TensorSlope2D slope = scale_tensor_slope(
                limited_tensor_slope(half_cell[a - 1], half_cell[a], half_cell[a + 1]),
                tensor_flattening_scale(half_cell[a - 1], half_cell[a], half_cell[a + 1], mat));
            left_half[a] = add_tensor_slope(half_cell[a], slope, -0.5);
            right_half[a] = add_tensor_slope(half_cell[a], slope, 0.5);
            enforce_tensor(left_half[a], mat);
            enforce_tensor(right_half[a], mat);
        }

        std::vector<TensorState2D> flux(n + 1);
        std::vector<std::array<double, 3>> faceF(n + 1);
        flux[0] = tensor_rusanov(tensor_sweep_ghost(left_half[1], dir, true), left_half[1], mat, dir);
        for (int col = 0; col < 3; ++col) {
            faceF[0][col] = 0.5 * (
                tensor_sweep_ghost(left_half[1], dir, true).rhoF[3 * dir + col] +
                left_half[1].rhoF[3 * dir + col]);
        }
        for (int a = 1; a < n; ++a) {
            flux[a] = tensor_rusanov(right_half[a], left_half[a + 1], mat, dir);
            for (int col = 0; col < 3; ++col) {
                faceF[a][col] = 0.5 * (
                    right_half[a].rhoF[3 * dir + col] +
                    left_half[a + 1].rhoF[3 * dir + col]);
            }
        }
        flux[n] = tensor_rusanov(right_half[n], right_half[n], mat, dir);
        for (int col = 0; col < 3; ++col) {
            faceF[n][col] = right_half[n].rhoF[3 * dir + col];
        }

        std::vector<TensorState2D> old_line(n);
        for (int a = 0; a < n; ++a) {
            old_line[a] = U[index(a, line)];
        }
        for (int a = 0; a < n; ++a) {
            TensorState2D next = old_line[a] - (dt / h) * (flux[a + 1] - flux[a]);
            const TensorPrim2D Pold = tensor_prim(old_line[a], mat);
            const std::array<double, 3> vel{Pold.vel[0], Pold.vel[1], 0.0};
            for (int col = 0; col < 3; ++col) {
                const double beta = (faceF[a + 1][col] - faceF[a][col]) / h;
                for (int row = 0; row < 3; ++row) {
                    next.rhoF[3 * row + col] -= dt * vel[row] * beta;
                }
            }
            enforce_tensor(next, mat);
            U[index(a, line)] = next;
        }
    }
}

inline double advance_tensor_2d(std::vector<TensorState2D>& U, const TensorMaterial& mat,
                                int nx, int ny, double dx, double dy, double cfl, double max_dt)
{
    const double dt = tensor_split_dt(U, mat, dx, dy, cfl, max_dt);
    advance_tensor_sweep(U, mat, nx, ny, dx, dt, 0);
    advance_tensor_sweep(U, mat, nx, ny, dy, dt, 1);
    apply_tensor_plastic_relaxation(U, mat, dt);
    return dt;
}

} // namespace solid::barton

// Cylindrical tensor reference update.

// One-dimensional cylindrical reference update for Barton tensor multidimensional validation.

namespace solid::barton {

inline double tensor_cyl_dt(
    const std::vector<TensorState2D>& U,
    const TensorMaterial& mat,
    double dr,
    double cfl,
    double max_dt)
{
    double speed = 1.0;
    const int count = static_cast<int>(U.size());
#pragma omp parallel for reduction(max:speed) schedule(static) if(count > 1024)
    for (int i = 0; i < count; ++i) {
        const TensorPrim2D P = tensor_prim(U[i], mat);
        speed = std::max(speed, std::abs(P.vel[0]) + P.wave_speed);
    }
    return std::min(max_dt, cfl * dr / speed);
}

inline double advance_tensor_cyl(
    std::vector<TensorState2D>& U,
    const TensorMaterial& mat,
    double r_min,
    double dr,
    double cfl,
    double max_dt)
{
    const int n = static_cast<int>(U.size());
    const double dt = tensor_cyl_dt(U, mat, dr, cfl, max_dt);

    std::vector<TensorState2D> ext(n + 2);
    for (int i = 0; i < n; ++i) {
        ext[i + 1] = U[i];
    }
    ext[0] = tensor_reflect_x(U.front());
    ext[n + 1] = U.back();

    std::vector<TensorState2D> left_cell(n + 2);
    std::vector<TensorState2D> right_cell(n + 2);
    std::vector<TensorState2D> half_cell(n + 2);
    half_cell[0] = ext[0];
    half_cell[n + 1] = ext[n + 1];
    for (int i = 1; i <= n; ++i) {
        const TensorSlope2D slope = scale_tensor_slope(
            limited_tensor_slope(ext[i - 1], ext[i], ext[i + 1]),
            tensor_flattening_scale(ext[i - 1], ext[i], ext[i + 1], mat));
        left_cell[i] = add_tensor_slope(ext[i], slope, -0.5);
        right_cell[i] = add_tensor_slope(ext[i], slope, 0.5);
        enforce_tensor(left_cell[i], mat);
        enforce_tensor(right_cell[i], mat);
        half_cell[i] = hancock_predict_tensor_cell(ext[i], left_cell[i], right_cell[i], mat, dt, dr, 0);
    }
    half_cell[0] = tensor_reflect_x(half_cell[1]);
    half_cell[n + 1] = half_cell[n];

    std::vector<TensorState2D> left_half(n + 2);
    std::vector<TensorState2D> right_half(n + 2);
    for (int i = 1; i <= n; ++i) {
        const TensorSlope2D slope = scale_tensor_slope(
            limited_tensor_slope(half_cell[i - 1], half_cell[i], half_cell[i + 1]),
            tensor_flattening_scale(half_cell[i - 1], half_cell[i], half_cell[i + 1], mat));
        left_half[i] = add_tensor_slope(half_cell[i], slope, -0.5);
        right_half[i] = add_tensor_slope(half_cell[i], slope, 0.5);
        enforce_tensor(left_half[i], mat);
        enforce_tensor(right_half[i], mat);
    }

    std::vector<TensorState2D> flux(n + 1);
    std::vector<std::array<double, 3>> faceF(n + 1);
    flux[0] = tensor_rusanov(tensor_reflect_x(left_half[1]), left_half[1], mat, 0);
    for (int col = 0; col < 3; ++col) {
        faceF[0][col] = 0.5 * (
            tensor_reflect_x(left_half[1]).rhoF[3 * 0 + col] +
            left_half[1].rhoF[3 * 0 + col]);
    }
    for (int i = 1; i < n; ++i) {
        flux[i] = tensor_rusanov(right_half[i], left_half[i + 1], mat, 0);
        for (int col = 0; col < 3; ++col) {
            faceF[i][col] = 0.5 * (
                right_half[i].rhoF[3 * 0 + col] +
                left_half[i + 1].rhoF[3 * 0 + col]);
        }
    }
    flux[n] = tensor_rusanov(right_half[n], right_half[n], mat, 0);
    for (int col = 0; col < 3; ++col) {
        faceF[n][col] = right_half[n].rhoF[3 * 0 + col];
    }

    std::vector<TensorState2D> next = U;
    for (int i = 0; i < n; ++i) {
        const double r = std::max(r_min + (i + 0.5) * dr, 0.5 * dr);
        const double r_lo = std::max(r_min + i * dr, 0.0);
        const double r_hi = r_min + (i + 1.0) * dr;
        next[i] = U[i] - (dt / (r * dr)) * (r_hi * flux[i + 1] - r_lo * flux[i]);

        const TensorPrim2D Pold = tensor_prim(U[i], mat);
        next[i].mom[0] += dt * (-Pold.sigma[4]) / r;

        const std::array<double, 3> vel{Pold.vel[0], 0.0, 0.0};
        for (int col = 0; col < 3; ++col) {
            const double beta = (r_hi * faceF[i + 1][col] - r_lo * faceF[i][col]) / (r * dr);
            for (int row = 0; row < 3; ++row) {
                next[i].rhoF[3 * row + col] -= dt * vel[row] * beta;
            }
        }

        next[i].rhoF[4] += dt * U[i].rhoF[4] * Pold.vel[0] / r;
        enforce_tensor(next[i], mat);
        apply_tensor_plastic_relaxation(next[i], mat, dt);
    }

    U.swap(next);
    return dt;
}

} // namespace solid::barton
