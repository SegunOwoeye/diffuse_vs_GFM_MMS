#pragma once

#include <array>
#include <cmath>
#include <stdexcept>


// [0] Dot Product
template<int DIM>
inline double dot(
    const std::array<double, DIM>& a,
    const std::array<double, DIM>& b
)
{
    double result = 0.0;

    for (int d = 0; d < DIM; ++d) {
        result += a[d] * b[d];
    }

    return result;
}


// [1] Vector Addition
template<int DIM>
inline std::array<double, DIM> add(
    const std::array<double, DIM>& a,
    const std::array<double, DIM>& b
)
{
    std::array<double, DIM> result{};

    for (int d = 0; d < DIM; ++d) {
        result[d] = a[d] + b[d];
    }

    return result;
}


// [2] Vector Subtraction
template<int DIM>
inline std::array<double, DIM> sub(
    const std::array<double, DIM>& a,
    const std::array<double, DIM>& b
)
{
    std::array<double, DIM> result{};

    for (int d = 0; d < DIM; ++d) {
        result[d] = a[d] - b[d];
    }

    return result;
}


// [3] Scalar-Vector Multiplication
template<int DIM>
inline std::array<double, DIM> scale(
    double s,
    const std::array<double, DIM>& a
)
{
    std::array<double, DIM> result{};

    for (int d = 0; d < DIM; ++d) {
        result[d] = s * a[d];
    }

    return result;
}


// [4] Magnitude
template<int DIM>
inline double norm(
    const std::array<double, DIM>& a
)
{
    return std::sqrt(dot<DIM>(a, a));
}


// [5] Normalisation
template<int DIM>
inline std::array<double, DIM> normalize(
    const std::array<double, DIM>& a,
    double tol = 1e-14
)
{
    const double mag = norm<DIM>(a);

    if (mag < tol) {
        throw std::runtime_error("normalize: zero magnitude vector");
    }

    return scale<DIM>(1.0 / mag, a);
}


// [6] Projection onto direction n
template<int DIM>
inline std::array<double, DIM> project(
    const std::array<double, DIM>& a,
    const std::array<double, DIM>& n
)
{
    const double denom = dot<DIM>(n, n);

    if (denom < 1e-14) {
        throw std::runtime_error("project: zero direction vector");
    }

    const double coeff = dot<DIM>(a, n) / denom;

    return scale<DIM>(coeff, n);
}

