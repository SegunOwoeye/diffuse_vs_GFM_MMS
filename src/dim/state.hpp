#pragma once

#include <array>
#include <algorithm>


// [0] Multimaterial Conserved State
template<int DIM, int NMAT>
struct Conserved {
    std::array<double, NMAT> arho{}; // arho[k] = alpha[k] * rho[k]
    std::array<double, DIM> mom{}; // mom = total momentum
    double E = 0.0; // E = total energy
    std::array<double, NMAT> alpha{}; // alpha = volume fractions (non-conservative)
};


// [1] Multimaterial Primitive State
template<int DIM, int NMAT>
struct Primitive {
    std::array<double, NMAT> rho{}; // rho[k] = material densities
    std::array<double, NMAT> alpha{}; // alpha = volume fractions
    std::array<double, DIM> vel{}; // vel = velocity
    double p = 0.0; // p = pressure

};


// [2] Addition
template<int DIM, int NMAT>
inline Conserved<DIM, NMAT> operator+(
    const Conserved<DIM, NMAT>& A,
    const Conserved<DIM, NMAT>& B
) {
    Conserved<DIM, NMAT> C;

    for (int k = 0; k < NMAT; ++k) {
        C.arho[k] = A.arho[k] + B.arho[k];
        C.alpha[k] = A.alpha[k] + B.alpha[k]; // algebra only
    }

    for (int d = 0; d < DIM; ++d) {
        C.mom[d] = A.mom[d] + B.mom[d];
    }

    C.E = A.E + B.E;

    return C;
}


// [3] Subtraction
template<int DIM, int NMAT>
inline Conserved<DIM, NMAT> operator-(
    const Conserved<DIM, NMAT>& A,
    const Conserved<DIM, NMAT>& B
) {
    Conserved<DIM, NMAT> C;

    for (int k = 0; k < NMAT; ++k) {
        C.arho[k] = A.arho[k] - B.arho[k];
        C.alpha[k] = A.alpha[k] - B.alpha[k];
    }

    for (int d = 0; d < DIM; ++d) {
        C.mom[d] = A.mom[d] - B.mom[d];
    }

    C.E = A.E - B.E;

    return C;
}


// [4] Scalar multiplication
template<int DIM, int NMAT>
inline Conserved<DIM, NMAT> operator*(
    double a,
    const Conserved<DIM, NMAT>& U
) {
    Conserved<DIM, NMAT> R;

    for (int k = 0; k < NMAT; ++k) {
        R.arho[k] = a * U.arho[k];
        R.alpha[k] = a * U.alpha[k];
    }

    for (int d = 0; d < DIM; ++d) {
        R.mom[d] = a * U.mom[d];
    }

    R.E = a * U.E;

    return R;
}


// [5] Scalar division
template<int DIM, int NMAT>
inline Conserved<DIM, NMAT> operator/(
    const Conserved<DIM, NMAT>& U,
    double s
) {
    Conserved<DIM, NMAT> R;

    for (int k = 0; k < NMAT; ++k) {
        R.arho[k] = U.arho[k] / s;
        R.alpha[k] = U.alpha[k] / s;
    }

    for (int d = 0; d < DIM; ++d) {
        R.mom[d] = U.mom[d] / s;
    }

    R.E = U.E / s;

    return R;
}


// [6] Total density
template<int DIM, int NMAT>
inline double total_density(const Conserved<DIM, NMAT>& U)
{
    double rho = 0.0;

    for (int k = 0; k < NMAT; ++k) {
        rho += U.arho[k];
    }

    return rho;
}


// [7] Renormalise alpha (enforce sum = 1)
template<int NMAT>
inline void renormalise_alpha(std::array<double, NMAT>& alpha)
{
    double sum = 0.0;

    for (int k = 0; k < NMAT; ++k) {
        alpha[k] = std::max(alpha[k], 0.0);
        sum += alpha[k];
    }

    if (sum > 0.0) {
        for (int k = 0; k < NMAT; ++k) {
            alpha[k] /= sum;
        }
    }
}