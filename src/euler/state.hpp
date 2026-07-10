#pragma once

#include <array>


// [0] Generic Conserved Euler State
template<int DIM>
struct Conserved {
    double rho = 0.0;
    std::array<double, DIM> mom{};
    double E = 0.0;
};


// [1] Primitive Euler State
template<int DIM>
struct Primitive {
    double rho = 0.0;
    std::array<double, DIM> vel{};
    double p = 0.0;
};


// [2] Conserved State Addition
template<int DIM>
inline Conserved<DIM> operator+(
    const Conserved<DIM>& A,
    const Conserved<DIM>& B
) {
    Conserved<DIM> C;
    C.rho = A.rho + B.rho;

    for (int d = 0; d < DIM; ++d)
        C.mom[d] = A.mom[d] + B.mom[d];

    C.E = A.E + B.E;

    return C;
}


// [3] Conserved State Subtraction
template<int DIM>
inline Conserved<DIM> operator-(
    const Conserved<DIM>& A,
    const Conserved<DIM>& B
) {
    Conserved<DIM> C;
    C.rho = A.rho - B.rho;

    for (int d = 0; d < DIM; ++d)
        C.mom[d] = A.mom[d] - B.mom[d];

    C.E = A.E - B.E;

    return C;
}


// [4] Scalar Multiplication
template<int DIM>
inline Conserved<DIM> operator*(
    double a,
    const Conserved<DIM>& U
) {
    Conserved<DIM> R;
    R.rho = a * U.rho;

    for (int d = 0; d < DIM; ++d)
        R.mom[d] = a * U.mom[d];

    R.E = a * U.E;

    return R;
}

// [5] Scalar Division
template<int DIM>
inline Conserved<DIM> operator/(
    const Conserved<DIM>& U, 
    double s)
{
    Conserved<DIM> R;
    R.rho = U.rho / s;

    for (int d = 0; d < DIM; ++d) {
        R.mom[d] = U.mom[d] / s;
    }

    R.E = U.E / s;

    return R;
}

