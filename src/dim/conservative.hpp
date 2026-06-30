#pragma once

#include <stdexcept>

#include "src/dim/state.hpp"

namespace dim {

    // [1] Vector Addition
    template<int DIM>
    inline Flux<DIM> operator+(
        const Flux<DIM>& A,
        const Flux<DIM>& B
    )
    {
        if (A.partial_mass.size() != B.partial_mass.size()) {
            throw std::runtime_error("dim::Flux operator+: partial_mass size mismatch");
        }

        Flux<DIM> C = make_flux<DIM>(static_cast<int>(A.partial_mass.size()));

        for (int k = 0; k < static_cast<int>(A.partial_mass.size()); ++k) {
            C.partial_mass[k] = A.partial_mass[k] + B.partial_mass[k];
        }

        for (int d = 0; d < DIM; ++d) {
            C.mom[d] = A.mom[d] + B.mom[d];
        }

        C.E = A.E + B.E;
        for (int q = 0; q < 9; ++q) {
            C.solid_rhoF[q] = A.solid_rhoF[q] + B.solid_rhoF[q];
        }
        C.solid_rho_eqps = A.solid_rho_eqps + B.solid_rho_eqps;
        C.solid_rho_damage = A.solid_rho_damage + B.solid_rho_damage;
        return C;
    }

    // [2] Vector Subtraction
    template<int DIM>
    inline Flux<DIM> operator-(
        const Flux<DIM>& A,
        const Flux<DIM>& B
    )
    {
        if (A.partial_mass.size() != B.partial_mass.size()) {
            throw std::runtime_error("dim::Flux operator-: partial_mass size mismatch");
        }

        Flux<DIM> C = make_flux<DIM>(static_cast<int>(A.partial_mass.size()));

        for (int k = 0; k < static_cast<int>(A.partial_mass.size()); ++k) {
            C.partial_mass[k] = A.partial_mass[k] - B.partial_mass[k];
        }

        for (int d = 0; d < DIM; ++d) {
            C.mom[d] = A.mom[d] - B.mom[d];
        }

        C.E = A.E - B.E;
        for (int q = 0; q < 9; ++q) {
            C.solid_rhoF[q] = A.solid_rhoF[q] - B.solid_rhoF[q];
        }
        C.solid_rho_eqps = A.solid_rho_eqps - B.solid_rho_eqps;
        C.solid_rho_damage = A.solid_rho_damage - B.solid_rho_damage;
        return C;
    }

    // [3] Scalar-vector multiplication
    template<int DIM>
    inline Flux<DIM> operator*(
        double a,
        const Flux<DIM>& F
    )
    {
        Flux<DIM> result = make_flux<DIM>(static_cast<int>(F.partial_mass.size()));

        for (int k = 0; k < static_cast<int>(F.partial_mass.size()); ++k) {
            result.partial_mass[k] = a * F.partial_mass[k];
        }

        for (int d = 0; d < DIM; ++d) {
            result.mom[d] = a * F.mom[d];
        }

        result.E = a * F.E;
        for (int q = 0; q < 9; ++q) {
            result.solid_rhoF[q] = a * F.solid_rhoF[q];
        }
        result.solid_rho_eqps = a * F.solid_rho_eqps;
        result.solid_rho_damage = a * F.solid_rho_damage;
        return result;
    }

    // [4] Scalar-vector division
    template<int DIM>
    inline Flux<DIM> operator/(
        const Flux<DIM>& F,
        double s
    )
    {
        return (1.0 / s) * F;
    }

    // [5] For Riemann-solver algebra, HLLC jump terms.
    template<int DIM>
    inline Flux<DIM> state_difference(
        const State<DIM>& A,
        const State<DIM>& B
    )
    {
        if (A.partial_mass.size() != B.partial_mass.size()) {
            throw std::runtime_error("dim::state_difference: partial_mass size mismatch");
        }

        Flux<DIM> delta = make_flux<DIM>(static_cast<int>(A.partial_mass.size()));

        for (int k = 0; k < static_cast<int>(A.partial_mass.size()); ++k) {
            delta.partial_mass[k] = A.partial_mass[k] - B.partial_mass[k];
        }

        for (int d = 0; d < DIM; ++d) {
            delta.mom[d] = A.mom[d] - B.mom[d];
        }

        delta.E = A.E - B.E;
        for (int q = 0; q < 9; ++q) {
            delta.solid_rhoF[q] = A.solid_rhoF[q] - B.solid_rhoF[q];
        }
        delta.solid_rho_eqps = A.solid_rho_eqps - B.solid_rho_eqps;
        delta.solid_rho_damage = A.solid_rho_damage - B.solid_rho_damage;
        return delta;
    }

    // [6] Cell update step that applies the net face fluxes to the conserved variables
    template<int DIM>
    inline State<DIM> conservative_update(
        const State<DIM>& U,
        const Flux<DIM>& flux_left,
        const Flux<DIM>& flux_right,
        double lambda
    )
    {
        if (flux_left.partial_mass.size() != flux_right.partial_mass.size() ||
            flux_left.partial_mass.size() != U.partial_mass.size()) {
            throw std::runtime_error("dim::conservative_update: partial_mass size mismatch");
        }

        State<DIM> result = U;

        for (int k = 0; k < static_cast<int>(U.partial_mass.size()); ++k) {
            result.partial_mass[k] -= lambda * (flux_right.partial_mass[k] - flux_left.partial_mass[k]);
        }

        for (int d = 0; d < DIM; ++d) {
            result.mom[d] -= lambda * (flux_right.mom[d] - flux_left.mom[d]);
        }

        result.E -= lambda * (flux_right.E - flux_left.E);
        for (int q = 0; q < 9; ++q) {
            result.solid_rhoF[q] -= lambda * (flux_right.solid_rhoF[q] - flux_left.solid_rhoF[q]);
        }
        result.solid_rho_eqps -= lambda *
            (flux_right.solid_rho_eqps - flux_left.solid_rho_eqps);
        result.solid_rho_damage -= lambda *
            (flux_right.solid_rho_damage - flux_left.solid_rho_damage);
        return result;
    }


} 
