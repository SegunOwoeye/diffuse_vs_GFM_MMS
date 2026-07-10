#pragma once

#include <array>

#include "src/dim/conservative.hpp"
#include "src/dim/primitives.hpp"
#include "src/math/vector_ops.hpp"

namespace dim {

    template<int DIM>
    inline Flux<DIM> compute_flux_normal(
        const State<DIM>& U,
        const Primitive<DIM>& P,
        const std::array<double, DIM>& normal
    )
    {
        Flux<DIM> F = make_flux<DIM>(static_cast<int>(U.partial_mass.size()));

        const double mag = norm<DIM>(normal);

        std::array<double, DIM> n{};
        if (mag < 1e-14) {
            n[0] = 1.0;
        }
        else {
            n = scale<DIM>(1.0 / mag, normal);
        }

        const double rho_total = total_density(U);
        const double un = dot<DIM>(P.vel, n);

        for (int k = 0; k < static_cast<int>(U.partial_mass.size()); ++k) {
            F.partial_mass[k] = U.partial_mass[k] * un;
        }

        for (int d = 0; d < DIM; ++d) {
            F.mom[d] = rho_total * P.vel[d] * un + P.p * n[d];
        }

        F.E = (U.E + P.p) * un;
        return F;
    }

} 
