#pragma once

#include <array>
#include <stdexcept>
#include <vector>

namespace dim {

    template<int DIM>
    struct State {
        std::vector<double> partial_mass{};
        std::array<double, DIM> mom{};
        double E = 0.0;
        std::vector<double> alpha{};
    };



    template<int DIM>
    struct Primitive {
        std::vector<double> rho{};
        std::vector<double> alpha{};
        std::array<double, DIM> vel{};
        double p = 0.0;
    };

    template<int DIM>
    struct Flux {
        std::vector<double> partial_mass{};
        std::array<double, DIM> mom{};
        double E = 0.0;
    };

    template<int DIM>
    struct RiemannResult {
        Flux<DIM> flux{};
        double face_velocity = 0.0;
    };

    template<int DIM>
    inline State<DIM> make_state(int nmat)
    {
        if (nmat <= 0) {
            throw std::runtime_error("dim::make_state: nmat must be positive");
        }

        State<DIM> U{};
        U.partial_mass.assign(nmat, 0.0);
        U.alpha.assign((nmat > 1) ? (nmat - 1) : 0, 0.0);
        return U;
    }

    template<int DIM>
    inline Flux<DIM> make_flux(int nmat)
    {
        if (nmat <= 0) {
            throw std::runtime_error("dim::make_flux: nmat must be positive");
        }

        Flux<DIM> F{};
        F.partial_mass.assign(nmat, 0.0);
        return F;
    }

    template<int DIM>
    inline void require_compatible_state(
        const State<DIM>& U,
        int nmat,
        const char* context
    )
    {
        if (static_cast<int>(U.partial_mass.size()) != nmat) {
            throw std::runtime_error(std::string(context) + ": partial_mass size mismatch");
        }

        const int expected_alpha = (nmat > 1) ? (nmat - 1) : 0;
        if (static_cast<int>(U.alpha.size()) != expected_alpha) {
            throw std::runtime_error(std::string(context) + ": alpha size mismatch");
        }
    }

} 
