#include <iostream>
#include <vector>
#include <array>
#include <cmath>

#include "src/dim/state.hpp"
#include "src/dim/primitives.hpp"
#include "src/dim/eos.hpp"
#include "src/dim/eos_params.hpp"
#include "src/dim/solver/advance_step.hpp"
#include "src/io/dim_write_csv.hpp"  


// ------------------------------------------------------------
// [0] Config
// ------------------------------------------------------------
constexpr int DIM  = 1;
constexpr int NMAT = 2;

using EOS = IdealGasEOS;


// ------------------------------------------------------------
// [1] Initialise Toro Test 1
// ------------------------------------------------------------
template<int DIM, int NMAT>
void initialise_toro1(
    std::vector<Conserved<DIM, NMAT>>& U,
    const std::array<int, DIM>& N,
    const std::array<double, DIM>& domain_min,
    const std::array<double, DIM>& dx,
    const EOSParams<NMAT>& params
)
{
    const int nx = N[0];

    U.resize(nx);

    for (int i = 0; i < nx; ++i) {

        double x = domain_min[0] + (i + 0.5) * dx[0];

        Primitive<DIM, NMAT> P{};

        // [1.1] Single-material via alpha
        P.alpha[0] = 1.0;
        P.alpha[1] = 0.0;

        if (x < 0.5) {
            P.rho[0] = 1.0;
            P.rho[1] = 1.0;
            P.vel[0] = 0.0;
            P.p      = 1.0;
        }
        else {
            P.rho[0] = 0.125;
            P.rho[1] = 0.125;
            P.vel[0] = 0.0;
            P.p      = 0.1;
        }

        U[i] = prim_to_cons<DIM, NMAT, EOS>(P, params);
    }
}


// ------------------------------------------------------------
// [2] Print snapshot
// ------------------------------------------------------------
template<int DIM, int NMAT>
void print_sample(
    const std::vector<Conserved<DIM, NMAT>>& U,
    int step
)
{
    std::cout << "Step " << step << "\n";

    for (int i = 0; i < static_cast<int>(U.size()); i += U.size() / 5) {

        double rho = 0.0;
        for (int k = 0; k < NMAT; ++k) {
            rho += U[i].arho[k];
        }

        double u = U[i].mom[0] / rho;

        std::cout << "i=" << i
                  << " rho=" << rho
                  << " u=" << u
                  << " E=" << U[i].E
                  << "\n";
    }

    std::cout << "\n";
}


// ------------------------------------------------------------
// [3] Main
// ------------------------------------------------------------
int main()
{
    // [3.1] Grid
    std::array<int, DIM> N{200};

    std::array<double, DIM> domain_min{0.0};
    std::array<double, DIM> domain_max{1.0};

    std::array<double, DIM> dx{};
    dx[0] = (domain_max[0] - domain_min[0]) / N[0];

    // [3.2] EOS params
    EOSParams<NMAT> params;
    params.gamma[0] = 1.4;
    params.gamma[1] = 1.4;

    // [3.3] Initialise
    std::vector<Conserved<DIM, NMAT>> U;

    initialise_toro1<DIM, NMAT>(
        U,
        N,
        domain_min,
        dx,
        params
    );

    // [3.4] Time loop
    double t = 0.0;
    const double tfinal = 0.25;
    const double cfl = 0.5;

    int step = 0;

    while (t < tfinal) {

        auto result = advance_one_step<DIM, NMAT, EOS>(
            U,
            N,
            dx,
            params,
            cfl,
            tfinal - t
        );

        U = result.U_new;

        t += result.dt;
        step++;

        if (step % 10 == 0) {
            print_sample<DIM, NMAT>(U, step);
        }
    }

    std::cout << "Finished at t=" << t << "\n";

    // --------------------------------------------------------
    // [3.5] Write final solution to CSV
    // --------------------------------------------------------
    write_csv<DIM, NMAT, EOS>(
        "toro1_dim.csv",
        U,
        N,
        domain_min,
        domain_max,
        params
    );

    std::cout << "CSV written: toro1_dim.csv\n";

    return 0;
}