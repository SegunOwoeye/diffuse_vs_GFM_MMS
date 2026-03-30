#pragma once

#include <algorithm>
#include <cmath>

#include "src/euler/state.hpp"
#include "src/euler/primitives.hpp"
#include "src/euler/eos_params.hpp"


// [0] Mixed Riemann Solver Result
struct MCRSResult1D {
    double p_star;
    double u_star;
    double rhoL_star;
    double rhoR_star;
};


// [1] Mixed-Component Riemann Solver in 1D Normal Direction
    // This solver currently uses ideal-gas-type shock and rarefaction relations
    // with possibly different gamma values on the left and right.
class MCRS1D {
public:
    // [1.1] Constructor
    MCRS1D(
        const Primitive<1>& PL,
        const Primitive<1>& PR,
        const EOSParams& paramsL,
        const EOSParams& paramsR,
        double tol = 1e-10,
        int max_iterations = 200
    )
    :
        PL_S(PL),
        PR_S(PR),
        paramsL_S(paramsL),
        paramsR_S(paramsR),
        tol_S(tol),
        max_iterations_S(max_iterations)
    {
        rhoL_S = std::max(PL_S.rho, tol_S);
        rhoR_S = std::max(PR_S.rho, tol_S);

        uL_S = PL_S.vel[0];
        uR_S = PR_S.vel[0];

        pL_S = std::max(PL_S.p, tol_S);
        pR_S = std::max(PR_S.p, tol_S);

        cL_S = std::sqrt(paramsL_S.gamma * pL_S / rhoL_S);
        cR_S = std::sqrt(paramsR_S.gamma * pR_S / rhoR_S);
    }


    // [1.2] Solve mixed-material Riemann problem
    inline MCRSResult1D solve() const
    {
        const double newton_tol = std::max(tol_S, 1e-12);
        const double p_floor = 1e-12;
        const double rho_floor = 1e-12;

        // [1.2.1] PVRS initial guess
        const double rho_bar = 0.5 * (rhoL_S + rhoR_S);
        const double a_bar = 0.5 * (cL_S + cR_S);

        const double pPV = 0.5 * (pL_S + pR_S)
            - 0.5 * (uR_S - uL_S) * rho_bar * a_bar;

        double p_star = std::max(pPV, p_floor);

        // [1.2.2] Newton iteration
        for (int n = 0; n < max_iterations_S; ++n) {
            const double F = fL(p_star) + fR(p_star) + (uR_S - uL_S);

            if (std::abs(F) < newton_tol) {
                break;
            }

            const double dp = std::max(1e-6 * p_star, 1e-12);
            const double F2 = fL(p_star + dp) + fR(p_star + dp) + (uR_S - uL_S);
            const double dF = (F2 - F) / dp;

            if (std::abs(dF) < 1e-14) {
                break;
            }

            const double p_new = p_star - F / dF;
            const double p_safe = std::max(p_new, p_floor);

            const double denom = std::max(1.0, std::abs(p_star));
            if (std::abs(p_safe - p_star) / denom < 10.0 * newton_tol) {
                p_star = p_safe;
                break;
            }

            p_star = p_safe;
        }

        // [1.2.3] Star velocity
        const double u_star =
            0.5 * (uL_S + uR_S + fR(p_star) - fL(p_star));

        // [1.2.4] Star densities
        double rhoL_star;
        double rhoR_star;

        if (p_star > pL_S) {
            const double gammaL = paramsL_S.gamma;

            const double num =
                (p_star / pL_S) + (gammaL - 1.0) / (gammaL + 1.0);

            const double den =
                ((gammaL - 1.0) / (gammaL + 1.0)) * (p_star / pL_S) + 1.0;

            rhoL_star = rhoL_S * (num / den);
        }
        else {
            rhoL_star = rhoL_S * std::pow(p_star / pL_S, 1.0 / paramsL_S.gamma);
        }

        if (p_star > pR_S) {
            const double gammaR = paramsR_S.gamma;

            const double num =
                (p_star / pR_S) + (gammaR - 1.0) / (gammaR + 1.0);

            const double den =
                ((gammaR - 1.0) / (gammaR + 1.0)) * (p_star / pR_S) + 1.0;

            rhoR_star = rhoR_S * (num / den);
        }
        else {
            rhoR_star = rhoR_S * std::pow(p_star / pR_S, 1.0 / paramsR_S.gamma);
        }

        rhoL_star = std::max(rhoL_star, rho_floor);
        rhoR_star = std::max(rhoR_star, rho_floor);

        return {p_star, u_star, rhoL_star, rhoR_star};
    }

private:
    Primitive<1> PL_S;
    Primitive<1> PR_S;

    EOSParams paramsL_S;
    EOSParams paramsR_S;

    double tol_S;
    int max_iterations_S;

    double rhoL_S;
    double rhoR_S;
    double uL_S;
    double uR_S;
    double pL_S;
    double pR_S;
    double cL_S;
    double cR_S;


    // [1.3] Left wave function
    inline double fL(double p_star) const
    {
        const double gammaL = paramsL_S.gamma;

        if (p_star > pL_S) {
            const double A = 2.0 / ((gammaL + 1.0) * rhoL_S);
            const double B = ((gammaL - 1.0) / (gammaL + 1.0)) * pL_S;

            return (p_star - pL_S) * std::sqrt(A / (p_star + B));
        }

        return (2.0 * cL_S) / (gammaL - 1.0)
            * (std::pow(p_star / pL_S, (gammaL - 1.0) / (2.0 * gammaL)) - 1.0);
    }


    // [1.4] Right wave function
    inline double fR(double p_star) const
    {
        const double gammaR = paramsR_S.gamma;

        if (p_star > pR_S) {
            const double A = 2.0 / ((gammaR + 1.0) * rhoR_S);
            const double B = ((gammaR - 1.0) / (gammaR + 1.0)) * pR_S;

            return (p_star - pR_S) * std::sqrt(A / (p_star + B));
        }

        return (2.0 * cR_S) / (gammaR - 1.0)
            * (std::pow(p_star / pR_S, (gammaR - 1.0) / (2.0 * gammaR)) - 1.0);
    }
};




