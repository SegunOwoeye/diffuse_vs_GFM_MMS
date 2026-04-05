#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "src/euler/primitives.hpp"
#include "src/euler/eos_params.hpp"
#include "src/math/numerical_safety.hpp"

// Completetly add in numerical saftey

// [0] Mixed Riemann Solver Result
struct MCRSResult1D {
    double p_star;
    double u_star;
    double rhoL_star;
    double rhoR_star;
};


// [1] Mixed-Component Riemann Solver (normal direction)
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
    ):
        PL_S(PL),
        PR_S(PR),
        paramsL_S(paramsL),
        paramsR_S(paramsR),
        tol_S(clamp_min(tol, 1e-14)),
        max_iterations_S(std::max(max_iterations, 1))
    {
        // enforce valid inputs
        rhoL_S = require_positive(PL_S.rho, "MCRS: rhoL invalid", tol_S);
        rhoR_S = require_positive(PR_S.rho, "MCRS: rhoR invalid", tol_S);

        uL_S = require_finite(PL_S.vel[0], "MCRS: uL invalid");
        uR_S = require_finite(PR_S.vel[0], "MCRS: uR invalid");

        pL_S = require_positive(PL_S.p, "MCRS: pL invalid", tol_S);
        pR_S = require_positive(PR_S.p, "MCRS: pR invalid", tol_S);

        // sound speeds (ideal gas assumption)
        cL_S = compute_c(rhoL_S, pL_S, paramsL_S);
        cR_S = compute_c(rhoR_S, pR_S, paramsR_S);
    }

    // [1.2] Solve mixed-material Riemann problem
    inline MCRSResult1D solve() const
    {
        check_vacuum();

        const double p_floor = tol_S;
        const double rho_floor = tol_S;

        // [1.2.1] PVRS initial guess
        const double rho_bar = 0.5 * (rhoL_S + rhoR_S);
        const double a_bar = 0.5 * (cL_S + cR_S);

        double p_star = clamp_min(
            0.5 * (pL_S + pR_S)
            - 0.5 * (uR_S - uL_S) * rho_bar * a_bar,
            p_floor
        );

        // [1.2.2] Newton solve
        bool converged = false;

        for (int n = 0; n < max_iterations_S; ++n) {

            const double FL  = fL(p_star);
            const double FR  = fR(p_star);
            const double dFL = dfL(p_star);
            const double dFR = dfR(p_star);

            const double F  = FL + FR + (uR_S - uL_S);
            const double dF = dFL + dFR;

            if (std::abs(F) < tol_S) {
                converged = true;
                break;
            }

            const double step = safe_div(F, dF, tol_S);
            const double p_new = p_star - step;

            const double p_safe = clamp_min(p_new, p_floor);

            const double rel = std::abs(p_safe - p_star) / std::max(1.0, std::abs(p_star));

            p_star = p_safe;

            if (rel < 10.0 * tol_S) {
                converged = true;
                break;
            }
        }

        if (!converged) {
            p_star = clamp_min(0.5 * (pL_S + pR_S), p_floor);
        }

        p_star = require_positive(p_star, "MCRS: p_star invalid", p_floor);

        // [1.2.3] velocity
        const double u_star = 0.5 * (uL_S + uR_S + fR(p_star) - fL(p_star));

        // [1.2.4] densities
        double rhoL_star = rho_star_L(p_star);
        double rhoR_star = rho_star_R(p_star);

        rhoL_star = require_positive(rhoL_star, "MCRS: rhoL* invalid", rho_floor);
        rhoR_star = require_positive(rhoR_star, "MCRS: rhoR* invalid", rho_floor);

        return {
            p_star,
            require_finite(u_star, "MCRS: u_star invalid"),
            rhoL_star,
            rhoR_star
        };
    }

private:
    Primitive<1> PL_S, PR_S;
    EOSParams paramsL_S, paramsR_S;

    double tol_S;
    int max_iterations_S;

    double rhoL_S, rhoR_S;
    double uL_S, uR_S;
    double pL_S, pR_S;
    double cL_S, cR_S;


    // sound speed 
    inline double compute_c(double rho, double p, const EOSParams& params) const
    {
        const double val = params.gamma * safe_div(p, rho, tol_S);
        return require_positive(std::sqrt(val), "MCRS: sound speed invalid", tol_S);
    }


    // vacuum check 
    inline void check_vacuum() const
    {
        const double crit =
            (2.0 * cL_S / (paramsL_S.gamma - 1.0))
          + (2.0 * cR_S / (paramsR_S.gamma - 1.0));

        if ((uR_S - uL_S) >= crit) {
            throw std::runtime_error("MCRS: vacuum state detected");
        }
    }


    // wave functions 
    inline double fL(double p_star) const
    {
        const double p = clamp_min(p_star, tol_S);
        const double g = paramsL_S.gamma;

        if (p > pL_S) {
            const double A = 2.0 / ((g + 1.0) * rhoL_S);
            const double B = ((g - 1.0) / (g + 1.0)) * pL_S;

            const double denom = clamp_min(p + B, tol_S);
            return (p - pL_S) * std::sqrt(safe_div(A, denom, tol_S));
        }

        const double ratio = safe_div(p, pL_S, tol_S);

        return (2.0 * cL_S) / (g - 1.0)
            * (std::pow(ratio, (g - 1.0) / (2.0 * g)) - 1.0);
    }


    inline double fR(double p_star) const
    {
        const double p = clamp_min(p_star, tol_S);
        const double g = paramsR_S.gamma;

        if (p > pR_S) {
            const double A = 2.0 / ((g + 1.0) * rhoR_S);
            const double B = ((g - 1.0) / (g + 1.0)) * pR_S;

            const double denom = clamp_min(p + B, tol_S);
            return (p - pR_S) * std::sqrt(safe_div(A, denom, tol_S));
        }

        const double ratio = safe_div(p, pR_S, tol_S);

        return (2.0 * cR_S) / (g - 1.0)
            * (std::pow(ratio, (g - 1.0) / (2.0 * g)) - 1.0);
    }


    // derivatives 
    inline double dfL(double p_star) const
    {
        const double p = clamp_min(p_star, tol_S);
        const double g = paramsL_S.gamma;

        if (p > pL_S) {
            const double A = 2.0 / ((g + 1.0) * rhoL_S);
            const double B = ((g - 1.0) / (g + 1.0)) * pL_S;

            const double denom = clamp_min(p + B, tol_S);
            const double sqrt_term = std::sqrt(safe_div(A, denom, tol_S));

            return sqrt_term * (1.0 - 0.5 * safe_div(p - pL_S, denom, tol_S));
        }

        const double ratio = safe_div(p, pL_S, tol_S);

        return safe_div(1.0, rhoL_S * cL_S, tol_S)
            * std::pow(ratio, -(g + 1.0) / (2.0 * g));
    }


    inline double dfR(double p_star) const
    {
        const double p = clamp_min(p_star, tol_S);
        const double g = paramsR_S.gamma;

        if (p > pR_S) {
            const double A = 2.0 / ((g + 1.0) * rhoR_S);
            const double B = ((g - 1.0) / (g + 1.0)) * pR_S;

            const double denom = clamp_min(p + B, tol_S);
            const double sqrt_term = std::sqrt(safe_div(A, denom, tol_S));

            return sqrt_term * (1.0 - 0.5 * safe_div(p - pR_S, denom, tol_S));
        }

        const double ratio = safe_div(p, pR_S, tol_S);

        return safe_div(1.0, rhoR_S * cR_S, tol_S)
            * std::pow(ratio, -(g + 1.0) / (2.0 * g));
    }


    // densities 
    inline double rho_star_L(double p_star) const
    {
        const double p = clamp_min(p_star, tol_S);
        const double g = paramsL_S.gamma;

        if (p > pL_S) {
            const double r = safe_div(p, pL_S, tol_S);

            const double num = r + (g - 1.0) / (g + 1.0);
            const double den = ((g - 1.0) / (g + 1.0)) * r + 1.0;

            return rhoL_S * safe_div(num, den, tol_S);
        }

        return rhoL_S * std::pow(safe_div(p, pL_S, tol_S), 1.0 / g);
    }


    inline double rho_star_R(double p_star) const
    {
        const double p = clamp_min(p_star, tol_S);
        const double g = paramsR_S.gamma;

        if (p > pR_S) {
            const double r = safe_div(p, pR_S, tol_S);

            const double num = r + (g - 1.0) / (g + 1.0);
            const double den = ((g - 1.0) / (g + 1.0)) * r + 1.0;

            return rhoR_S * safe_div(num, den, tol_S);
        }

        return rhoR_S * std::pow(safe_div(p, pR_S, tol_S), 1.0 / g);
    }
};


