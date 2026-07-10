#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/euler/state.hpp"
#include "src/euler/primitives.hpp"
#include "src/euler/thermo_state.hpp"
#include "src/euler/thermo_compute.hpp"
#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/math/numerical_safety.hpp"


// [0] Exact Riemann Star-State Result
struct ExactStarState1D {
    double p_star = 0.0;
    double u_star = 0.0;
};


// [0.1] Exact sampled state
struct ExactSampleState1D {
    Primitive<1> primitive{};
    double e = 0.0;   // specific internal energy
    double E = 0.0;   // total energy density
    double gamma = 0.0;
    int material_id = -1;
};


/* [1] Exact 1D Riemann Solver
    This solver is intended for ideal/stiffened-gas exact validation problems.
    It is templated on EOS for consistency with the rest of the codebase,
    and uses side-specific EOS parameters for mixed-material wave curves.
*/
template<typename EOS>
class ExactRiemannSolver1D {
public:
    // [1.1] Constructor
    ExactRiemannSolver1D(
        const EOSParams& params,
        double tol = 1e-10,
        int max_iter = 200,
        double p_stepping = 1e-6
    )
    :
        paramsL_S(params),
        paramsR_S(params),
        tol_S(tol),
        max_iter_S(max_iter),
        p_stepping_S(p_stepping)
    {}


    ExactRiemannSolver1D(
        const EOSParams& paramsL,
        const EOSParams& paramsR,
        double tol = 1e-10,
        int max_iter = 200,
        double p_stepping = 1e-6
    )
    :
        paramsL_S(paramsL),
        paramsR_S(paramsR),
        tol_S(tol),
        max_iter_S(max_iter),
        p_stepping_S(p_stepping)
    {}


    // [1.2] Solve for p* and u*
    inline ExactStarState1D star_solver(
        const Conserved<1>& UL,
        const Conserved<1>& UR
    ) const
    {
        validate_side_params(paramsL_S, "left");
        validate_side_params(paramsR_S, "right");

        const ThermoState<1> TL = compute_thermo<1, EOS>(UL, paramsL_S);
        const ThermoState<1> TR = compute_thermo<1, EOS>(UR, paramsR_S);

        const double rhoL = require_positive(TL.rho, "exact solver: invalid rhoL", tol_S);
        const double uL = require_finite(TL.vel[0], "exact solver: invalid uL");
        const double pL = require_positive(TL.p, "exact solver: invalid pL", tol_S);

        const double rhoR = require_positive(TR.rho, "exact solver: invalid rhoR", tol_S);
        const double uR = require_finite(TR.vel[0], "exact solver: invalid uR");
        const double pR = require_positive(TR.p, "exact solver: invalid pR", tol_S);

        const double aL = require_positive(TL.c, "exact solver: invalid aL", tol_S);
        const double aR = require_positive(TR.c, "exact solver: invalid aR", tol_S);

        const double pPV = 0.5 * (pL + pR)
            - 0.125 * (uR - uL) * (rhoL + rhoR) * (aL + aR);

        const double p = solve_star_pressure(
            rhoL, uL, pL, aL, paramsL_S,
            rhoR, uR, pR, aR, paramsR_S,
            pPV
        );

        const double fL_final = f_k(rhoL, pL, p, paramsL_S);
        const double fR_final = f_k(rhoR, pR, p, paramsR_S);
        const double F_final = fL_final + fR_final + (uR - uL);

        require_finite(F_final, "exact solver: non-finite final residual");

        const double residual_scale = std::max({
            1.0,
            std::abs(uL),
            std::abs(uR),
            aL,
            aR
        });

        if (std::abs(F_final) >= 100.0 * tol_S * residual_scale) {
            throw std::runtime_error("exact solver: pressure solve residual too large");
        }

        const double u_star = 0.5 * (uL + uR + fR_final - fL_final);

        require_finite(p, "exact solver: non-finite p_star");
        require_finite(u_star, "exact solver: non-finite u_star");

        return {p, u_star};
    }


    // [1.3] Compute rho* from one side
    inline double rho_star(
        double rho0,
        double p0,
        double p_star,
        const EOSParams& params
    ) const
    {
        const double gamma = params.gamma;

        rho0 = require_positive(rho0, "exact solver: invalid rho0 in rho_star", tol_S);
        p0 = require_positive(p0, "exact solver: invalid p0 in rho_star", tol_S);
        p_star = require_positive(p_star, "exact solver: invalid p_star in rho_star", tol_S);

        double rho_star_val = 0.0;

        if (p_star > p0) {
            const double ratio = safe_div(
                shifted_pressure(p_star, params),
                shifted_pressure(p0, params),
                tol_S
            );

            const double numerator =
                ratio + (gamma - 1.0) / (gamma + 1.0);

            const double denominator =
                ((gamma - 1.0) / (gamma + 1.0)) * ratio + 1.0;

            rho_star_val = rho0 * safe_div(numerator, denominator, tol_S);
        }
        else {
            rho_star_val = rho0 * std::pow(
                safe_div(
                    shifted_pressure(p_star, params),
                    shifted_pressure(p0, params),
                    tol_S
                ),
                1.0 / gamma
            );
        }

        return require_positive(
            rho_star_val,
            "exact solver: invalid rho_star",
            tol_S
        );
    }


    // [1.4] Build thermodynamically complete sampled state
    inline ExactSampleState1D sample_state(
        const Conserved<1>& UL,
        const Conserved<1>& UR,
        const ExactStarState1D& star,
        double xi
    ) const
    {
        const ThermoState<1> TL = compute_thermo<1, EOS>(UL, paramsL_S);
        const ThermoState<1> TR = compute_thermo<1, EOS>(UR, paramsR_S);

        const double rhoL = require_positive(TL.rho, "exact sampling: invalid rhoL", tol_S);
        const double uL = require_finite(TL.vel[0], "exact sampling: invalid uL");
        const double pL = require_positive(TL.p, "exact sampling: invalid pL", tol_S);

        const double rhoR = require_positive(TR.rho, "exact sampling: invalid rhoR", tol_S);
        const double uR = require_finite(TR.vel[0], "exact sampling: invalid uR");
        const double pR = require_positive(TR.p, "exact sampling: invalid pR", tol_S);

        const double p_star = require_positive(star.p_star, "exact sampling: invalid p_star", tol_S);
        const double u_star = require_finite(star.u_star, "exact sampling: invalid u_star");

        const double rho_starL = rho_star(rhoL, pL, p_star, paramsL_S);
        const double rho_starR = rho_star(rhoR, pR, p_star, paramsR_S);

        const double aL = require_positive(TL.c, "exact sampling: invalid aL", tol_S);
        const double aR = require_positive(TR.c, "exact sampling: invalid aR", tol_S);

        Primitive<1> S{};
        const EOSParams* sample_params = &paramsL_S;
        int material_id = 0;

        if (xi <= u_star) {
            const double gamma = paramsL_S.gamma;
            if (p_star > pL) {
                const double sL = uL - aL * std::sqrt(
                    shock_speed_factor(p_star, pL, paramsL_S)
                );

                require_finite(sL, "exact sampling: invalid left shock speed");

                if (xi <= sL) {
                    S.rho = rhoL;
                    S.vel[0] = uL;
                    S.p = pL;
                }
                else {
                    S.rho = rho_starL;
                    S.vel[0] = u_star;
                    S.p = p_star;
                }
            }
            else {
                const double shL = uL - aL;

                const double a_star = aL * std::pow(
                    safe_div(
                        shifted_pressure(p_star, paramsL_S),
                        shifted_pressure(pL, paramsL_S),
                        tol_S
                    ),
                    (gamma - 1.0) / (2.0 * gamma)
                );

                const double stL = u_star - a_star;

                require_finite(shL, "exact sampling: invalid left rarefaction head");
                require_finite(stL, "exact sampling: invalid left rarefaction tail");

                if (xi <= shL) {
                    S.rho = rhoL;
                    S.vel[0] = uL;
                    S.p = pL;
                }
                else if (xi <= stL) {
                    const double u = (2.0 / (gamma + 1.0))
                        * (aL + 0.5 * (gamma - 1.0) * uL + xi);

                    const double a = (2.0 / (gamma + 1.0))
                        * (aL + 0.5 * (gamma - 1.0) * (uL - xi));

                    const double a_safe = require_positive(
                        a,
                        "exact sampling: invalid left fan sound speed",
                        tol_S
                    );

                    S.rho = rhoL * std::pow(
                        safe_div(a_safe, aL, tol_S),
                        2.0 / (gamma - 1.0)
                    );
                    S.vel[0] = u;
                    S.p = shifted_pressure(pL, paramsL_S) * std::pow(
                        safe_div(a_safe, aL, tol_S),
                        2.0 * gamma / (gamma - 1.0)
                    ) - paramsL_S.p_inf;
                }
                else {
                    S.rho = rho_starL;
                    S.vel[0] = u_star;
                    S.p = p_star;
                }
            }
        }
        else {
            const double gamma = paramsR_S.gamma;
            sample_params = &paramsR_S;
            material_id = 1;
            if (p_star > pR) {
                const double sR = uR + aR * std::sqrt(
                    shock_speed_factor(p_star, pR, paramsR_S)
                );

                require_finite(sR, "exact sampling: invalid right shock speed");

                if (xi >= sR) {
                    S.rho = rhoR;
                    S.vel[0] = uR;
                    S.p = pR;
                }
                else {
                    S.rho = rho_starR;
                    S.vel[0] = u_star;
                    S.p = p_star;
                }
            }
            else {
                const double shR = uR + aR;

                const double a_star = aR * std::pow(
                    safe_div(
                        shifted_pressure(p_star, paramsR_S),
                        shifted_pressure(pR, paramsR_S),
                        tol_S
                    ),
                    (gamma - 1.0) / (2.0 * gamma)
                );

                const double stR = u_star + a_star;

                require_finite(shR, "exact sampling: invalid right rarefaction head");
                require_finite(stR, "exact sampling: invalid right rarefaction tail");

                if (xi >= shR) {
                    S.rho = rhoR;
                    S.vel[0] = uR;
                    S.p = pR;
                }
                else if (xi >= stR) {
                    const double u = (2.0 / (gamma + 1.0))
                        * (-aR + 0.5 * (gamma - 1.0) * uR + xi);

                    const double a = (2.0 / (gamma + 1.0))
                        * (aR - 0.5 * (gamma - 1.0) * (uR - xi));

                    const double a_safe = require_positive(
                        a,
                        "exact sampling: invalid right fan sound speed",
                        tol_S
                    );

                    S.rho = rhoR * std::pow(
                        safe_div(a_safe, aR, tol_S),
                        2.0 / (gamma - 1.0)
                    );
                    S.vel[0] = u;
                    S.p = shifted_pressure(pR, paramsR_S) * std::pow(
                        safe_div(a_safe, aR, tol_S),
                        2.0 * gamma / (gamma - 1.0)
                    ) - paramsR_S.p_inf;
                }
                else {
                    S.rho = rho_starR;
                    S.vel[0] = u_star;
                    S.p = p_star;
                }
            }
        }

        ExactSampleState1D out{};
        out.primitive = S;

        const double rho = require_positive(S.rho, "exact sampling: invalid sampled rho", tol_S);
        const double u = require_finite(S.vel[0], "exact sampling: invalid sampled velocity");
        const double p = require_positive(S.p, "exact sampling: invalid sampled pressure", tol_S);

        out.e = EOS::internal_energy(rho, p, *sample_params);
        out.e = require_finite(out.e, "exact sampling: invalid specific internal energy");

        out.E = rho * out.e + 0.5 * rho * u * u;
        out.E = require_finite(out.E, "exact sampling: invalid total energy density");
        out.gamma = sample_params->gamma;
        out.material_id = material_id;

        return out;
    }


    // [1.5] Primitive-only sampling using precomputed star state
    inline Primitive<1> sampling(
        const Conserved<1>& UL,
        const Conserved<1>& UR,
        const ExactStarState1D& star,
        double xi
    ) const
    {
        return sample_state(UL, UR, star, xi).primitive;
    }


    // [1.6] Convenience overload: compute star state internally once for one sample
    inline Primitive<1> sampling(
        const Conserved<1>& UL,
        const Conserved<1>& UR,
        double xi
    ) const
    {
        const ExactStarState1D star = star_solver(UL, UR);
        return sampling(UL, UR, star, xi);
    }


    // [1.7] Exact Godunov flux at interface
    inline Conserved<1> flux(
        const Conserved<1>& UL,
        const Conserved<1>& UR,
        double xi = 0.0
    ) const
    {
        const ExactStarState1D star = star_solver(UL, UR);
        const ExactSampleState1D S = sample_state(UL, UR, star, xi);

        const double rho = require_positive(S.primitive.rho, "exact flux: invalid rho", tol_S);
        const double u = require_finite(S.primitive.vel[0], "exact flux: invalid u");
        const double p = require_positive(S.primitive.p, "exact flux: invalid p", tol_S);
        const double E = require_finite(S.E, "exact flux: invalid total energy");

        Conserved<1> F{};
        F.rho = rho * u;
        F.mom[0] = rho * u * u + p;
        F.E = (E + p) * u;

        return F;
    }

private:
    EOSParams paramsL_S;
    EOSParams paramsR_S;
    double tol_S;
    int max_iter_S;
    double p_stepping_S;

    inline void validate_side_params(
        const EOSParams& params,
        const char* side
    ) const
    {
        if (params.gamma <= 1.0 || !std::isfinite(params.gamma)) {
            throw std::runtime_error(std::string("exact solver: invalid ") + side + " gamma");
        }

        if (!eos_has_stiffened_gas_wave_curve(params)) {
            throw std::runtime_error(
                std::string("exact solver: unsupported ") + side +
                " EOS for exact stiffened-gas wave curves"
            );
        }
    }

    inline double shifted_pressure(
        double p,
        const EOSParams& params
    ) const
    {
        return require_positive(
            p + params.p_inf,
            "exact solver: invalid shifted pressure",
            tol_S
        );
    }

    inline double shock_speed_factor(
        double p,
        double p0,
        const EOSParams& params
    ) const
    {
        const double gamma = params.gamma;
        const double ratio = safe_div(
            shifted_pressure(p, params),
            shifted_pressure(p0, params),
            tol_S
        );

        return require_positive(
            ((gamma + 1.0) / (2.0 * gamma)) * ratio +
            ((gamma - 1.0) / (2.0 * gamma)),
            "exact sampling: invalid shock speed factor",
            tol_S
        );
    }

    // [1.8] Single-side wave function
    inline double f_k(
        double rho,
        double p0,
        double p,
        const EOSParams& params
    ) const
    {
        const double gamma = params.gamma;

        rho = require_positive(rho, "exact f_k: invalid rho", tol_S);
        p0 = require_positive(p0, "exact f_k: invalid p0", tol_S);
        p = require_positive(p, "exact f_k: invalid p", tol_S);

        if (p > p0) {
            const double A = safe_div(2.0, (gamma + 1.0) * rho, tol_S);
            const double B =
                ((gamma - 1.0) / (gamma + 1.0)) * p0 +
                ((2.0 * gamma) / (gamma + 1.0)) * params.p_inf;

            const double value = (p - p0) * std::sqrt(
                safe_div(A, p + B, tol_S)
            );

            return require_finite(value, "exact f_k: non-finite shock branch");
        }

        const double cs = std::sqrt(
            require_positive(
                gamma * safe_div(shifted_pressure(p0, params), rho, tol_S),
                "exact f_k: invalid sound speed square",
                tol_S
            )
        );

        const double exponent = (gamma - 1.0) / (2.0 * gamma);

        const double value = (2.0 * cs / (gamma - 1.0))
            * (std::pow(
                safe_div(
                    shifted_pressure(p, params),
                    shifted_pressure(p0, params),
                    tol_S
                ),
                exponent
            ) - 1.0);

        return require_finite(value, "exact f_k: non-finite rarefaction branch");
    }

    inline double df_k(
        double rho,
        double p0,
        double p,
        const EOSParams& params
    ) const
    {
        const double gamma = params.gamma;

        rho = require_positive(rho, "exact df_k: invalid rho", tol_S);
        p0 = require_positive(p0, "exact df_k: invalid p0", tol_S);
        p = require_positive(p, "exact df_k: invalid p", tol_S);

        if (p > p0) {
            const double A = safe_div(2.0, (gamma + 1.0) * rho, tol_S);
            const double B =
                ((gamma - 1.0) / (gamma + 1.0)) * p0 +
                ((2.0 * gamma) / (gamma + 1.0)) * params.p_inf;
            const double denom = safe_denom(p + B, tol_S);
            const double sqrt_term = std::sqrt(safe_div(A, denom, tol_S));

            return require_finite(
                sqrt_term * (1.0 - 0.5 * safe_div(p - p0, denom, tol_S)),
                "exact df_k: non-finite shock branch"
            );
        }

        const double cs = std::sqrt(
            require_positive(
                gamma * safe_div(shifted_pressure(p0, params), rho, tol_S),
                "exact df_k: invalid sound speed square",
                tol_S
            )
        );
        const double ratio = safe_div(
            shifted_pressure(p, params),
            shifted_pressure(p0, params),
            tol_S
        );

        return require_finite(
            safe_div(1.0, rho * cs, tol_S) *
                std::pow(ratio, -(gamma + 1.0) / (2.0 * gamma)),
            "exact df_k: non-finite rarefaction branch"
        );
    }

    inline double pressure_residual(
        double p,
        double rhoL,
        double uL,
        double pL,
        const EOSParams& paramsL,
        double rhoR,
        double uR,
        double pR,
        const EOSParams& paramsR
    ) const
    {
        return require_finite(
            f_k(rhoL, pL, p, paramsL) +
            f_k(rhoR, pR, p, paramsR) +
            (uR - uL),
            "exact solver: non-finite pressure residual"
        );
    }

    inline double solve_star_pressure(
        double rhoL,
        double uL,
        double pL,
        double aL,
        const EOSParams& paramsL,
        double rhoR,
        double uR,
        double pR,
        double aR,
        const EOSParams& paramsR,
        double p_guess
    ) const
    {
        const double p_floor = std::max({
            tol_S,
            -paramsL.p_inf + tol_S,
            -paramsR.p_inf + tol_S
        });

        double lo = p_floor;
        double flo = pressure_residual(lo, rhoL, uL, pL, paramsL, rhoR, uR, pR, paramsR);

        if (flo > 0.0) {
            throw std::runtime_error("exact solver: vacuum state detected");
        }

        double hi = std::max({p_guess, pL, pR, p_floor * 2.0, 1.0});
        double fhi = pressure_residual(hi, rhoL, uL, pL, paramsL, rhoR, uR, pR, paramsR);

        for (int i = 0; i < max_iter_S && fhi <= 0.0; ++i) {
            hi = std::max(2.0 * hi, hi + 1.0);
            fhi = pressure_residual(hi, rhoL, uL, pL, paramsL, rhoR, uR, pR, paramsR);
        }

        if (fhi <= 0.0) {
            throw std::runtime_error("exact solver: failed to bracket p_star");
        }

        double p = std::min(std::max(p_guess, lo), hi);
        const double residual_scale = std::max({
            1.0,
            std::abs(uL),
            std::abs(uR),
            aL,
            aR
        });

        for (int i = 0; i < max_iter_S; ++i) {
            const double f = pressure_residual(p, rhoL, uL, pL, paramsL, rhoR, uR, pR, paramsR);

            if (std::abs(f) <= tol_S * residual_scale) {
                return require_positive(p, "exact solver: invalid p_star", tol_S);
            }

            if (f > 0.0) {
                hi = p;
            }
            else {
                lo = p;
            }

            const double dF =
                df_k(rhoL, pL, p, paramsL) +
                df_k(rhoR, pR, p, paramsR);
            double candidate = p - safe_div(f, dF, tol_S);

            if (!std::isfinite(candidate) || candidate <= lo || candidate >= hi) {
                candidate = 0.5 * (lo + hi);
            }

            p = candidate;

            if (std::abs(hi - lo) <= tol_S * std::max(1.0, p)) {
                return require_positive(0.5 * (lo + hi), "exact solver: invalid p_star", tol_S);
            }
        }

        return require_positive(0.5 * (lo + hi), "exact solver: invalid p_star", tol_S);
    }
};


