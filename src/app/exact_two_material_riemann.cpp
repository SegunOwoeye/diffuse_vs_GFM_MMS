#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <functional>
#include <vector>

#include "src/euler/eos.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/primitives.hpp"
#include "src/euler/riemann/exact_riemann.hpp"

/*
    Small command-line driver for generating one-dimensional exact references
    with different left/right material parameters.

    It is used by tools/generate_fedkiw_exact_references.py so the quantitative
    runner can compare Fedkiw-style multimaterial shock tubes against a solver
    reference that respects side-specific gamma and p_inf values.
*/
namespace {

struct Args {
    double rho_L = 0.0;
    double u_L = 0.0;
    double p_L = 0.0;
    double gamma_L = 0.0;
    double rho_R = 0.0;
    double u_R = 0.0;
    double p_R = 0.0;
    double gamma_R = 0.0;
    double x0 = 0.0;
    double t = 0.0;
    double x_min = 0.0;
    double x_max = 0.0;
    int N = 0;
    std::string output;
    double p_inf_L = 0.0;
    double p_inf_R = 0.0;
};

struct MieGruneisenArgs {
    double rho_L = 0.0;
    double u_L = 0.0;
    double p_L = 0.0;
    double rho_R = 0.0;
    double u_R = 0.0;
    double p_R = 0.0;
    double gamma = 0.0;
    double rho_ref = 0.0;
    double A1 = 0.0;
    double A2 = 0.0;
    double E1 = 0.0;
    double E2 = 0.0;
    double x0 = 0.0;
    double t = 0.0;
    double x_min = 0.0;
    double x_max = 0.0;
    int N = 0;
    std::string output;
};

struct MieGruneisenState {
    double rho = 0.0;
    double u = 0.0;
    double p = 0.0;
};

struct MieGruneisenStar {
    double p = 0.0;
    double u = 0.0;
    double rho_L = 0.0;
    double rho_R = 0.0;
};

double parse_double(const char* text, const char* name)
{
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value)) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return value;
}

int parse_int(const char* text, const char* name)
{
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value <= 0) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
    return static_cast<int>(value);
}

bool is_mie_gruneisen_mode(int argc, char** argv)
{
    return argc >= 2 && std::string(argv[1]) == "--mie-gruneisen";
}

Args parse_args(int argc, char** argv)
{
    if (argc != 14 && argc != 15 && argc != 17) {
        throw std::runtime_error(
            "Usage: exact_two_material_riemann "
            "rho_L u_L p_L gamma_L rho_R u_R p_R gamma_R "
            "x0 t x_min x_max N [output.csv] [p_inf_L p_inf_R]"
        );
    }

    Args args{};
    args.rho_L = parse_double(argv[1], "rho_L");
    args.u_L = parse_double(argv[2], "u_L");
    args.p_L = parse_double(argv[3], "p_L");
    args.gamma_L = parse_double(argv[4], "gamma_L");
    args.rho_R = parse_double(argv[5], "rho_R");
    args.u_R = parse_double(argv[6], "u_R");
    args.p_R = parse_double(argv[7], "p_R");
    args.gamma_R = parse_double(argv[8], "gamma_R");
    args.x0 = parse_double(argv[9], "x0");
    args.t = parse_double(argv[10], "t");
    args.x_min = parse_double(argv[11], "x_min");
    args.x_max = parse_double(argv[12], "x_max");
    args.N = parse_int(argv[13], "N");

    if (argc >= 15) {
        args.output = argv[14];
    }
    if (argc == 17) {
        args.p_inf_L = parse_double(argv[15], "p_inf_L");
        args.p_inf_R = parse_double(argv[16], "p_inf_R");
    }

    if (args.rho_L <= 0.0 || args.rho_R <= 0.0) {
        throw std::runtime_error("densities must be positive");
    }
    if (args.p_L + args.p_inf_L <= 0.0 || args.p_R + args.p_inf_R <= 0.0) {
        throw std::runtime_error("shifted pressures p + p_inf must be positive");
    }
    if (args.gamma_L <= 1.0 || args.gamma_R <= 1.0) {
        throw std::runtime_error("gammas must be greater than 1");
    }
    if (args.t < 0.0) {
        throw std::runtime_error("t must be non-negative");
    }
    if (args.x_max <= args.x_min) {
        throw std::runtime_error("x_max must be greater than x_min");
    }

    return args;
}

MieGruneisenArgs parse_mie_gruneisen_args(int argc, char** argv)
{
    if (argc != 20) {
        throw std::runtime_error(
            "Usage: exact_two_material_riemann --mie-gruneisen "
            "rho_L u_L p_L rho_R u_R p_R gamma rho_ref A1 A2 E1 E2 "
            "x0 t x_min x_max N output.csv"
        );
    }

    MieGruneisenArgs args{};
    args.rho_L = parse_double(argv[2], "rho_L");
    args.u_L = parse_double(argv[3], "u_L");
    args.p_L = parse_double(argv[4], "p_L");
    args.rho_R = parse_double(argv[5], "rho_R");
    args.u_R = parse_double(argv[6], "u_R");
    args.p_R = parse_double(argv[7], "p_R");
    args.gamma = parse_double(argv[8], "gamma");
    args.rho_ref = parse_double(argv[9], "rho_ref");
    args.A1 = parse_double(argv[10], "A1");
    args.A2 = parse_double(argv[11], "A2");
    args.E1 = parse_double(argv[12], "E1");
    args.E2 = parse_double(argv[13], "E2");
    args.x0 = parse_double(argv[14], "x0");
    args.t = parse_double(argv[15], "t");
    args.x_min = parse_double(argv[16], "x_min");
    args.x_max = parse_double(argv[17], "x_max");
    args.N = parse_int(argv[18], "N");
    args.output = argv[19];

    if (args.rho_L <= 0.0 || args.rho_R <= 0.0 || args.rho_ref <= 0.0) {
        throw std::runtime_error("Mie-Gruneisen densities must be positive");
    }
    if (args.gamma <= 1.0) {
        throw std::runtime_error("Mie-Gruneisen gamma must be greater than 1");
    }
    if (args.E1 == 1.0 || args.E2 == 1.0) {
        throw std::runtime_error("Mie-Gruneisen exponents must not equal 1");
    }
    if (args.t < 0.0) {
        throw std::runtime_error("t must be non-negative");
    }
    if (args.x_max <= args.x_min) {
        throw std::runtime_error("x_max must be greater than x_min");
    }

    return args;
}

Primitive<1> make_primitive(double rho, double u, double p)
{
    Primitive<1> primitive{};
    primitive.rho = rho;
    primitive.vel[0] = u;
    primitive.p = p;
    return primitive;
}

EOSParams make_stiffened_params(double gamma, double p_inf)
{
    EOSParams params{};
    params.kind = EOSKind::stiffened_gas;
    params.gamma = gamma;
    params.p_inf = p_inf;
    return params;
}

void write_solution(std::ostream& out, const Args& args)
{
    /*
        The exact solver samples the self-similar solution at cell centres so the
        generated CSVs align directly with finite-volume output conventions.
    */
    const EOSParams paramsL = make_stiffened_params(args.gamma_L, args.p_inf_L);
    const EOSParams paramsR = make_stiffened_params(args.gamma_R, args.p_inf_R);
    const Primitive<1> PL = make_primitive(args.rho_L, args.u_L, args.p_L);
    const Primitive<1> PR = make_primitive(args.rho_R, args.u_R, args.p_R);

    const Conserved<1> UL = prim_to_cons<1, MaterialEOS>(PL, paramsL);
    const Conserved<1> UR = prim_to_cons<1, MaterialEOS>(PR, paramsR);

    const ExactRiemannSolver1D<MaterialEOS> solver(paramsL, paramsR);
    const ExactStarState1D star = (args.t > 0.0)
        ? solver.star_solver(UL, UR)
        : ExactStarState1D{};

    out << std::setprecision(17);
    out << "x,rho,u,p,e,gamma,material_id\n";

    const double dx = (args.x_max - args.x_min) / static_cast<double>(args.N);
    for (int i = 0; i < args.N; ++i) {
        const double x = args.x_min + (static_cast<double>(i) + 0.5) * dx;

        double rho = 0.0;
        double u = 0.0;
        double p = 0.0;
        double e = 0.0;
        double gamma = 0.0;
        int material_id = 0;

        if (args.t <= 0.0) {
            const bool left = x < args.x0;
            const EOSParams& params = left ? paramsL : paramsR;
            rho = left ? args.rho_L : args.rho_R;
            u = left ? args.u_L : args.u_R;
            p = left ? args.p_L : args.p_R;
            e = MaterialEOS::internal_energy(rho, p, params);
            gamma = params.gamma;
            material_id = left ? 0 : 1;
        }
        else {
            const double xi = (x - args.x0) / args.t;
            const ExactSampleState1D sample = solver.sample_state(UL, UR, star, xi);
            rho = sample.primitive.rho;
            u = sample.primitive.vel[0];
            p = sample.primitive.p;
            e = sample.e;
            gamma = sample.gamma;
            material_id = sample.material_id;
        }

        out << x << ","
            << rho << ","
            << u << ","
            << p << ","
            << e << ","
            << gamma << ","
            << material_id << "\n";
    }
}

double mie_power_ratio(double rho, const MieGruneisenArgs& args)
{
    return std::max(rho / args.rho_ref, 1.0e-14);
}

double mie_cold_energy(double rho, const MieGruneisenArgs& args)
{
    const double r = mie_power_ratio(rho, args);
    const double term1 = args.A1 / (args.rho_ref * (args.E1 - 1.0))
        * (std::pow(r, args.E1 - 1.0) - 1.0);
    const double term2 = args.A2 / (args.rho_ref * (args.E2 - 1.0))
        * (std::pow(r, args.E2 - 1.0) - 1.0);
    return term1 - term2;
}

double mie_cold_pressure(double rho, const MieGruneisenArgs& args)
{
    const double r = mie_power_ratio(rho, args);
    return args.A1 * std::pow(r, args.E1) - args.A2 * std::pow(r, args.E2);
}

double mie_cold_pressure_derivative(double rho, const MieGruneisenArgs& args)
{
    const double r = mie_power_ratio(rho, args);
    return (
        args.A1 * args.E1 * std::pow(r, args.E1 - 1.0)
        - args.A2 * args.E2 * std::pow(r, args.E2 - 1.0)
    ) / args.rho_ref;
}

double mie_internal_energy(double rho, double p, const MieGruneisenArgs& args)
{
    return (p - mie_cold_pressure(rho, args)) / ((args.gamma - 1.0) * rho)
        + mie_cold_energy(rho, args);
}

double mie_entropy_invariant(const MieGruneisenState& state, const MieGruneisenArgs& args)
{
    const double thermal_pressure = std::max(
        state.p - mie_cold_pressure(state.rho, args),
        1.0e-12
    );
    return thermal_pressure / std::pow(state.rho, args.gamma);
}

double mie_pressure_on_isentrope(double rho, double invariant, const MieGruneisenArgs& args)
{
    return invariant * std::pow(rho, args.gamma) + mie_cold_pressure(rho, args);
}

double mie_sound_speed_on_isentrope(double rho, double invariant, const MieGruneisenArgs& args)
{
    const double c2 = args.gamma * invariant * std::pow(rho, args.gamma - 1.0)
        + mie_cold_pressure_derivative(rho, args);
    return std::sqrt(std::max(c2, 1.0e-12));
}

double bisect_root(
    const std::function<double(double)>& fn,
    double lo,
    double hi,
    int iterations = 100
)
{
    double flo = fn(lo);
    double fhi = fn(hi);
    if (flo == 0.0) {
        return lo;
    }
    if (fhi == 0.0) {
        return hi;
    }
    if (flo * fhi > 0.0) {
        throw std::runtime_error("root is not bracketed");
    }

    for (int i = 0; i < iterations; ++i) {
        const double mid = 0.5 * (lo + hi);
        const double fmid = fn(mid);
        if (flo * fmid <= 0.0) {
            hi = mid;
            fhi = fmid;
        }
        else {
            lo = mid;
            flo = fmid;
        }
    }

    return 0.5 * (lo + hi);
}

double integrate_simpson(
    const std::function<double(double)>& fn,
    double a,
    double b,
    int intervals = 512
)
{
    if (a == b) {
        return 0.0;
    }
    if (intervals % 2 != 0) {
        ++intervals;
    }
    const double h = (b - a) / static_cast<double>(intervals);
    double total = fn(a) + fn(b);
    for (int i = 1; i < intervals; ++i) {
        total += (i % 2 == 0 ? 2.0 : 4.0) * fn(a + static_cast<double>(i) * h);
    }
    return total * h / 3.0;
}

double mie_density_from_isentrope(
    double p,
    const MieGruneisenState& state,
    const MieGruneisenArgs& args
)
{
    const double invariant = mie_entropy_invariant(state, args);
    double lo = 1.0e-6 * args.rho_ref;
    double hi = 100.0 * args.rho_ref;

    const auto residual = [&](double rho) {
        return mie_pressure_on_isentrope(rho, invariant, args) - p;
    };

    while (residual(lo) > 0.0) {
        lo *= 0.5;
    }
    while (residual(hi) < 0.0) {
        hi *= 2.0;
    }

    return bisect_root(residual, lo, hi);
}

double mie_rarefaction_function(
    double p,
    const MieGruneisenState& state,
    const MieGruneisenArgs& args
)
{
    const double invariant = mie_entropy_invariant(state, args);
    const double rho_star = mie_density_from_isentrope(p, state, args);
    return integrate_simpson(
        [&](double rho) {
            return mie_sound_speed_on_isentrope(rho, invariant, args) / rho;
        },
        state.rho,
        rho_star
    );
}

double mie_shock_density(
    double p,
    const MieGruneisenState& state,
    const MieGruneisenArgs& args
)
{
    const double e0 = mie_internal_energy(state.rho, state.p, args);
    const auto residual = [&](double rho) {
        const double e = mie_internal_energy(rho, p, args);
        return e - e0 + 0.5 * (p + state.p) * (1.0 / rho - 1.0 / state.rho);
    };

    const double lo = state.rho * (1.0 + 1.0e-12);
    double hi = state.rho * 2.0;
    while (residual(hi) > 0.0) {
        hi *= 2.0;
    }
    return bisect_root(residual, lo, hi);
}

double mie_wave_function(
    double p,
    const MieGruneisenState& state,
    const MieGruneisenArgs& args
)
{
    if (p <= state.p) {
        return mie_rarefaction_function(p, state, args);
    }
    const double rho_star = mie_shock_density(p, state, args);
    return std::sqrt(std::max((p - state.p) * (1.0 / state.rho - 1.0 / rho_star), 0.0));
}

double mie_star_density(
    double p,
    const MieGruneisenState& state,
    const MieGruneisenArgs& args
)
{
    if (p <= state.p) {
        return mie_density_from_isentrope(p, state, args);
    }
    return mie_shock_density(p, state, args);
}

MieGruneisenStar solve_mie_gruneisen_star(
    const MieGruneisenState& left,
    const MieGruneisenState& right,
    const MieGruneisenArgs& args
)
{
    const auto residual = [&](double p) {
        return mie_wave_function(p, left, args)
            + mie_wave_function(p, right, args)
            + right.u
            - left.u;
    };

    const double lo = 1.0e-10;
    double hi = std::max({left.p, right.p, 1.0});
    while (residual(hi) < 0.0) {
        hi *= 2.0;
    }

    const double p_star = bisect_root(residual, lo, hi, 120);
    const double f_L = mie_wave_function(p_star, left, args);
    const double f_R = mie_wave_function(p_star, right, args);
    const double u_star = 0.5 * (left.u + right.u + f_R - f_L);
    return {
        p_star,
        u_star,
        mie_star_density(p_star, left, args),
        mie_star_density(p_star, right, args)
    };
}

MieGruneisenState sample_mie_left_rarefaction(
    double xi,
    const MieGruneisenState& left,
    const MieGruneisenStar& star,
    const MieGruneisenArgs& args
)
{
    const double invariant = mie_entropy_invariant(left, args);
    const auto velocity_at = [&](double rho) {
        const double integral = integrate_simpson(
            [&](double rr) {
                return mie_sound_speed_on_isentrope(rr, invariant, args) / rr;
            },
            left.rho,
            rho
        );
        return left.u - integral;
    };

    const auto residual = [&](double rho) {
        return velocity_at(rho) - mie_sound_speed_on_isentrope(rho, invariant, args) - xi;
    };

    const double rho = bisect_root(residual, star.rho_L, left.rho);
    return {rho, velocity_at(rho), mie_pressure_on_isentrope(rho, invariant, args)};
}

MieGruneisenState sample_mie_gruneisen_solution(
    double x,
    const MieGruneisenState& left,
    const MieGruneisenState& right,
    const MieGruneisenStar& star,
    const MieGruneisenArgs& args
)
{
    if (args.t <= 0.0) {
        return x < args.x0 ? left : right;
    }

    const double xi = (x - args.x0) / args.t;
    const double invariant_L = mie_entropy_invariant(left, args);
    const double c_L = mie_sound_speed_on_isentrope(left.rho, invariant_L, args);
    const double c_L_star = mie_sound_speed_on_isentrope(star.rho_L, invariant_L, args);

    if (xi < left.u - c_L) {
        return left;
    }
    if (xi < star.u - c_L_star) {
        return sample_mie_left_rarefaction(xi, left, star, args);
    }
    if (xi < star.u) {
        return {star.rho_L, star.u, star.p};
    }

    if (star.p > right.p) {
        const double mass_flux = std::sqrt(
            (star.p - right.p) / (1.0 / right.rho - 1.0 / star.rho_R)
        );
        const double shock_speed = right.u + mass_flux / right.rho;
        if (xi < shock_speed) {
            return {star.rho_R, star.u, star.p};
        }
        return right;
    }

    return xi < star.u ? MieGruneisenState{star.rho_R, star.u, star.p} : right;
}

void write_mie_gruneisen_solution(std::ostream& out, const MieGruneisenArgs& args)
{
    const MieGruneisenState left{args.rho_L, args.u_L, args.p_L};
    const MieGruneisenState right{args.rho_R, args.u_R, args.p_R};
    const MieGruneisenStar star = solve_mie_gruneisen_star(left, right, args);

    std::cerr << std::setprecision(8)
              << "Mie-Gruneisen star: p=" << star.p
              << ", u=" << star.u
              << ", rho_L=" << star.rho_L
              << ", rho_R=" << star.rho_R << "\n";

    out << std::setprecision(17);
    out << "x,rho,u,p,e,gamma,material_id,time,eos\n";

    const double dx = (args.x_max - args.x_min) / static_cast<double>(args.N);
    for (int i = 0; i < args.N; ++i) {
        const double x = args.x_min + (static_cast<double>(i) + 0.5) * dx;
        const MieGruneisenState sample = sample_mie_gruneisen_solution(
            x,
            left,
            right,
            star,
            args
        );
        const double e = mie_internal_energy(sample.rho, sample.p, args);
        const double xi = args.t > 0.0 ? (x - args.x0) / args.t : 0.0;
        const int material_id = args.t > 0.0
            ? (xi <= star.u ? 0 : 1)
            : (x < args.x0 ? 0 : 1);

        out << x << ","
            << sample.rho << ","
            << sample.u << ","
            << sample.p << ","
            << e << ","
            << args.gamma << ","
            << material_id << ","
            << args.t << ","
            << "mie_gruneisen\n";
    }
}

void write_mie_gruneisen_solution_to_path(const MieGruneisenArgs& args)
{
    if (args.output.empty() || args.output == "-") {
        write_mie_gruneisen_solution(std::cout, args);
        return;
    }

    const std::filesystem::path output_path(args.output);
    if (output_path.has_parent_path()) {
        std::filesystem::create_directories(output_path.parent_path());
    }

    std::ofstream file(args.output);
    if (!file) {
        throw std::runtime_error("failed to open output file: " + args.output);
    }
    write_mie_gruneisen_solution(file, args);
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (is_mie_gruneisen_mode(argc, argv)) {
            write_mie_gruneisen_solution_to_path(parse_mie_gruneisen_args(argc, argv));
            return 0;
        }

        const Args args = parse_args(argc, argv);

        if (args.output.empty() || args.output == "-") {
            write_solution(std::cout, args);
            return 0;
        }

        const std::filesystem::path output_path(args.output);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }

        std::ofstream file(args.output);
        if (!file) {
            throw std::runtime_error("failed to open output file: " + args.output);
        }

        write_solution(file, args);
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
