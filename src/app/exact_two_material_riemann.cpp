#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
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

} // namespace

int main(int argc, char** argv)
{
    try {
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
