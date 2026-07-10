#pragma once

// Sulsky-Chen-Schreyer material-point kinematic tracking for Barton Eulerian fields.

#include "src/solid/elastoplastic/barton/output.hpp"

namespace solid::barton {

template<int DIM>
struct MaterialPoint {
    std::array<double, 3> x0{};
    std::array<double, 3> x{};
    double initial_radius = 0.0;
};

template<int DIM>
inline std::array<double, 3> material_point_velocity(
    const std::vector<TensorState<DIM>>& U,
    const TensorMaterial& mat,
    const TensorSolverConfig& cfg,
    const MaterialPoint<DIM>& p)
{
    const int nx = cfg.cells[0];
    const int ny = cfg.cells[1];
    const int nz = DIM == 3 ? cfg.cells[2] : 1;
    const std::array<double, 3> h{
        (cfg.domain_max[0] - cfg.domain_min[0]) / nx,
        (cfg.domain_max[1] - cfg.domain_min[1]) / ny,
        DIM == 3 ? (cfg.domain_max[2] - cfg.domain_min[2]) / nz : 1.0
    };

    std::array<double, 3> xi{};
    std::array<int, 3> lo{};
    std::array<double, 3> w{};
    for (int d = 0; d < DIM; ++d) {
        xi[d] = (p.x[d] - cfg.domain_min[d]) / h[d] - 0.5;
        lo[d] = static_cast<int>(std::floor(xi[d]));
        w[d] = xi[d] - lo[d];
        lo[d] = std::clamp(lo[d], 0, cfg.cells[d] - 1);
        w[d] = std::clamp(w[d], 0.0, 1.0);
        if (lo[d] == cfg.cells[d] - 1) {
            w[d] = 0.0;
        }
    }

    auto cell_index = [&](int i, int j, int k) {
        i = std::clamp(i, 0, nx - 1);
        j = std::clamp(j, 0, ny - 1);
        k = std::clamp(k, 0, nz - 1);
        if constexpr (DIM == 2) {
            (void)k;
            return hidx(i, j, nx);
        }
        else {
            return hidx3(i, j, k, nx, ny);
        }
    };

    std::array<double, 3> velocity{};
    const int kz_count = DIM == 3 ? 2 : 1;
    for (int kk = 0; kk < kz_count; ++kk) {
        const double wz = DIM == 3 ? (kk == 0 ? 1.0 - w[2] : w[2]) : 1.0;
        for (int jj = 0; jj < 2; ++jj) {
            const double wy = jj == 0 ? 1.0 - w[1] : w[1];
            for (int ii = 0; ii < 2; ++ii) {
                const double wx = ii == 0 ? 1.0 - w[0] : w[0];
                const double weight = wx * wy * wz;
                const TensorPrim<DIM> P = tensor_prim(U[cell_index(lo[0] + ii, lo[1] + jj, lo[2] + kk)], mat);
                for (int d = 0; d < DIM; ++d) {
                    velocity[d] += weight * P.vel[d];
                }
            }
        }
    }
    return velocity;
}

template<int DIM>
inline std::vector<MaterialPoint<DIM>> seed_material_points(const TensorSolverConfig& cfg)
{
    const int stride = std::max(cfg.material_point_stride, 1);
    const int nx = cfg.cells[0];
    const int ny = cfg.cells[1];
    const int nz = DIM == 3 ? cfg.cells[2] : 1;
    const std::array<double, 3> h{
        (cfg.domain_max[0] - cfg.domain_min[0]) / nx,
        (cfg.domain_max[1] - cfg.domain_min[1]) / ny,
        DIM == 3 ? (cfg.domain_max[2] - cfg.domain_min[2]) / nz : 1.0
    };

    std::vector<MaterialPoint<DIM>> points;
    points.reserve((nx / stride + 1) * (ny / stride + 1) * (nz / stride + 1));
    for (int k = 0; k < nz; k += stride) {
        for (int j = 0; j < ny; j += stride) {
            for (int i = 0; i < nx; i += stride) {
                MaterialPoint<DIM> p{};
                p.x[0] = cfg.domain_min[0] + (i + 0.5) * h[0];
                p.x[1] = cfg.domain_min[1] + (j + 0.5) * h[1];
                p.x[2] = DIM == 3 ? cfg.domain_min[2] + (k + 0.5) * h[2] : 0.0;
                p.x0 = p.x;
                p.initial_radius = std::sqrt(p.x0[0] * p.x0[0] + p.x0[1] * p.x0[1] + p.x0[2] * p.x0[2]);
                points.push_back(p);
            }
        }
    }
    return points;
}

template<int DIM>
inline void advect_material_points(
    std::vector<MaterialPoint<DIM>>& points,
    const std::vector<TensorState<DIM>>& U,
    const TensorMaterial& mat,
    const TensorSolverConfig& cfg,
    double dt)
{
    const int count = static_cast<int>(points.size());
#pragma omp parallel for schedule(static) if(count > 4096)
    for (int n = 0; n < count; ++n) {
        const auto velocity = material_point_velocity(U, mat, cfg, points[n]);
        for (int d = 0; d < DIM; ++d) {
            points[n].x[d] += dt * velocity[d];
            points[n].x[d] = std::clamp(points[n].x[d], cfg.domain_min[d], cfg.domain_max[d]);
        }
    }
}

template<int DIM>
inline void write_material_points_vtp(
    const std::string& filename,
    const std::vector<MaterialPoint<DIM>>& points)
{
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Cannot write Barton material-point VTP: " + filename);
    }
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <PolyData>\n";
    out << "    <Piece NumberOfPoints=\"" << points.size() << "\" NumberOfVerts=\"" << points.size() << "\">\n";
    out << "      <PointData Scalars=\"initial_radius\">\n";
    out << "        <DataArray type=\"Float64\" Name=\"initial_radius\" format=\"ascii\">\n";
    for (const auto& p : points) out << p.initial_radius << " ";
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Float64\" Name=\"displacement_magnitude\" format=\"ascii\">\n";
    for (const auto& p : points) {
        const double dx = p.x[0] - p.x0[0];
        const double dy = p.x[1] - p.x0[1];
        const double dz = p.x[2] - p.x0[2];
        out << std::sqrt(dx * dx + dy * dy + dz * dz) << " ";
    }
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Float64\" Name=\"radial_displacement\" format=\"ascii\">\n";
    for (const auto& p : points) {
        const double r = std::sqrt(p.x[0] * p.x[0] + p.x[1] * p.x[1] + p.x[2] * p.x[2]);
        out << r - p.initial_radius << " ";
    }
    out << "\n        </DataArray>\n";
    out << "      </PointData>\n";
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (const auto& p : points) {
        out << p.x[0] << " " << p.x[1] << " " << p.x[2] << " ";
    }
    out << "\n        </DataArray>\n";
    out << "      </Points>\n";
    out << "      <Verts>\n";
    out << "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n";
    for (std::size_t i = 0; i < points.size(); ++i) out << i << " ";
    out << "\n        </DataArray>\n";
    out << "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n";
    for (std::size_t i = 0; i < points.size(); ++i) out << i + 1 << " ";
    out << "\n        </DataArray>\n";
    out << "      </Verts>\n";
    out << "    </Piece>\n";
    out << "  </PolyData>\n";
    out << "</VTKFile>\n";
}

inline std::string material_point_vtp_path(const std::string& base, const std::string& label)
{
    return base + "_material_points_" + label + ".vtp";
}

} // namespace solid::barton
