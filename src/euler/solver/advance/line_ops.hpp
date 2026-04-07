#pragma once

#include <vector>
#include <array>
#include <stdexcept>

#include "src/euler/state.hpp"
#include "src/euler/eos_params.hpp"
#include "src/euler/grid/grid_utils.hpp"
#include "src/euler/reconstruction/muscl.hpp"

template<int DIM, int DIR>
inline void extract_line(
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::array<int, DIM>& N,
    const std::array<int, DIM>& base_idx,
    std::vector<Conserved<DIM>>& U_line,
    std::vector<int>& mat_line,
    std::vector<int>& id_line
)
{
    const int L = N[DIR];

    U_line.resize(L);
    mat_line.resize(L);
    id_line.resize(L);

    for (int i = 0; i < L; ++i) {
        std::array<int, DIM> idx = base_idx;
        idx[DIR] = i;

        const int id = flatten_index<DIM>(idx, N);

        U_line[i] = U[id];
        mat_line[i] = material_id.empty() ? 0 : material_id[id];
        id_line[i] = id;
    }
}

template<int DIM, typename EOS>
inline void reconstruct_line_interfaces_dispatch(
    int dir,
    const std::vector<Conserved<DIM>>& U_line,
    const std::vector<EOSParams>& cell_params,
    double dt,
    double dx,
    std::vector<Conserved<DIM>>& UL_face,
    std::vector<Conserved<DIM>>& UR_face,
    double rho_floor = 1e-10,
    double p_floor = 1e-10
)
{
    if constexpr (DIM >= 1) {
        if (dir == 0) {
            reconstruct_line_interfaces<DIM, 0, EOS>(
                U_line, cell_params, dt, dx,
                UL_face, UR_face,
                rho_floor, p_floor
            );
            return;
        }
    }

    if constexpr (DIM >= 2) {
        if (dir == 1) {
            reconstruct_line_interfaces<DIM, 1, EOS>(
                U_line, cell_params, dt, dx,
                UL_face, UR_face,
                rho_floor, p_floor
            );
            return;
        }
    }

    if constexpr (DIM >= 3) {
        if (dir == 2) {
            reconstruct_line_interfaces<DIM, 2, EOS>(
                U_line, cell_params, dt, dx,
                UL_face, UR_face,
                rho_floor, p_floor
            );
            return;
        }
    }

    throw std::runtime_error("invalid direction");
}

template<int DIM>
inline int compute_num_lines(int dir, const std::array<int, DIM>& N)
{
    int num = 1;
    for (int d = 0; d < DIM; ++d)
        if (d != dir) num *= N[d];
    return num;
}

template<int DIM>
inline void line_id_to_base_idx(
    int line_id,
    int dir,
    const std::array<int, DIM>& N,
    std::array<int, DIM>& base_idx
)
{
    int tmp = line_id;

    for (int d = DIM - 1; d >= 0; --d) {
        if (d == dir) {
            base_idx[d] = 0;
            continue;
        }

        base_idx[d] = tmp % N[d];
        tmp /= N[d];
    }
}

template<int DIM>
inline void extract_line_dispatch(
    int dir,
    const std::vector<Conserved<DIM>>& U,
    const std::vector<int>& material_id,
    const std::array<int, DIM>& N,
    const std::array<int, DIM>& base_idx,
    std::vector<Conserved<DIM>>& U_line,
    std::vector<int>& mat_line,
    std::vector<int>& id_line
)
{
    if constexpr (DIM >= 1)
        if (dir == 0) { extract_line<DIM,0>(U,material_id,N,base_idx,U_line,mat_line,id_line); return; }

    if constexpr (DIM >= 2)
        if (dir == 1) { extract_line<DIM,1>(U,material_id,N,base_idx,U_line,mat_line,id_line); return; }

    if constexpr (DIM >= 3)
        if (dir == 2) { extract_line<DIM,2>(U,material_id,N,base_idx,U_line,mat_line,id_line); return; }

    throw std::runtime_error("invalid direction");
}