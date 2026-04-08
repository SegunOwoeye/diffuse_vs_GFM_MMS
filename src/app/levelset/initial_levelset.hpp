#pragma once

#include <array>
#include <stdexcept>
#include <vector>

#include "src/app/geometry/region_utils.hpp"
#include "src/io/config.hpp"
#include "src/euler/grid/grid_utils.hpp"


// [0] Initial level set container
template<int DIM>
struct InitialLevelSetData {
    std::vector<std::vector<double>> phi_list;
    std::vector<int> phi_material_ids;
    int background_material_id = 0;
};


// [1] Build initial level sets from region ICs for multimaterial GFM
template<int DIM>
inline InitialLevelSetData<DIM> initialise_phi_data_from_regions(
    const Config<DIM>& cfg,
    const std::array<int, DIM>& N
)
{
    InitialLevelSetData<DIM> out{};

    if (cfg.interface_method != "GFM") {
        return out;
    }

    if (!cfg.use_level_set) {
        throw std::runtime_error("initialise_phi_data_from_regions: GFM requires use_level_set = true");
    }

    if (cfg.initial_condition != "regions") {
        throw std::runtime_error(
            "initialise_phi_data_from_regions: current multimaterial GFM initialisation only supports region ICs"
        );
    }

    if (cfg.regions.size() < 2) {
        throw std::runtime_error(
            "initialise_phi_data_from_regions: GFM requires at least 2 regions/materials"
        );
    }

    // [1.1] Select background material from largest region
    int bg_region_idx = 0;
    double bg_volume = region_volume<DIM>(cfg.regions[0]);

    for (int r = 1; r < static_cast<int>(cfg.regions.size()); ++r) {
        const double vol = region_volume<DIM>(cfg.regions[r]);

        if (vol > bg_volume) {
            bg_volume = vol;
            bg_region_idx = r;
        }
    }

    out.background_material_id = cfg.regions[bg_region_idx].material_id;

    int total_cells = 1;
    for (int d = 0; d < DIM; ++d) {
        if (N[d] <= 0) {
            throw std::runtime_error("initialise_phi_data_from_regions: invalid grid size");
        }
        total_cells *= N[d];
    }

    std::array<double, DIM> dx{};
    for (int d = 0; d < DIM; ++d) {
        dx[d] = (cfg.domain_max[d] - cfg.domain_min[d]) / static_cast<double>(N[d]);
    }

    // [1.2] Build one signed-distance field per non-background region
    for (int r = 0; r < static_cast<int>(cfg.regions.size()); ++r) {
        if (r == bg_region_idx) {
            continue;
        }

        const Region<DIM>& region = cfg.regions[r];

        std::vector<double> phi(total_cells, 0.0);
        std::array<int, DIM> idx{};

        for (int linear = 0; linear < total_cells; ++linear) {
            int tmp = linear;

            for (int d = DIM - 1; d >= 0; --d) {
                idx[d] = tmp % N[d];
                tmp /= N[d];
            }

            const auto x = compute_cell_center<DIM>(idx, cfg.domain_min, dx);
            phi[linear] = signed_distance_to_box<DIM>(x, region);
        }

        out.phi_list.push_back(std::move(phi));
        out.phi_material_ids.push_back(region.material_id);
    }

    return out;
}

