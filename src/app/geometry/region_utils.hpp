#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "src/io/config.hpp"


// [0] Compute axis-aligned region volume
template<int DIM>
inline double region_volume(
    const Region<DIM>& r
)
{
    double vol = 1.0;

    for (int d = 0; d < DIM; ++d) {
        const double width = r.upper[d] - r.lower[d];

        if (width <= 0.0) {
            throw std::runtime_error("region_volume: non-positive region width");
        }

        vol *= width;
    }

    return vol;
}

// [1] Signed distance to axis-aligned box
template<int DIM>
inline double signed_distance_to_box(
    const std::array<double, DIM>& x,
    const Region<DIM>& region
)
{
    double outside_sq = 0.0;
    double inside_dist = std::numeric_limits<double>::max();
    bool inside = true;

    for (int d = 0; d < DIM; ++d) {
        if (x[d] < region.lower[d]) {
            const double dist = region.lower[d] - x[d];
            outside_sq += dist * dist;
            inside = false;
        }
        else if (x[d] > region.upper[d]) {
            const double dist = x[d] - region.upper[d];
            outside_sq += dist * dist;
            inside = false;
        }
        else {
            const double dist_to_lower = x[d] - region.lower[d];
            const double dist_to_upper = region.upper[d] - x[d];
            inside_dist = std::min(inside_dist, std::min(dist_to_lower, dist_to_upper));
        }
    }

    return inside ? -inside_dist : std::sqrt(outside_sq);
}

