#pragma once

#include <algorithm>
#include <vector>

// [0] Metadata for one tracked signed-distance field.
struct TrackedInterface {
    int negative_material_id = -1;
    int component_id = -1;
    std::vector<int> negative_side_material_ids{};
};

inline bool tracked_interface_contains_negative_material(
    const TrackedInterface& tracked,
    int material_id
)
{
    if (!tracked.negative_side_material_ids.empty()) {
        return std::find(
            tracked.negative_side_material_ids.begin(),
            tracked.negative_side_material_ids.end(),
            material_id
        ) != tracked.negative_side_material_ids.end();
    }

    return tracked.negative_material_id == material_id;
}
