#pragma once

// [0] Metadata for one tracked signed-distance field.
//     phi < 0 always identifies the tracked component/material.
struct TrackedInterface {
    int negative_material_id = -1;
    int component_id = -1;
};
