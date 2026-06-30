#pragma once

#include "src/euler/eos_params.hpp"
#include "src/solid/elastoplastic/barton/state.hpp"

namespace dim::barton_dim {

struct Materials {
    ::EOSParams fluid{};
    solid::barton::TensorMaterial solid{};
};

} // namespace dim::barton_dim
