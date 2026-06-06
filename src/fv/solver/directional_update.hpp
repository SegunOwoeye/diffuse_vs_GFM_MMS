#pragma once

#include "src/core/fv/directional_update.hpp"

namespace fv {

template<int DIM, typename State, typename SweepFn, typename AfterDirectionFn>
inline void advance_split_directions(
    std::vector<State>& U_stage,
    SweepFn&& sweep_direction,
    AfterDirectionFn&& after_direction
)
{
    core::fv::advance_split_directions<DIM>(
        U_stage,
        std::forward<SweepFn>(sweep_direction),
        std::forward<AfterDirectionFn>(after_direction));
}


template<
    int DIM,
    typename State,
    typename SweepFn,
    typename AccumulateDeltaFn,
    typename AfterAccumulationFn
>
inline void advance_unsplit_directions(
    const std::vector<State>& U_base,
    std::vector<State>& U_stage,
    SweepFn&& sweep_direction,
    AccumulateDeltaFn&& accumulate_delta,
    AfterAccumulationFn&& after_accumulation
)
{
    core::fv::advance_unsplit_directions<DIM>(
        U_base,
        U_stage,
        std::forward<SweepFn>(sweep_direction),
        std::forward<AccumulateDeltaFn>(accumulate_delta),
        std::forward<AfterAccumulationFn>(after_accumulation));
}

} 
