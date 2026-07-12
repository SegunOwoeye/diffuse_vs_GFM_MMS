#pragma once

#include <vector>

namespace core::fv {

template<int DIM, typename State, typename SweepFn, typename AfterDirectionFn>
inline void advance_split_directions(
    std::vector<State>& stage,
    SweepFn&& sweep_direction,
    AfterDirectionFn&& after_direction)
{
    for (int dir = 0; dir < DIM; ++dir) {
        std::vector<State> next(stage.size());
        sweep_direction(dir, stage, next);
        after_direction(dir, next);
        stage.swap(next);
    }
}

template<
    int DIM,
    typename State,
    typename SweepFn,
    typename AccumulateDeltaFn,
    typename AfterAccumulationFn>
inline void advance_unsplit_directions(
    const std::vector<State>& base,
    std::vector<State>& stage,
    SweepFn&& sweep_direction,
    AccumulateDeltaFn&& accumulate_delta,
    AfterAccumulationFn&& after_accumulation)
{
    std::vector<State> accumulated = base;
    for (int dir = 0; dir < DIM; ++dir) {
        std::vector<State> directional = base;
        sweep_direction(dir, base, directional);
        accumulate_delta(dir, accumulated, directional, base);
    }
    after_accumulation(accumulated);
    stage.swap(accumulated);
}

} // namespace core::fv
