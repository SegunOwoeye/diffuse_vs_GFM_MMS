#pragma once

#include <vector>

namespace fv {

template<int DIM, typename State, typename SweepFn, typename AfterDirectionFn>
inline void advance_split_directions(
    std::vector<State>& U_stage,
    SweepFn&& sweep_direction,
    AfterDirectionFn&& after_direction
)
{
    for (int dir = 0; dir < DIM; ++dir) {
        std::vector<State> U_next = U_stage;

        sweep_direction(dir, U_stage, U_next);
        after_direction(dir, U_next);

        U_stage.swap(U_next);
    }
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
    std::vector<State> U_accum = U_base;

    for (int dir = 0; dir < DIM; ++dir) {
        std::vector<State> U_dir = U_base;

        sweep_direction(dir, U_base, U_dir);
        accumulate_delta(dir, U_accum, U_dir, U_base);
    }

    after_accumulation(U_accum);
    U_stage.swap(U_accum);
}

} 
