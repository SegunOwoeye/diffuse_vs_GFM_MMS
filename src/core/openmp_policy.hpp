#pragma once

#include <cstddef>

namespace runtime {

inline bool openmp_should_parallelize_cells(std::size_t item_count)
{
    return item_count > 512;
}

inline bool openmp_should_parallelize_lines(int line_count)
{
    return line_count > 1;
}

} // namespace runtime
