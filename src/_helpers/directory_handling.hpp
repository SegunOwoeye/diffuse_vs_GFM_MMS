#pragma once
#include <filesystem>
#include <string>

// Create directory handles both directory paths and file paths
inline void ensure_directory(const std::string& path)
{
    if (path.empty()) return;

    std::filesystem::path p(path);

    // If path has extension -> treat as file -> use parent
    if (p.has_extension()) {
        std::filesystem::create_directories(p.parent_path());
    } else {
        std::filesystem::create_directories(p);
    }
}