#pragma once

#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace runtime {

struct OpenMPRuntimeInfo {
    bool compiled = false;
    int max_threads = 1;
    int observed_threads = 1;
    int dynamic_enabled = 0;
    std::string num_threads_env;
    std::string proc_bind_env;
    std::string places_env;
    std::string schedule_env;
    std::string quant_cpu_list_env;
    std::string cpus_allowed_list;
    std::string thread_cpus_allowed_lists;
};

inline std::string getenv_string(const char* name)
{
    if (const char* value = std::getenv(name)) {
        return value;
    }
    return {};
}

inline std::string read_linux_cpus_allowed_list()
{
    std::ifstream status("/proc/self/status");
    if (!status) {
        return {};
    }

    std::string line;
    const std::string prefix = "Cpus_allowed_list:";
    while (std::getline(status, line)) {
        if (line.rfind(prefix, 0) == 0) {
            std::string value = line.substr(prefix.size());
            const auto first = value.find_first_not_of(" \t");
            return first == std::string::npos ? std::string{} : value.substr(first);
        }
    }
    return {};
}

inline std::string read_linux_thread_cpus_allowed_list()
{
#ifdef __linux__
    const auto tid = static_cast<long>(syscall(SYS_gettid));
    std::ifstream status("/proc/self/task/" + std::to_string(tid) + "/status");
    if (!status) {
        return {};
    }

    std::string line;
    const std::string prefix = "Cpus_allowed_list:";
    while (std::getline(status, line)) {
        if (line.rfind(prefix, 0) == 0) {
            std::string value = line.substr(prefix.size());
            const auto first = value.find_first_not_of(" \t");
            return first == std::string::npos ? std::string{} : value.substr(first);
        }
    }
#endif
    return {};
}

inline std::string join_allowed_lists(const std::set<std::string>& values)
{
    std::ostringstream out;
    bool first = true;
    for (const auto& value : values) {
        if (value.empty()) {
            continue;
        }
        if (!first) {
            out << ";";
        }
        out << value;
        first = false;
    }
    return out.str();
}

inline OpenMPRuntimeInfo observe_openmp_runtime()
{
    OpenMPRuntimeInfo info{};

    info.num_threads_env = getenv_string("OMP_NUM_THREADS");
    info.proc_bind_env = getenv_string("OMP_PROC_BIND");
    info.places_env = getenv_string("OMP_PLACES");
    info.schedule_env = getenv_string("OMP_SCHEDULE");
    info.quant_cpu_list_env = getenv_string("QUANT_CPU_LIST");
    info.cpus_allowed_list = read_linux_cpus_allowed_list();

#ifdef _OPENMP
    info.compiled = true;
    info.max_threads = omp_get_max_threads();
    info.dynamic_enabled = omp_get_dynamic();

    int observed_threads = 1;
    std::set<std::string> thread_allowed_lists;
    #pragma omp parallel
    {
        #pragma omp single
        observed_threads = omp_get_num_threads();

        const std::string allowed = read_linux_thread_cpus_allowed_list();
        #pragma omp critical
        thread_allowed_lists.insert(allowed);
    }
    info.observed_threads = observed_threads;
    info.thread_cpus_allowed_lists = join_allowed_lists(thread_allowed_lists);
#endif

    return info;
}

} // namespace runtime
