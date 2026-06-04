#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

#include "src/solid/elastoplastic/barton.hpp"

int main(int argc, char** argv)
{
    try {
        if (argc < 2) {
            throw std::runtime_error("Usage: ./solid_main <config>");
        }

        const std::string config_file = argv[1];
        if (solid::barton::detect_model(config_file) != "barton") {
            throw std::runtime_error("solid_main currently supports model=barton configs");
        }
        return solid::barton::run(config_file);
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
