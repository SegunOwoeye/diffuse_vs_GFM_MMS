#master file to run all tests

#!/bin/bash

# [1] Compile Files
chmod +x tests/basic_toro_tests.sh
chmod +x tests/GFM_validation_tests.sh
chmod +x tests/DIM_validation_tests.sh

# [2] Run Files

./tests/basic_toro_tests.sh
./tests/GFM_validation_tests.sh
./tests/DIM_validation_tests.sh