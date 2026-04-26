#master file to run all tests

#!/bin/bash

# [1] Compile Files
chmod +x tests/basic_toro_tests.sh
chmod +x tests/GFM_validation_tests.sh
chmod +x tests/DIM_validation_tests.sh


# [2] Clear CSV Output
CSV_DIR="data/csv"

if [ -d "$CSV_DIR" ]; then
    find "$CSV_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
else
    mkdir -p "$CSV_DIR"
fi

# [3] Run Files

bash ./tests/basic_toro_tests.sh
bash ./tests/GFM_validation_tests.sh
bash ./tests/DIM_validation_tests.sh
