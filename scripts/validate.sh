#!/bin/bash

MAX_THREADS="16" # <--- Include maximum number of threads (physical cores)

# Define the Python file to run
PYTHON_FILE="benchmark/compare_matmul_cuda.py"

# Define lists of values for each variable
OPS=("ax" "adx" "dadx")
NCOLUMNS=(512)
ITERATIONS=(50)
RTOL=1e-5
ATOL=0
# Define ALPHAS implicitly by dataset (only datasets that appear here are used)
declare -A ALPHA_MAP
ALPHA_MAP["ca-HepPh"]=""  # Space-separated values instead of an array
ALPHA_MAP["ca-AstroPh"]=""  # Empty string means revert to default (no --alpha)
# ALPHA_MAP["Cora"]=""  # Space-separated values instead of an array
# ALPHA_MAP["PubMed"]=""  # Empty string means revert to default (no --alpha)
# ALPHA_MAP["COLLAB"]=""  # Space-separated values instead of an array
# ALPHA_MAP["coPapersCiteseer"]=""  # Empty string means revert to default (no --alpha)
# ALPHA_MAP["coPapersDBLP"]=""  # Space-separated values instead of an array
# ALPHA_MAP["ogbn-proteins-raw"]=""  # Empty string means revert to default (no --alpha)

# Extract dataset names automatically
DATASETS=(${!ALPHA_MAP[@]})

# Temporary file to store results
mkdir -p results

# Temporary file to store results
RESULTS_FILE="results/validate_results.txt"
> $RESULTS_FILE
> val_temp_results.txt

# Generate all possible combinations
for OP in "${OPS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for COL in "${NCOLUMNS[@]}"; do
            for ITER in "${ITERATIONS[@]}"; do
                # Read alpha values as an array
                IFS=' ' read -r -a ALPHAS <<< "${ALPHA_MAP[$DATASET]}"
                if [ -z "${ALPHAS[*]}" ]; then
                ALPHAS=("")  # Ensures at least one iteration without --alpha
                fi
                for ALPHA in "${ALPHAS[@]}"; do
                    ARGS="--operation $OP --dataset $DATASET --iterations $ITER --rtol $RTOL --atol $ATOL"
                    if [ -n "$COL" ]; then
                        ARGS="$ARGS --columns $COL"
                    fi
                    if [ -n "$ALPHA" ]; then
                        ARGS="$ARGS --alpha $ALPHA"
                    fi
                    
                    echo "Running: OMP_NUM_THREADS=$MAX_THREADS GOMP_CPU_AFFINITY=\"0-$((MAX_THREADS - 1))\" python $PYTHON_FILE $ARGS"
                    
                    # Execute the Python script with the environment variables
                    OUTPUT=$(OMP_NUM_THREADS=$MAX_THREADS GOMP_CPU_AFFINITY="0-$((MAX_THREADS - 1))" python $PYTHON_FILE $ARGS)
                    
                    # Extract performance metrics from the output
                    PASSED=$(echo "$OUTPUT" | grep -oP "Passed: \K[\d\.]+")
                    FAILED=$(echo "$OUTPUT" | grep -oP "Failed: \K[\d\.]+")
                    
                    # Save the results in a structured format
                    echo -e "[nthreads: $THREAD]\t[$ARGS]:\t\t\tPassed Test: $PASSED\tFailed Tests: $FAILED" >> val_temp_results.txt
                done
            done
        done
    done
done

# Print a pretty table of the results
column -t -s $'\t' val_temp_results.txt > $RESULTS_FILE
rm val_temp_results.txt