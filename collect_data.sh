#!/bin/bash

# --- Configuration ---
INSTANCES_IN_PARALLEL=10  # This is your 'x'
TOTAL_BATCHES=10         # This is your 'y'
PYTHON_EXEC="/Users/jiaanli/miniforge3/envs/cbf/bin/python"
SCRIPT_PATH="/Users/jiaanli/ura/research-geo-diff-opt-layer/simulator.py"

echo "Starting data collection: $TOTAL_BATCHES batches of $INSTANCES_IN_PARALLEL instances."

for (( i=1; i<=TOTAL_BATCHES; i++ ))
do
    echo "Starting Batch $i of $TOTAL_BATCHES..."
    
    for (( j=1; j<=INSTANCES_IN_PARALLEL; j++ ))
    do
        # Run the command in the background using '&'
        $PYTHON_EXEC $SCRIPT_PATH &
    done

    # 'wait' tells the script to pause until all background jobs in this batch are done
    wait
    echo "Batch $i completed."
done

echo "All data collection tasks are finished!"