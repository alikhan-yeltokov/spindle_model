#!/bin/bash
# Load the conda module (if not already loaded)
module load miniconda3

# Specify the path to your conda environment
conda_env_path=/home/ayn6k/.conda/envs/spindle

# Activate the conda environment
source activate "$conda_env_path"

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <concatenate_directory> <stats_directory>"
  #  exit 1
fi
cc
concatenate_directory=$1
stats_directory=$2
#name="spindle_9213_model_FE_90deg_PINS_10_MT80_AL_0.25_1_pull_3.6_repel_0.25_push_4_spread_270_SL_1.8_mcd_0.025_ts_0.05_run_15"
name="spindle_9214_FE_AL_1_0.3_SL_1.9_ts_0.05_pull_3.6_PINS_15_MT_60_push_0"

#concatenate_directory="./2024-10-27/${name}/${name}_stats"
#stats_directory="./2024-10-27/${name}"

# Run concatenate.py with the specified directory
python3 concatenate.py "$concatenate_directory" "$stats_directory" "$name"

# Run stats.py with the specified directory
python3 plots.py "$stats_directory" "$name"



