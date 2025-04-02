#!/bin/bash

# Load the conda module (if not already loaded)
module load miniconda3

# Specify the path to your conda environment
conda_env_path=/home/ayn6k/.conda/envs/spindle

# Activate the conda environment
source activate "$conda_env_path"

# Check if the stats_directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <stats_directory>"
  exit 1
fi

# Assign the stats_directory argument


# Directory containing Excel files (change this to your directory)
excel_directory="$1"

# Iterate over each Excel file in the directory
for excel_file in "$excel_directory"/*.xlsx; do
  # Extract the base name of the file (without extension)
  name=$(basename "$excel_file" .xlsx)
  
  name_with_extension="$name.xlsx"
  echo "Processing file: $excel_file"
  
  echo "Extracted name: $name"
  # Run the python script with the extracted name and stats_directory
  python3 plots.py "$excel_directory" "$name_with_extension"
done
