#!/bin/bash
# Load the conda module (if not already loaded)
module load miniconda3

# Specify the path to your conda environment
conda_env_path=/home/ayn6k/.conda/envs/spindle

# Activate the conda environment
source activate "$conda_env_path"
# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: ./automate_concat.sh <path_to_today_date_folder>"
    exit 1
fi

    
# Get the absolute path to the base folder (e.g., today_date)
BASE_FOLDER=$(realpath "$1")


PROJECT_PATH="./char_project_non_opt_SL_1.9_forces_3.6_4_0"
mkdir -p "$PROJECT_PATH"
"$PROJECT_PATH"


# Verify that the provided folder exists
if [ ! -d "$BASE_FOLDER" ]; then
    echo "Error: Folder '$BASE_FOLDER' does not exist."
    exit 1
fi

# Loop through all subfolders in the base folder
for folder in "$BASE_FOLDER"/*/; do
    # Extract the folder name (e.g., run_x)
    folder_name=$(basename "$folder")

    # Construct the stats folder path (e.g., /full/path/to/run_x/run_x_stats)
    stats_folder="${folder}${folder_name}_stats"

    # Check if the stats folder exists
    if [ ! -d "$stats_folder" ]; then
        echo "Warning: Stats folder not found for '$folder_name'. Skipping."
        continue
    fi

    # Run the concatenate.py script with the full path and folder name
    echo "Processing $folder_name..."
    python3 concatenate.py "$stats_folder" "$PROJECT_PATH" "$folder_name" 
    
    #python3 plots.py "$stats_folder" "$folder_name"
    
    # Check if the script ran successfully
    if [ $? -eq 0 ]; then
        echo "Successfully processed $folder_name."
    else
        echo "Error: Failed to process $folder_name."
    fi
done
