#!/bin/bash
#SBATCH --partition general
#SBATCH --job-name FE_rpl
#SBATCH --array 1-2
#SBATCH --mem-per-cpu=4gb
#SBATCH -t 0-8:00
#SBATCH --output=./outputs/output-%A_%a.out

# Load the conda module (if not already loaded)
module load miniconda3

# Specify the path to your conda environment
conda_env_path=/home/ayn6k/.conda/envs/spindle

# Activate the conda environment
source activate "$conda_env_path"


# Define the directory where you want to create the folders

# Get today's date in YYYY-MM-DD format
today_date=$(date +'%Y-%m-%d')

# Define the directory where you want to create the folders
target_directory=$(pwd)

# Check if the folder with today's date already exists
date_folder="$target_directory/$today_date"
if [ ! -d "$date_folder" ]; then
    mkdir -p "$date_folder"
    echo "Created folder with today's date: $date_folder"
else
    echo "Folder with today's date already exists: $date_folder"
fi

# Get today's date in YYYY-MM-DD format
# DC=double correction angle and position
#RT=rotation then translation
#mpd=min_push_dist
#name="spindle_9214_FE_AL_1_0.3_SL_1.9_ts_1_pull_3.6_PINS_15_MT_50_push_4"
# Check if the folder with ensemble date already exists

name=$1
ensemble_folder="$target_directory/$today_date/${name}"
if [ ! -d "$ensemble_folder" ]; then
    mkdir -p "$ensemble_folder"
    echo "Created folder ensemb;e: $ensemble_folder"
else
    echo "Folder with ensemble already exists: $ensemble_folder"
fi

images_directory="${name}_images"
stats_directory="${name}_stats"


images_folder="${target_directory}/$today_date/${name}/$images_directory"
if [ ! -d "$images_folder" ]; then
    mkdir -p "$images_folder"
    echo "Created 'test' folder: $images_folder"
else
    echo "'Test' folder already exists: $images_folder"
fi

data_folder="${target_directory}/$today_date/${name}/$stats_directory"
if [ ! -d "$data_folder" ]; then
    mkdir -p "$data_folder"
    echo "Created 'test' folder: $data_folder"
else
    echo "'Test' folder already exists: $data_folder"
fi

# Run the Python script and pass the path to the "test" folder as an argument

python3 spindle_v9214_FE.py "$images_folder" "$data_folder" "$name"




