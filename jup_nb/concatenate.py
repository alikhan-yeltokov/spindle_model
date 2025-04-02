#!/usr/bin/python3
# Define the directory containing the Excel files
import pandas as pd 
import os, sys
excel_files_directory = sys.argv[1]#"./2024-01-21/spindle_9205_CELL_FE_10_data"

concat_files_directory=sys.argv[2]

filename=sys.argv[3]
# List all Excel files in the directory
excel_files = [f for f in os.listdir(excel_files_directory) if f.endswith(".xlsx")]

# Create an empty DataFrame to store the concatenated data
concatenated_angle_data = pd.DataFrame()
concatenated_center_data = pd.DataFrame()
concatenated_ypos_data = pd.DataFrame()
# concatenated_time_data = pd.DataFrame()
concatenated_angle_RMS_data = pd.DataFrame()
concatenated_center_RMS_data = pd.DataFrame()
concatenated_N_pull_data=pd.DataFrame()
concatenated_N_push_data=pd.DataFrame()
concatenated_length_data=pd.DataFrame()

# Iterate through the Excel files and concatenate them horizontally
for excel_file in excel_files:
    file_path = os.path.join(excel_files_directory, excel_file)
    #print(f'file name{excel_file}')
    # Read the Excel file into a DataFrame
    try:
        angle_data = pd.read_excel(file_path, engine='openpyxl', sheet_name="Angle")
        center_data= pd.read_excel(file_path, engine='openpyxl', sheet_name="Center")
        # ypos_data= pd.read_excel(file_path, engine='openpyxl', sheet_name="Y-pos")
        N_pull_data=pd.read_excel(file_path, engine='openpyxl', sheet_name="N_pull")
        N_push_data=pd.read_excel(file_path, engine='openpyxl', sheet_name="N_push")
        # length_data=pd.read_excel(file_path, engine='openpyxl', sheet_name="Length")
        # time_data= pd.read_excel(file_path, engine='openpyxl', sheet_name="Time")
        # RMS_angle_data=pd.read_excel(file_path, engine='openpyxl', sheet_name="Angle RMS")
#         RMS_center_data=pd.read_excel(file_path, engine='openpyxl', sheet_name="Center RMS")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        continue

    # Concatenate the data horizontally by adding columns
    concatenated_angle_data = pd.concat([concatenated_angle_data, angle_data], axis=1)
    concatenated_center_data = pd.concat([concatenated_center_data, center_data], axis=1)
    # concatenated_ypos_data = pd.concat([concatenated_ypos_data, ypos_data], axis=1)
    # # concatenated_time_data = pd.concat([concatenated_time_data, time_data], axis=1)
    concatenated_N_pull_data=pd.concat([concatenated_N_pull_data, N_pull_data], axis=1)
    concatenated_N_push_data=pd.concat([concatenated_N_push_data, N_push_data], axis=1)
    # concatenated_length_data=pd.concat([concatenated_length_data, length_data], axis=1)
    # concatenated_angle_RMS_data = pd.concat([concatenated_angle_RMS_data, RMS_angle_data], axis=1)
#     concatenated_center_RMS_data = pd.concat([concatenated_center_RMS_data, RMS_center_data], axis=1)

# Reset the index of the concatenated DataFrame
#concatenated_data.reset_index(drop=True, inplace=True)

# Save the concatenated data to a new Excel file
file_name = filename + "_stats.xlsx"
output_excel_file = os.path.join(concat_files_directory, file_name)

#concatenated_data.to_excel(output_excel_file, index=False)
with pd.ExcelWriter(output_excel_file) as excel_writer:
    concatenated_angle_data.to_excel(excel_writer, sheet_name='Angle', index=True)
    concatenated_center_data.to_excel(excel_writer, sheet_name='Center', index=True)
    # concatenated_ypos_data.to_excel(excel_writer, sheet_name='Y-pos', index=True)
    # # concatenated_time_data.to_excel(excel_writer, sheet_name='Time', index=True)
    concatenated_N_pull_data.to_excel(excel_writer, sheet_name='N_pull', index=True)
    concatenated_N_push_data.to_excel(excel_writer, sheet_name='N_push', index=True)
    # concatenated_length_data.to_excel(excel_writer, sheet_name='Length', index=True)
    # concatenated_angle_RMS_data.to_excel(excel_writer, sheet_name='RMS Angle', index=True)
#     concatenated_center_RMS_data.to_excel(excel_writer, sheet_name='RMS Center', index=True)