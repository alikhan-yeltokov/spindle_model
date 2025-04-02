#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA
import shapely.geometry
from shapely import box, LineString, normalize, Polygon
import os, sys
import ast
import cv2
import re
from matplotlib.patches import Ellipse
# from IPython.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))


# In[19]:

def extract_parameters_name(directory_path, filename):
    

    excel_file_name = filename
    excel_file_path = os.path.join(directory_path, excel_file_name)
    # Define the strings to search for
    search_strings = ['endo', 'fe','celegans']

    # Check if each search string is present in the file name
    found_endo, found_FE, found_celegans = [string in excel_file_name.lower() for string in search_strings]
    if found_endo:
        print("Endo data found!")
    if found_FE:
        print("FE data found!")
    if found_celegans:
        print("celegans data found!")
 
    
    if found_endo:
        
        match = re.search(r'cell_([\d.]+)', excel_file_path)
        if match:
            c = int(float(match.group(1)))
            #print("Time Step:", time_step)
        else:
            #print("Time step not found in the file name.")
            c=1
    
    match = re.search(r'ts_([\d.]+)', excel_file_path)
    if match:
        time_step = float(match.group(1))
         #print("Time Step:", time_step)
    else:
         #print("Time step not found in the file name.")
         time_step=1
         
    return time_step, 0, excel_file_name, found_endo, found_FE, found_celegans

# def extract_parameters_name(directory_path):
    
#     excel_files = [file for file in os.listdir(directory_path) if file.endswith('.xlsx')]

#     # Check if there's exactly one Excel file
#     if len(excel_files) == 1:
#         excel_file = excel_files[0]

#     files = os.listdir(directory_path)

#     # Filter only Excel files
#     excel_files = [file for file in files if file.endswith('.xlsx')]

#     if len(excel_files) == 1:
#         # Extract the file name from the list
#         excel_file_name = excel_files[0]
#         excel_file_path = os.path.join(directory_path, excel_file_name)
#         # Define the strings to search for
#         search_strings = ['endo', 'fe','celegans']

#         # Check if each search string is present in the file name
#         found_endo, found_FE, found_celegans = [string in excel_file_name.lower() for string in search_strings]
#         if found_endo:
#             print("Endo data found!")
#         if found_FE:
#             print("FE data found!")
#         if found_celegans:
#             print("celegans data found!")
 
#     else:
#         print("Error: No or multiple Excel files found in the directory.")


#     if found_endo:
        
#         match = re.search(r'cell_([\d.]+)', excel_file_path)
#         if match:
#             c = int(float(match.group(1)))
#             #print("Time Step:", time_step)
#         else:
#             #print("Time step not found in the file name.")
#             c=1
    
            
#     match = re.search(r'ts_([\d.]+)', excel_file_path)
#     if match:
#         time_step = float(match.group(1))
#          #print("Time Step:", time_step)
#     else:
#          #print("Time step not found in the file name.")
#          time_step=1
#     return time_step, 0, excel_file_name, found_endo, found_FE, found_celegans
 
def enhance_cell(cell):
    new_cell=[]
    max_interj_dist=0.005
    #max_interj_dist=natural_spacing
    for i in range (len(cell)):
        if(i==len(cell)-1):
            next=0
        else:
            next=i+1
        dist=LA.norm([cell[i,0]-cell[next,0],cell[i,1]-cell[next,1]])
        if (dist>max_interj_dist):
            new_cell.append(cell[i])
            spread=np.linspace(0,dist,int(dist/max_interj_dist))[1:-1]
            vec=np.array([cell[next,0]-cell[i,0],cell[next,1]-cell[i,1]])/dist#+cell[i]
            for j in range (len(spread)):
                new_add=spread[j]*vec+cell[i]
                new_cell.append(new_add)    
        else:
            new_cell.append(cell[i])
    new_cell_ar=np.zeros((len(new_cell),2))
    for i in range (len(new_cell)):
        new_cell_ar[i]=new_cell[i]
    return new_cell_ar

def get_contours(image_path):

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise and detail
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours in the edge-detected image
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # rescale=[315,400,360,400,420]
        # x=contours[0]
        # cell_input=np.zeros((np.shape(x)[0],2))
        # for i in range (len(cell_input)):
        #     cell_input[i,0]=x[i,0,0]/rescale[c-1]
        #     cell_input[i,1]=x[i,0,1]/rescale[c-1]
        smoothed_contours = []

        # Smoothen each contour
        for contour in contours:
            # Approximate the contour with a smoother curve
            epsilon = 0.0001 * cv2.arcLength(contour, True)
            smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
            smoothed_contours.append(smoothed_contour)
    
        # Concatenate all contour points into a single array
        all_points = np.concatenate([contour.squeeze() for contour in smoothed_contours])
        all_points[:, 1] = -all_points[:, 1]
        return all_points
    
def get_cell(image_path, c):
        all_points=get_contours(image_path)
        cell_input=np.zeros((len(all_points),2))
        for i in range (len(cell_input)):
            cell_input[i,0]=all_points[i,0]/rescale[c-1]
            cell_input[i,1]=all_points[i,1]/rescale[c-1]
        # print(f'cell input top,bot,left,right={np.max(cell_input[:,1]),np.min(cell_input[:,1]), np.min(cell_input[:,0]), np.max(cell_input[:,0]) }')
        cell=enhance_cell(cell_input)

        spindle_contour=get_contours('./spindles/spindle_'+str(c)+'/Spindle_'+str(starts[c-1])+'.jpg')
        spindle_input=np.zeros((len(spindle_contour),2))
        for i in range (len(spindle_input)):
            spindle_input[i,0]=spindle_contour[i,0]/rescale[c-1]
            spindle_input[i,1]=spindle_contour[i,1]/rescale[c-1]
        spindle_length=max(spindle_input[:,1])-min(spindle_input[:,1])
       
        shapely_string = LineString(spindle_input)
        spindle_center=np.array([shapely_string.centroid.x,shapely_string.centroid.y])
        cell=cell-spindle_center
        
        return cell
    
    
# Define a function for coordinate conversion
def convert_coordinates(coord):
    if isinstance(coord, (str, list)):
        # Split the string using both commas and spaces, remove empty strings
        values = [float(val) for val in str(coord).replace(',', ' ').strip('[]').split() if val]
        return values if len(values) == 2 else [float(coord), 0]
    else:
        return [float(coord), 0]

from matplotlib.patches import Ellipse

def make_center_time_plots(directory_path, filename):
    # Extract parameters and file paths
    time_step, c, excel_file_name, found_endo, found_FE, found_celegans = extract_parameters_name(directory_path, filename)
    excel_file_path = os.path.join(directory_path, excel_file_name)
    df_center = pd.read_excel(excel_file_path, sheet_name='Center')

    # Get the number of columns and rows for subplots
    num_columns = len(df_center.columns[1:])
    num_rows = 4

    # Calculate the number of subplots needed
    num_subplots = num_columns // num_rows + int(num_columns % num_rows > 0)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_subplots, figsize=(25, 25))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Iterate through all columns
    for idx, column_name in enumerate(df_center.columns[1:21]):
        # Apply the updated coordinate conversion function
        df_center[column_name] = df_center[column_name].apply(convert_coordinates)

        # Extract X and Y coordinates over time
        x_values = df_center[column_name].apply(lambda coord: coord[0])
        y_values = df_center[column_name].apply(lambda coord: coord[1])
        time_values = df_center.index  # Time steps

        # Plot X and Y positions as a function of time
        axes[idx].plot(time_values, x_values, label=f'{column_name} - X Position', color='blue')
        axes[idx].plot(time_values, y_values, label=f'{column_name} - Y Position', color='orange')

        # Customize each subplot
        axes[idx].set_title(f'Center Position vs Time - {column_name}')
        axes[idx].set_xlabel('Time Step')
        axes[idx].set_ylabel('Position')
        axes[idx].legend()

        # Add horizontal lines for the final Y position (if needed)
        axes[idx].axhline(y=y_values.iloc[-1], color='gold', linestyle='--', label=f'Final Y Position: {y_values.iloc[-1]:.2f}')

        # Set limits and aspect ratio (if needed)
        if found_endo:
            cell_i = get_cell('./cells/cell_' + str(c) + '/Mask_' + str(starts[c-1]) + '.jpg', c)
            cell_f = get_cell('./cells/cell_' + str(c) + '/Mask_' + str(ends[c-1]) + '.jpg', c)
            top = max(cell_i[:, 1]) + 0.5
            bottom = min(cell_i[:, 1]) - 0.5
            axes[idx].set_ylim(bottom, top)
        elif found_FE:
            axes[idx].set_ylim(-2, 2)  # Adjust limits for FE case
        elif found_celegans:
            axes[idx].set_ylim(-3, 3)  # Adjust limits for C. elegans case

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the plot
    file_name = filename + '_center_time_plots.pdf'
    save_path = os.path.join(directory_path, file_name)
    plt.savefig(save_path)

    print("Center position vs time plots are ready")
# def make_center_scatter_plots(directory_path):
    
#     time_step, c, excel_file_name, found_endo, found_FE, found_celegans =extract_parameters_name(directory_path, filename)

#     excel_file_path = os.path.join(directory_path, excel_file_name)
#     df_center = pd.read_excel(excel_file_path,sheet_name='Center')
        
#     # df_exp=pd.read_excel('./cells_data.xlsx', sheet_name='cell_'+str(c))
#     # ys=np.asarray(df_exp['Spindle center y'])
#     # Get the number of columns and rows for subplots
#     num_columns =len(df_center.columns[1:])
#     num_rows = 4

#     # Calculate the number of subplots needed
#     num_subplots = num_columns // num_rows + int(num_columns % num_rows > 0)

#     # Create subplots
#     fig, axes = plt.subplots(nrows=num_rows, ncols=num_subplots, figsize=(25, 25))

#     # Flatten the axes array for easy iteration
#     axes = axes.flatten()

#     # Iterate through all columns
#     for idx, column_name in enumerate(df_center.columns[1:21]):
#         # Apply the updated coordinate conversion function
#         df_center[column_name] = df_center[column_name].apply(convert_coordinates)

#         # Plotting the scatter plot with equal aspect ratio
#         x_values = df_center[column_name].apply(lambda coord: coord[0])
#         y_values = df_center[column_name].apply(lambda coord: coord[1])

#         # Plotting the scatter plot for each column in a subplot
#         scatter = axes[idx].scatter(x_values, y_values, c=df_center.index, cmap='viridis', marker='o', label=column_name)

#         # Adding an ellipse with a=2.5, b=1.5 to each subplot
#     #     ellipse = Ellipse(xy=(0, 0), width=5, height=3, edgecolor='red', linewidth=2, fill=False)
#     #     axes[idx].add_patch(ellipse)
#         if found_endo:
#             cell_i=get_cell('./cells/cell_'+str(c)+'/Mask_'+str(starts[c-1])+'.jpg', c)
#             cell_f=get_cell('./cells/cell_'+str(c)+'/Mask_'+str(ends[c-1])+'.jpg', c)
#             axes[idx].plot(cell_i[:,0], cell_i[:,1], label='Cell boundary', color='blue')
#             axes[idx].plot(cell_f[:,0], cell_f[:,1], label='Cell boundary', color='gold')
#         elif found_FE:
#             ellipse = Ellipse(xy=(0, 0), width=2, height=2, edgecolor='blue', linewidth=2, fill=False)
#             axes[idx].add_patch(ellipse)
#             a,b=1,1
#         elif found_celegans:
#             ellipse = Ellipse(xy=(0, 0), width=5, height=3, edgecolor='blue', linewidth=2, fill=False)
#             axes[idx].add_patch(ellipse)
#             a,b=2.5,1.5
#         # Customize each subplot as needed
#         axes[idx].set_title(f'Spindle center - {column_name}')
#         axes[idx].set_xlabel('X-axis')
#         axes[idx].set_ylabel('Y-axis')
        
#         axes[idx].axhline(y=y_values.iloc[-1], xmin=-2, xmax=2, c="gold", linewidth=2,linestyle='--', zorder=0, label='theory y-position= %.2f'%(y_values.iloc[-1]))
#         # axes[idx].axhline(y=y_final[c-1], xmin=-2, xmax=2, c="black", linewidth=2, linestyle='--', zorder=0, label='exp y-position= %.2f'%(ys[-1]))#_final[c-1]))
#         if found_endo:
#             top = max(cell_i[:,1])+0.5
#             bottom =min(cell_i[:,1])-0.5
#             axes[idx].set_ylim(bottom, top)
#             axes[idx].set_xlim(-2, 2)
#         else:
#             axes[idx].set_ylim(-a-1, a+1)
#             axes[idx].set_xlim(-b-1, b+1)
#         axes[idx].set_aspect('equal', adjustable='box')
#         axes[idx].legend()

#     #cbar = fig.colorbar(scatter, ax=axes, label='Time step', pad=0.2)

#     # Adjust layout for better spacing
#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter to leave space for colorbar
#     file_name=filename+'_centers.pdf'
#     save_path = os.path.join(directory_path, file_name)

#     plt.savefig(save_path) 
    
#     print("scatterplots are ready")
    
def time_csv(directory_path):

    
    time_step, c, excel_file_name, found_endo, found_FE, found_celegans = extract_parameters_name(directory_path, filename)

   
    excel_file_path = os.path.join(directory_path, excel_file_name)
    
    # model_df = pd.read_excel("./2024-10-25/spindle_9213_FE_cutoff_15_deg_spindle_90deg_PINS_10_MT80_OLD_AL_1_0.3_pull_3.6_repel_0.25_push_4_spread_270_SL_1.9_ts_0.05_run_59/spindle_9213_FE_cutoff_15_deg_spindle_90deg_PINS_10_MT80_OLD_AL_1_0.3_pull_3.6_repel_0.25_push_4_spread_270_SL_1.9_ts_0.05_run_59_stats.xlsx",sheet_name='Y-pos')
    
    model_df = pd.read_excel(excel_file_path,sheet_name='Y-pos')
    model_df = model_df.iloc[:, 1:]

    non_null_counts = model_df.notnull().sum()*time_step

    # Save the counts to a CSV file
    output_filename=filename+'_time.csv'
    save_path = os.path.join(directory_path, output_filename)
    non_null_counts.to_csv(save_path, header=["Non-Null Row Count"])
    
    print(f"Non-null row counts saved to {save_path}.")

    # # Create a new DataFrame to store the result
    # result_df = pd.DataFrame({'Number of Columns': [num_columns]})
    
    # # Specify the file path where you want to save the CSV
    # output_filename=filename+'_time.csv'
    # save_path = os.path.join(directory_path, output_filename)
    
    # # Save to CSV
    # result_df.to_csv(save_path, index=False)
    

   
    
    
def make_center_average_plots(directory_path):
    time_step, c, excel_file_name, found_endo, found_FE, found_celegans = extract_parameters_name(directory_path, filename)

   
    excel_file_path = os.path.join(directory_path, excel_file_name)
    df_center = pd.read_excel(os.path.join(directory_path, excel_file_name),sheet_name='Y-pos')
    # df_time = pd.read_excel(os.path.join(directory_path, excel_file),sheet_name='Time')
    
    # df_exp=pd.read_excel('./cells_data_normalized_y.xlsx', sheet_name='cell_'+str(c))
    
    # Read the Excel file into a DataFrame
    line_styles = ['-', '--', '-']
    line_colors = sns.color_palette("Set2")
    rescaled_index = df_center.index * time_step
    
    df_10 = df_center.iloc[:, 1:11]
    df_20 = df_center.iloc[:, 1:21]
    df_30 = df_center.iloc[:, 1:31]
    df_40 = df_center.iloc[:, 1:41]
    df_50 = df_center.iloc[:, 1:]
    
    avg_10 = np.mean(df_10, axis=1)
    std_10 = np.std(df_10, axis=1)
    
    avg_20 = np.mean(df_20, axis=1)
    std_20 = np.std(df_20, axis=1)
    
    avg_30 = np.mean(df_30, axis=1)
    std_30 = np.std(df_30, axis=1)
    
    avg_40 = np.mean(df_40, axis=1)
    std_40 = np.std(df_40, axis=1)
    
    avg_50 = np.mean(df_50, axis=1)
    std_50 = np.std(df_50, axis=1)
    
    plt.figure(figsize=(15,10))

    
    
    
    #plt.ylim(-30,120)
    
    #plt.xlim(0,1500)
    #DATAFRAME 1
    plt.plot( rescaled_index, avg_10,  color=sns.color_palette('deep')[0], linewidth=6,label='10 runs')
    plt.fill_between(rescaled_index, avg_10-1*std_10, avg_10+1*std_10,
        alpha=0.3, edgecolor='#3F7F4C', facecolor=sns.color_palette('pastel')[0],
        linewidth=4, linestyle='dotted', antialiased=True)
    #DATAFRAME 2
    plt.plot( rescaled_index, avg_20,  color=sns.color_palette('deep')[2],linewidth=6,label='20 runs')
    plt.fill_between(rescaled_index, avg_20-std_20, avg_20+std_20,
        alpha=0.3, edgecolor='#3F7F4C', facecolor=sns.color_palette('pastel')[2],
        linewidth=4, linestyle='dotted', antialiased=True)
    #DATAFRAME 3
    plt.plot( rescaled_index, avg_30, color=sns.color_palette('deep')[3],linewidth=6,label='30 runs',)
    plt.fill_between(rescaled_index, avg_30-std_30, avg_30+std_30,
        alpha=0.3, edgecolor='#3F7F4C', facecolor=sns.color_palette('pastel')[3],
        linewidth=4, linestyle='dotted', antialiased=True)
    # #DATAFRAME 4
    plt.plot( rescaled_index, avg_40,  color=sns.color_palette('deep')[4],linewidth=6,label='40 runs')
    plt.fill_between(rescaled_index, avg_40-std_40, avg_40+std_40,
        alpha=0.3, edgecolor='#3F7F4C', facecolor=sns.color_palette('pastel')[4],
        linewidth=4, linestyle='dotted', antialiased=True)
    
    # #DATAFRAME 5
    plt.plot( rescaled_index, avg_50,  color=sns.color_palette('deep')[5],linewidth=6,label='50 runs')
    plt.fill_between(rescaled_index, avg_50-std_50, avg_50+std_50,
        alpha=0.3, edgecolor='#3F7F4C', facecolor=sns.color_palette('pastel')[5],
        linewidth=4, linestyle='dotted', antialiased=True)
    
    plt.legend(loc='upper right')#, title='Number of FGs and MTs: 50 and 80')

    
    plt.xlabel("Time, s")
    plt.ylabel("y_pos")
    plt.legend(loc='upper right', title='Spindle orientation models')
    file_name=filename+'_stochasticity_test.pdf'
    save_path = os.path.join(directory_path, file_name)
 
    plt.savefig(save_path)  
    print("average plots are ready")
    
def make_center_plots(directory_path):
    
    time_step, c, excel_file_name, found_endo, found_FE, found_celegans = extract_parameters_name(directory_path, filename)
   
    excel_file_path = os.path.join(directory_path, excel_file_name)
    df_center = pd.read_excel(os.path.join(directory_path, excel_file_name),sheet_name='Y-pos')
    # df_time = pd.read_excel(os.path.join(directory_path, excel_file),sheet_name='Time')
    
    # df_exp=pd.read_excel('./cells_data_normalized_y.xlsx', sheet_name='cell_'+str(c))
    
    # Read the Excel file into a DataFrame
    line_styles = ['-', '--', '-']
    line_colors = sns.color_palette("Set2")
    rescaled_index = df_center.index * time_step
    
    # print(f'df_center={df_center}')
    
    # print(f'rescaled={rescaled_index}')
    
    # # Check the shapes and data types
   
    df_10 = df_center.iloc[:, 1:11]
    
    avg = np.mean(df_10, axis=1)
    std = np.std(df_10, axis=1)
    
    plt.figure(figsize=(15,10))
    column_names = df_center.columns[1:]
    # Iterate through all columns
    
    # print(f'rescaled_index={rescaled_index.shape}')
    # print(f'df_center={df_center[column_names[0]].shape}')
    
    # print(f'rescaled_index={rescaled_index[0]}')
    # print(f'df_center={df_center[column_names[0]][0]}')
    
    for i, column_name in enumerate(df_center.columns[1:]):
        line_style = line_styles[i % len(line_styles)]
        line_color = line_colors[i % len(line_colors)]
        # df_center[column_name] = df_center[column_name].apply(convert_coordinates)

        # Plotting the scatter plot with equal aspect ratio
        # x_values = df_center[column_name].apply(lambda coord: coord[0])
        # y_values = df_center[column_name].apply(lambda coord: coord[1])
        
        plt.plot(rescaled_index, df_center[column_name], label=column_name,linestyle=line_style, color=line_color)

    # plt.plot(df_exp['Time'], 0.1*df_exp['Spindle center y'], label='Experiment', linestyle='--', color='k', linewidth=3)
    
    plt.plot( rescaled_index, avg,  color=sns.color_palette('deep')[0], linewidth=6,label='average')
    plt.fill_between(rescaled_index, avg-std, avg+std,
        alpha=0.3, edgecolor='#3F7F4C', facecolor=sns.color_palette('pastel')[0],
        linewidth=4, linestyle='dotted', antialiased=True)
    
        # Add legend and labels
        #fig = plt.figure(figsize = (15,15))
    #plt.axhline(y=y_final[c-1], color='m', linestyle='--',linewidth=4 ,label='Final position')
    plt.legend(loc='upper left', ncol=2)
    plt.xlabel("Time, s")
    plt.ylabel("Distance, ")
    #plt.xlim(0,180)
    # Show the plot
    plt.ylim(-2,2)
    #plt.ylim(-2.5,2.5)
    plt.axhline(y=0, color='k', linestyle='--')
    # plt.axhline(y=-90, color='k', linestyle='--')
    plt.axhline(y=-1, color='k', linestyle='--')

    # plt.axhline(y=0, color='k', linestyle='--')
    plt.axhline(y=1, color='k', linestyle='--')
    # plt.axhline(y=-90, color='k', linestyle='--')
    #plt.savefig('spindle_9202_-195deg_lateral_sym_t_1_100_50_spindle_9202_-195deg_lateral_sym_t_1_100_50__data.jpeg')
    file_name=filename+'_y-pos.pdf'
    save_path = os.path.join(directory_path, file_name)

    plt.savefig(save_path)  

    print("center plots are ready")
    
def make_angle_plots(directory_path):

    time_step, c, excel_file_name, found_endo, found_FE, found_celegans = extract_parameters_name(directory_path, filename)

    excel_file_path = os.path.join(directory_path, excel_file_name)
    df_angle = pd.read_excel(excel_file_path,sheet_name='Angle')
    # df_exp=pd.read_excel('./cells_data_normalized_y.xlsx', sheet_name='cell_'+str(c))
    df = df_angle.iloc[:, 1:]


    plt.figure(figsize=(15,10))
    column_names = df.columns[1:]

   
    
    avg = np.mean(df, axis=1)
    std = np.std(df, axis=1)
    # ax = plt.subplot(111)
    # ax.set_color_cycle(sns.color_palette("coolwarm_r",num_lines))

    # for i in range(num_lines):
    #     x = np.linspace(0,20,200)
    #     ax.plot(x,np.sin(x)+i)

    # plt.show()

    line_styles = ['-', '--', '-']
    line_colors = sns.color_palette("Set2")
    rescaled_index = df.index * time_step

    # Plot each column
    for i, column in enumerate(df.columns[1:]):
        line_style = line_styles[i % len(line_styles)]
        line_color = line_colors[i % len(line_colors)]
        plt.plot(rescaled_index, df[column], label=column,linestyle=line_style, color=line_color)
    # plt.plot(df_exp['Time'], np.rad2deg(np.asarray(df_exp['Spindle angle'])), label='Experiment', linestyle='--', color='k', linewidth=3)
    # Add legend and labels
    #fig = plt.figure(figsize = (15,15))
    plt.plot( rescaled_index, avg,  color=sns.color_palette('deep')[0], linewidth=6,label='average')
    plt.fill_between(rescaled_index, avg-std, avg+std,
        alpha=0.3, edgecolor='#3F7F4C', facecolor=sns.color_palette('pastel')[0],
        linewidth=4, linestyle='dotted', antialiased=True)
    plt.axhline(y=90, color='k', linestyle='--')
    #plt.axhline(y=final_angle[c-1], color='m', linestyle='--', label='Final angle')
    plt.legend(loc='lower left', ncol=2)
    plt.xlabel("Time, s")
    plt.ylabel("Angle, deg°")
    #plt.xlim(0,180)
    # Show the plot
    plt.ylim(-180,180)
    #plt.ylim(-2.5,2.5)
    # plt.axhline(y=0, color='k', linestyle='--')
    # plt.axhline(y=-90, color='k', linestyle='--')
    # plt.axhline(y=0, color='k', linestyle='--')
    # plt.axhline(y=180, color='k', linestyle='--')
    # plt.axhline(y=-90, color='k', linestyle='--')
    #plt.savefig('spindle_9202_-195deg_lateral_sym_t_1_100_50_spindle_9202_-195deg_lateral_sym_t_1_100_50__data.jpeg')
    file_name=filename+'_angles.pdf'
    save_path = os.path.join(directory_path, file_name)

    plt.savefig(save_path)  
    print("angle plots are ready")
    #plt.show()
def row_average(row):
    return row.mean()
def make_count_plots(directory_path):
        # List all Excel files in the directory
    time_step, c, excel_file_name, found_endo, found_FE, found_celegans = extract_parameters_name(directory_path, filename)

    excel_file_path = os.path.join(directory_path, excel_file_name)

    df_N_pull = pd.read_excel(excel_file_path,sheet_name='N_pull')
    df_N_push = pd.read_excel(excel_file_path,sheet_name='N_push')
   

    df = df_N_pull.iloc[:, 0:]
    df_N_pull = df_N_pull.drop(df.columns[0], axis=1)
    df_N_push = df_N_push.drop(df.columns[0], axis=1)

    # for i in range (len(df)):
    #     df[i]=flip_angles(df[i])

    plt.figure(figsize=(15,10))
    column_names = df.columns[1:]


    # ax = plt.subplot(111)
    # ax.set_color_cycle(sns.color_palette("coolwarm_r",num_lines))

    # for i in range(num_lines):
    #     x = np.linspace(0,20,200)
    #     ax.plot(x,np.sin(x)+i)

    # plt.show()

    line_styles = ['-', '--', '-']
    line_colors = sns.color_palette("Set2")
    rescaled_index = df.index * time_step
    
    pull_avg = df_N_pull.mean(axis=1)
    push_avg = df_N_push.mean(axis=1)
    # Plot each column
    # for i, column in enumerate(df.columns[1:]):
    #     line_style = line_styles[i % len(line_styles)]
    #     line_color = line_colors[i % len(line_colors)]
    #     plt.plot(rescaled_index, df[column], label=column,linestyle=line_style, color=line_color)
    
    plt.plot(rescaled_index, pull_avg, label='N pull' )
    plt.plot(rescaled_index, push_avg, label='N push' )

    # Add legend and labels
    #fig = plt.figure(figsize = (15,15))
    plt.legend(loc='lower left', ncol=2)
    plt.xlabel("Time, s")
    plt.ylabel("N")
    #plt.xlim(0,180)
    # Show the plot
    #plt.ylim(0,df_N_push.to_numpy().max()+df_N_pull.to_numpy().max())
    #plt.ylim(-2.5,2.5)
    plt.axhline(y=0, color='k', linestyle='--')
    # plt.axhline(y=-90, color='k', linestyle='--')
    # plt.axhline(y=90, color='k', linestyle='--')

    # plt.axhline(y=0, color='k', linestyle='--')
    # plt.axhline(y=180, color='k', linestyle='--')
    # plt.axhline(y=-90, color='k', linestyle='--')
    #plt.savefig('spindle_9202_-195deg_lateral_sym_t_1_100_50_spindle_9202_-195deg_lateral_sym_t_1_100_50__data.jpeg')
    file_name=filename+'_counts.pdf'
    save_path = os.path.join(directory_path, file_name)

    plt.savefig(save_path)  
    print("count plots are ready")
    #plt.show()
def make_length_plots(directory_path):

    time_step, c, excel_file_name, found_endo, found_FE, found_celegans = extract_parameters_name(directory_path, filename)

   
    excel_file_path = os.path.join(directory_path, excel_file_name)
    df= pd.read_excel(os.path.join(directory_path, excel_file_name),sheet_name='Length')

    df = df.iloc[:, 0:]

    # for i in range (len(df)):
    #     df[i]=flip_angles(df[i])

    plt.figure(figsize=(15,10))
    column_names = df.columns[1:]


    # ax = plt.subplot(111)
    # ax.set_color_cycle(sns.color_palette("coolwarm_r",num_lines))

    # for i in range(num_lines):
    #     x = np.linspace(0,20,200)
    #     ax.plot(x,np.sin(x)+i)

    # plt.show()

    line_styles = ['-', '--', '-']
    line_colors = sns.color_palette("Set2")
    rescaled_index = df.index * time_step

    # Plot each column
    for i, column in enumerate(df.columns[1:]):
        line_style = line_styles[i % len(line_styles)]
        line_color = line_colors[i % len(line_colors)]
        plt.plot(rescaled_index, df[column], label=column,linestyle=line_style, color=line_color)

    # Add legend and labels
    #fig = plt.figure(figsize = (15,15))
    plt.legend(loc='lower left', ncol=2)
    plt.xlabel("Time, s")
    plt.ylabel("Angle, deg°")
    #plt.xlim(0,180)
    # Show the plot
    #plt.ylim(0,5)
    #plt.ylim(-2.5,2.5)
    # plt.axhline(y=0, color='k', linestyle='--')
    # # plt.axhline(y=-90, color='k', linestyle='--')
    # plt.axhline(y=90, color='k', linestyle='--')

    # # plt.axhline(y=0, color='k', linestyle='--')
    # plt.axhline(y=180, color='k', linestyle='--')
    # plt.axhline(y=-90, color='k', linestyle='--')
    #plt.savefig('spindle_9202_-195deg_lateral_sym_t_1_100_50_spindle_9202_-195deg_lateral_sym_t_1_100_50__data.jpeg')
    file_name=filename+'_mean_length.pdf'
    save_path = os.path.join(directory_path, file_name)

    plt.savefig(save_path)  
    print("length plots are ready")
    #plt.show()


    

# MAIN

directory_path = sys.argv[1]
filename=sys.argv[2]
# make_center_scatter_plots(directory_path)


#time_csv(directory_path)

#make_center_average_plots(directory_path)
make_angle_plots(directory_path)
#make_count_plots(directory_path)
#make_length_plots(directory_path)
make_center_time_plots(directory_path, filename)
#make_center_plots(directory_path)



