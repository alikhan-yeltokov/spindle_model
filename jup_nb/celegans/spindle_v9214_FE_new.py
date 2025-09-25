#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import pandas as pd
import openpyxl
import os, re
from datetime import date
import time, sys
import math
import pickle
import random
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from numpy import linalg as LA
from scipy.stats import norm
import shapely
import shapely.geometry
from shapely import box, LineString, normalize, Polygon
from shapely.geometry import LineString, Polygon, Point
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from scipy.ndimage import gaussian_filter1d

def make_spots(motors,a, b, cell, spindle_poles, frame):
    """
    Generate spots based on the provided parameters.
    """
    
    if(cell_type=='celegans'):

        # n_points=motors
        # # spots=cell[1::int(len(cell)/n_points)]
        
        # angles=get_sequence_celegans(int(0.6*n_points))
        # spots=np.zeros((len(angles),2)) # Right side or posterior has n/2 points

        # a, b, n = 2.5, 1.5, 2.2
        # na = 2 / n  # Exponent adjustment
    
        # spots[:,0] = (np.abs(np.cos(angles)) ** na) * a * sgn(np.cos(angles))
        # spots[:,1] = (np.abs(np.sin(angles)) ** na) * b * sgn(np.sin(angles))
        
        # angles2=get_sequence_celegans(int(0.4*n_points))+np.pi
        n_points=motors
        # spots=cell[1::int(len(cell)/n_points)]
        # gaussian_spaced_points=gauss_points(int(0.4*n_points))

        # Anterior FGs
        gaussian_spaced_points=gauss_points(int(0.4*n_points))
        # angles = np.linspace(np.deg2rad(70),2*np.pi-np.deg2rad(70), int(0.6*n_points))#np.deg2rad(gaussian_spaced_points)-np.pi/2
        angles=np.deg2rad(gaussian_spaced_points)+np.pi
        low_limit= np.deg2rad(90)
        high_limit= 2*np.pi-np.deg2rad(90)
        angles = angles[(angles >= low_limit) & (angles <= high_limit)]
        # angles=get_sequence_celegans(int(0.6*n_points))
        spots=np.zeros((len(angles),2)) # Right side or posterior has n/2 points
        a, b, n = 2.5, 1.5, 2#  2.2
        na = 2 / n  # Exponent adjustment
        spots[:,0] = (np.abs(np.cos(angles)) ** na) * a * sgn(np.cos(angles))
        spots[:,1] = (np.abs(np.sin(angles)) ** na) * b * sgn(np.sin(angles))

        # Posterior FGs
        # angles2=get_sequence_celegans(int(0.4*n_points))
        gaussian_spaced_points_2=gauss_points(int(0.6*n_points))
        # angles2 = np.linspace(np.deg2rad(-60),np.deg2rad(+60), int(0.4*n_points))#np.deg2rad(gaussian_spaced_points_2)-np.pi/2
        angles2=np.deg2rad(gaussian_spaced_points_2)
        low_limit2= np.deg2rad(-90)
        high_limit2= np.deg2rad(+90)
        angles2 = angles2[(angles2 >= low_limit2) & (angles2 <= high_limit2)]
        spots2=np.zeros((len(angles2),2)) # Left side or anterior has n/3 points
        spots2[:,0] = (np.abs(np.cos(angles2)) ** na) * a * sgn(np.cos(angles2))
        spots2[:,1] = (np.abs(np.sin(angles2)) ** na) * b * sgn(np.sin(angles2))
        spots=np.concatenate((spots,spots2))
        
        # if (uniform_status==1):
        #     n_top=n_points//2 # Add an arbitrary number of uniformly distributed FGs in addition to the existing ones
        #     spots_on_top=np.zeros((int(0.5*n_points),2))
        #     angles_top = np.linspace(0, 2*np.pi, int(0.5*n_points)+1)[:-1].copy()
        #     spots_on_top[:,0] = a*np.cos(angles_top)
        #     spots_on_top[:,1] = b*np.sin(angles_top)
        #     spots=np.concatenate((spots,spots_on_top))# uncomment to add spots_on_top lol
       
    elif(cell_type=='endo'):
        
        n_points=int(motors*calculate_perimeter(cell))
        juncs=transform_junctions()
        
        dist1=distance_matrix(np.array([juncs[frame-1,1]]),cell) #find new closest spot
        end_spot_index=np.argmin(dist1[0])

        dist2=distance_matrix(np.array([juncs[frame-1,0]]),cell) #find new closest spot
        start_spot_index=np.argmin(dist2[0])

        # Slice FGs from cell array based on junction start and end position and motor density
        partial_perimeter=calculate_partial_perimeter(cell, start_spot_index, end_spot_index)

        # print(f'partial P={partial_perimeter}')
        n_points_junc=4*n_points
        # print(f'n_points_junc={n_points_junc}')
        spots = slice_array(cell, start_spot_index, end_spot_index, int(n_points_junc))
        
    elif(cell_type=='FE'):

        n_points=motors
        
        # spots=cell[1::int(len(cell)/n_points)]
        
        spots=np.zeros((n_points//2,2)) #make right half of the points first, then reflect and concatenate both sides into one array


        N = n_points//2
        gaussian_spaced_points=gauss_points(N)
        angles = np.deg2rad(gaussian_spaced_points)-np.pi/2
        angles = angles[np.abs(angles) <= np.pi/3]
        # Initialize spots array
        spots = np.zeros((len(angles), 2))
        
        # Compute x and y coordinates
        spots[:, 0] = np.cos(angles )  # x-coordinates
        spots[:, 1] = np.sin(angles )  # y-coordinates
        
        # For squeezed cells to readjust FGs positions because they are based on angles need to readjust angles
        if (a!=b):
            for i in range (len(angles)):
                dumb=delta_theta(a,b,angles[i])
                angles[i]=dumb
                
        # Calculate positions of the spots on the right half of the cell and mirror them to create the left half and concatenate two halves    
        spots[:,0] = a*np.cos(angles-0*np.pi)
        spots[:,1] = b*np.sin(angles-0*np.pi)
        
        other_half=np.zeros((len(angles),2)) #spots on left side
        # other_half[:,0]=-spots[1:-1,0]
        # other_half[:,1]=spots[1:-1,1]
        other_half[:,0]=-spots[:,0]
        other_half[:,1]=spots[:,1]
        # other_half=other_half[1:]
        spots=np.concatenate((spots,other_half))

        # Add extra FGs at the top/apical surface of the cell
        if (config.top_status==1):
            
            n_top=n_points
            # angles=np.linspace(np.pi/6, 5*np.pi/6, n_top)
            # angles=get_sequence_top(n_top) #The option to make top/apical FGs non-uniformly distributed
            angles=get_sequence_celegans(n_top)+np.pi/2
            spots_apical=np.zeros((len(angles),2))
            spots_apical[:,0] = a*np.cos(angles)
            spots_apical[:,1] = b*np.sin(angles)            

            # spots=np.concatenate((spots,spots_apical)) 
            
    return spots#_apical

def transform_junctions():
    # Get spindle size from its contours
    if (c>12):
        spindle_contour=get_contours('./spindles/spindle_'+str(c)+'/Mask_'+str(starts[c-1])+'.jpg')
    else:
        spindle_contour=get_contours('./spindles/spindle_'+str(c)+'/Spindle_'+str(starts[c-1])+'.jpg')
    spindle_input=np.zeros((len(spindle_contour),2))
    for i in range (len(spindle_input)):
        spindle_input[i,0]=spindle_contour[i,0]/rescale[c-1]
        spindle_input[i,1]=spindle_contour[i,1]/rescale[c-1]
    shapely_string = LineString(spindle_input)
    spindle_center=np.array([shapely_string.centroid.x,shapely_string.centroid.y])
    
    # print(f'spindle center -{20.61,31.83} -> {spindle_center}')
    measured_spindle_center=np.array([[20.61,-31.83],
                                       [23.6,-47.2],
                                       [16.15,-52.10],
                                       [0,0],
                                       [14.59,-38.55],
                                       [0,0],
                                       [17.87,-25.61],
                                       [26.79, -28.4],
                                       [14.77, -33.83],
                                       [11.2, -24.55],
                                       [22.29, -55.57],
                                       [16.42, -79.26],
                                       [21.76, -42.18],
                                       [21.33, -49.93],
                                       [27.72, -73.78]])
    # FGs
    file_path = "./junctions/C"+str(c)+"_Mask_Movie_Junctions.txt"
    data = np.loadtxt(file_path)
    data[:, 1] = -data[:, 1]
    juncs = data.reshape(-1, 2, 2)
   
    transf_coef=spindle_center/measured_spindle_center[c-1]

    juncs=juncs*transf_coef-spindle_center

    return juncs

def gauss_points(N, mu=0, sigma=30):
    # Gaussian parameters from the fit (replace with your actual values)
    mu = 0#100.7  # Mean of the Gaussian
    # sigma=30
    sigma = 60#30.3  # Standard deviation of the Gaussian
    # fitted_amplitude, fitted_mean, fitted_stddev = popt
    
    # Generate 100 uniformly spaced points in [0, 1]
    uniform_points = np.linspace(0, 1, N)
    
    # Map the uniform points to [0, 180] using the inverse CDF (quantile function) of the Gaussian
    gaussian_spaced_points = norm.ppf(uniform_points, loc=mu, scale=sigma)
    
    # Clip points to ensure they stay within [0, 180]
    # gaussian_spaced_points = np.clip(gaussian_spaced_points, 0, 180)
    
    # Sort the points (optional, but ensures they are in ascending order)
    gaussian_spaced_points = np.sort(gaussian_spaced_points)
    return gaussian_spaced_points
    
# def make_spots_reserve(FG_density,a, b,cell, spindle_poles, frame):
#     """
#     Generate spots based on the provided parameters.
#     """
#     # uniform_status=1 # Manual switch for adding a set of uniformly distributed FGs in addition to the existing ones
#     n_points=int(FG_density*calculate_perimeter(cell))

    
#     if(cell_type=='celegans'):
        
#         spots=cell[1::int(len(cell)/n_points)]
        
#         # angles=get_sequence_celegans(int(0.3*n_points))
#         # spots=np.zeros((len(angles),2)) # Right side or posterior has n/2 points
#         # spots[:,0] = a*np.cos(angles)
#         # spots[:,1] = b*np.sin(angles)
#         # angles2=get_sequence_celegans(int(0.2*n_points))
#         # spots2=np.zeros((len(angles2),2)) # Left side or anterior has n/3 points
#         # spots2[:,0] = -a*np.cos(angles2)
#         # spots2[:,1] = b*np.sin(angles2)
#         # spots=np.concatenate((spots,spots2))
#         # 
#         # if (uniform_status==1):
#         #     n_top=n_points//2 # Add an arbitrary number of uniformly distributed FGs in addition to the existing ones
#         #     spots_on_top=np.zeros((int(0.5*n_points),2))
#         #     angles_top = np.linspace(0, 2*np.pi, int(0.5*n_points)+1)[:-1].copy()
#         #     spots_on_top[:,0] = a*np.cos(angles_top)
#         #     spots_on_top[:,1] = b*np.sin(angles_top)
#         #     spots=np.concatenate((spots,spots_on_top))# uncomment to add spots_on_top lol
       
#     elif(cell_type=='endo'):
        
#         # For Uniformly distrubuted FGs
#         # spots=cell[1::int(len(cell)/n_points)]

#         # For FGs overlapping with cell-cell junctions
#         center_junc_angle=np.deg2rad(np.linspace(junc_initial_angle[c-1],junc_final_angle[c-1], (ends[c-1]-starts[c-1]))) 
#         center=np.array([0,0])
#         temp=np.deg2rad(junc_spread[c-1])

#         # Calculate junction end and start points on the cell perimeter
#         astral_intersect1, geom_type=intersect_cell(a,b,center_junc_angle[frame-1]+temp,center,cell)
#         dist1=distance_matrix(np.array([astral_intersect1]),cell) #find new closest spot
#         end_spot_index=np.argmin(dist1[0])
        
#         astral_intersect2, geom_type=intersect_cell(a,b,center_junc_angle[frame-1]-temp,center,cell)
#         dist2=distance_matrix(np.array([astral_intersect2]),cell) #find new closest spot
#         start_spot_index=np.argmin(dist2[0])

#         # Slice FGs from cell array based on junction start and end position and motor density
#         partial_perimeter=calculate_partial_perimeter(cell, start_spot_index, end_spot_index)
#         n_points_junc=4*FG_density*partial_perimeter
#         spots = slice_array(cell, start_spot_index, end_spot_index, int(n_points_junc))

#         # spots=np.concatenate((spots,spots2))
#     elif(cell_type=='FE'):
        
#         spots=np.zeros((n_points//2,2)) #make right half of the points first, then reflect and concatenate both sides into one array
#         angles=get_sequence(n_points//2)# get_sequence_celegans(int(0.5*n_points))#

#         # For squeezed cells to readjust FGs positions because they are based on angles need to readjust angles
#         if (a!=b):
#             for i in range (len(angles)):
#                 dumb=delta_theta(a,b,angles[i])
#                 angles[i]=dumb
                
#         # Calculate positions of the spots on the right half of the cell and mirror them to create the left half and concatenate two halves    
#         spots[:,0] = a*np.cos(angles-0*np.pi)
#         spots[:,1] = b*np.sin(angles-0*np.pi)
        
#         other_half=np.zeros((n_points//2,2)) #spots on left side
#         other_half[:,0]=-spots[:,0]
#         other_half[:,1]=spots[:,1]
#         other_half=other_half[1:]
#         spots=np.concatenate((spots[1:],other_half))

#         # Add extra FGs at the top/apical surface of the cell
#         if (config.top_status==1):
            
#             n_top=n_points
#             angles=np.linspace(np.pi/6, 5*np.pi/6, n_top)
#             # angles=get_sequence_top(n_top) The option to make top/apical FGs non-uniformly distributed
#             spots_apical=np.zeros((len(angles),2))
#             spots_apical[:,0] = a*np.cos(angles)
#             spots_apical[:,1] = b*np.sin(angles)            

#             # spots=np.concatenate((spots,spots_apical)) 
            
    # return spots#_apical

def gaussian_spaced_indices(x, y, n_samples, mean=None, std_dev=None):
    """
    Generate indices spaced according to a Gaussian distribution between x and y.
    
    Args:
        x (int): Start index (left bound).
        y (int): End index (right bound).
        n_samples (int): Number of indices to generate.
        mean (float): Center of the Gaussian (default: midpoint between x and y).
        std_dev (float): Spread of the Gaussian (default: (y - x)/6 to cover 99.7% of range).
    
    Returns:
        np.ndarray: Array of indices (float or int).
    """
    if mean is None:
        mean = (x + y) / 2  # Midpoint as default mean
    if std_dev is None:
        std_dev = (y - x) / 6  # Default spread (3σ covers 99.7% of range)

    
    # Generate evenly-spaced quantiles (avoid extremes 0 and 1 to prevent infinite values)
    quantiles = np.linspace(0.001, 0.999, n_samples)
    
    # Map quantiles to Gaussian-distributed indices
    indices = norm.ppf(quantiles, loc=mean, scale=std_dev*0.7)#std_dev)


    # Clip to ensure indices stay within [x, y] (optional, but recommended)
    indices = np.clip(indices, x, y)

    
    return indices.astype(int)  # Convert to integer indices




def slice_array(array, start, end, num_points):
    if num_points < 2:
        raise ValueError("num_points must be at least 2 to form a slice with distinct start and end points")
    
    # Calculate the step size
    step = (end - start) / (num_points - 1)
    
    # Generate the indices for the slice
    indices = np.linspace(start, end, num_points).astype(int)

    # indices = gaussian_indexes_simple(start, end, num_points)
    # indices=gaussian_spaced_indices(start, end, num_points)
    # Use the indices to slice the array
    return array[indices]

def calculate_perimeter(coordinates):
    # Ensure the input is a NumPy array
    coordinates = np.asarray(coordinates)
    
    # Check that the array is not empty and has the correct shape
    if coordinates.size == 0 or coordinates.shape[1] != 2:
        raise ValueError("Input must be a non-empty array with shape (N, 2)")
    
    # Calculate the distance between consecutive points
    distances = np.sqrt(np.sum(np.diff(coordinates, axis=0)**2, axis=1))
    
    # Add the distance between the last and the first point to close the loop
    closing_distance = np.sqrt(np.sum((coordinates[0] - coordinates[-1])**2))
    
    # Sum all distances to get the perimeter
    perimeter = np.sum(distances) + closing_distance
    
    return perimeter

def calculate_partial_perimeter(coordinates, i, j):
    # Ensure the input is a NumPy array
    coordinates = np.asarray(coordinates)
    
    # Check that the array is not empty and has the correct shape
    if coordinates.size == 0 or coordinates.shape[1] != 2:
        raise ValueError("Input must be a non-empty array with shape (N, 2)")
    
    # Normalize indices to ensure i < j
    if i > j:
        i, j = j, i
    
    # Extract the subarray of coordinates between indices i and j (inclusive)
    subarray = coordinates[i:j+1]
    
    # Calculate the distance between consecutive points
    distances = np.sqrt(np.sum(np.diff(subarray, axis=0)**2, axis=1))
    
    # Add the distance between the last and the first point to close the loop
    closing_distance = np.sqrt(np.sum((subarray[0] - subarray[-1])**2))
    
    # Sum all distances to get the perimeter
    perimeter = np.sum(distances) + closing_distance
    
    return perimeter
    
def enhance_cell(cell):
    """
    The spacings between raw cell contour vertices are unequal. This function fixes it. 
    """
    new_cell=[]
    max_interj_dist=0.005
    #max_interj_dist=config.natural_spacing
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

# Log function to non-linearly distribute FGs 
def log_func(x): 
    return 0.5*(np.log(x+1/np.exp(np.pi)))

def log_func_top(x):
    return 0.3*np.log(x+1/np.exp(1))+np.pi/4+0.3

def log_func_top2(x):
    return -0.3*np.log(x+1/np.exp(1))+3*np.pi/4-0.3

def log_func_celegans(x):
    return -np.log(x+1)+np.pi/3

def get_sequence(n_points): #gives you the sequence of angles (theta) that acts as coordinates (x=r cos theta, y=r sin theta) where to put spots or protein machines
    
    sequence=np.zeros((n_points))
    step=((np.exp(np.pi/4))-1/(np.exp(np.pi)))/n_points
    for i in range (n_points):
        sequence[i]=log_func(i*step)
        
    return sequence

def get_sequence_top(n_points): #same but for the apical surface
    
    sequence=np.zeros((n_points)//2+1)
    sequence2=np.zeros((n_points)//2)
    step=(4.67)/(n_points//2)
    for i in range (n_points//2):
        sequence[i]=log_func_top(i*step)
    for i in range (n_points//2):
        sequence2[i]=log_func_top2(i*step)
    dum=sequence2[::-1]
    sequence[(n_points)//2]=0.5*(sequence[-2]+dum[0])
    sequence=np.concatenate((sequence,dum))

    return sequence

def get_sequence_celegans(n_points):
    
    sequence=np.zeros((n_points)//2+1)
    step=(np.exp(np.pi/3)-1)/(n_points//2)
    for i in range (n_points//2+1):
        sequence[i]=log_func_celegans(i*step)
    dum=-sequence[::-1]
    sequence=np.concatenate((sequence,dum[1:]))
    
    return sequence


def point_in_polygon(point, polygon):
    """
    For checking if spindle poles leave the cell boundary.
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= x_inters:
                    inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def intersect_cell(a,b, angle,start_point, cell):
    """
    Deploys shapely module to find an intersection between astral MTs and cell cortex
    """
    # NumPy array
    end_point=start_point+4*np.array([np.cos(angle),np.sin(angle)])
    numpy_line_coords = np.array([start_point,end_point])
    # Convert to Shapely LineString
    shapely_line = shapely.geometry.LineString(numpy_line_coords)
    shapely_string = LineString(cell)
    inter=shapely.intersection(shapely_string, shapely_line)

    
    if(inter.geom_type=='Point'):
        return np.array([inter.x,inter.y])  , inter.geom_type
    elif(inter.geom_type=='MultiPoint'):
        points = [p for p in inter.geoms]
        dist=[]
        for i in range (len(points)):
            dist.append(LA.norm([start_point[0]-points[i].x,start_point[1]-points[i].y]))
        return np.array([points[np.argsort(dist)[0]].x,points[np.argsort(dist)[0]].y]),  inter.geom_type
    else:
        #print(f'intersection type:{inter.geom_type}, start={start_point}, end={end_point}')
        #print(np.asanyarray(inter))
        intersect=intersect_cell_old(a,b, angle,start_point, cell)
        return intersect, inter.geom_type #start_point+0.8*MT_max_length*np.array([np.cos(angle),np.sin(angle)])

def intersect_cell_old(a,b,angle,C,cell):
    """
    When astral MT's is shrinking but spindle moves towards the cortex there are 3 cases:
    1. End is outside
    2. End is inside
    This function basically tells if the end is inside keep it there, if its outside, find the cortex intersect and take the end there.
    """
    L_cell=2*np.pi/config.number_of_sides
    D=C
    L_min=min(a,b)#
    j=0
    dist=distance_matrix(np.array([C]),cell)[0]
    while (L_min>L_cell):# and (C[0]/a)**2+(C[1]/b)**2<1.05):
        if (j>10):
            break
        dist=distance_matrix(np.array([C]),cell)[0]
        L_min=np.min(dist)
        C=C+L_min*np.array([np.cos(angle),np.sin(angle)])
        j=j+1
    
    D=C+2*np.array([np.cos(angle),np.sin(angle)])
    k = 2
    result = np.argpartition(dist, k)
    cell_small1,cell_small2=result[:k]
    small_dist1,small_dist2=dist[result[:k]]
    A=cell[cell_small1]
    B=cell[cell_small2]
    # Line AB represented as a1x + b1y = c1
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1*(A[0]) + b1*(A[1])
    # Line CD represented as a2x + b2y = c2
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2*(C[0]) + b2*(C[1])
    determinant = a1*b2 - a2*b1
    if (determinant == 0):
        return D
    else:
        x = (b2*c1 - b1*c2)/determinant
        y = (a1*c2 - a2*c1)/determinant
        intersect=np.array([x, y])
        return intersect

        
def distance_matrix(xz1, xz2):
    mutx1 = np.outer(np.ones(len(xz2[:,0])), xz1[:,0])
    mutz1 = np.outer(np.ones(len(xz2[:,0])), xz1[:,1])
    mutx2 = np.outer(xz2[:,0], np.ones(len(xz1[:,0])))
    mutz2 = np.outer(xz2[:,1], np.ones(len(xz1[:,0])))
    return np.sqrt((mutx1-mutx2)**2+(mutz1-mutz2)**2).transpose()

def delta_theta(a,b,theta):

    if a*b ==3:# or theta <0 or theta>2*np.pi:
        #return 'What the fuck....'
        c=1+3
    else:
        if (theta<0):
            theta=theta+2*np.pi
        if (theta>2*np.pi):
            theta=theta-2*np.pi
        if a>=b:
            if theta<=np.pi/2:
                return -np.arccos((a*np.cos(theta)**2+b*np.sin(theta)**2) / np.sqrt((a*np.cos(theta))**2+(b*np.sin(theta))**2))+theta
            elif theta<=np.pi:
                return np.arccos((a*np.cos(theta)**2+b*np.sin(theta)**2) / np.sqrt((a*np.cos(theta))**2+(b*np.sin(theta))**2))+theta
            elif theta<=3*np.pi/2:
                return -np.arccos((a*np.cos(theta)**2+b*np.sin(theta)**2) / np.sqrt((a*np.cos(theta))**2+(b*np.sin(theta))**2))+theta
            else:
                return np.arccos((a*np.cos(theta)**2+b*np.sin(theta)**2) / np.sqrt((a*np.cos(theta))**2+(b*np.sin(theta))**2))+theta
        else:
            if theta<=np.pi/2:
                return np.arccos((a*np.cos(theta)**2+b*np.sin(theta)**2) / np.sqrt((a*np.cos(theta))**2+(b*np.sin(theta))**2))+theta
            elif theta<=np.pi:
                return -np.arccos((a*np.cos(theta)**2+b*np.sin(theta)**2) / np.sqrt((a*np.cos(theta))**2+(b*np.sin(theta))**2))+theta
            elif theta<=3*np.pi/2:
                return np.arccos((a*np.cos(theta)**2+b*np.sin(theta)**2) / np.sqrt((a*np.cos(theta))**2+(b*np.sin(theta))**2))+theta
            else:
                return -np.arccos((a*np.cos(theta)**2+b*np.sin(theta)**2) / np.sqrt((a*np.cos(theta))**2+(b*np.sin(theta))**2))+theta


def get_contours(image_path):

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise and detail
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edge-detected image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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

def smoothen_cell(cell):
    # Example: Replace this with your actual contour coordinates (N, 2)
    contours = cell  # Replace with actual contour coordinates
    
    # Separate x and y coordinates
    x = contours[:, 0]
    y = contours[:, 1]
    
    # Apply Gaussian smoothing to both x and y coordinates
    sigma = 10  # Adjust sigma for more or less smoothing
    x_smooth = gaussian_filter1d(x, sigma=sigma)
    y_smooth = gaussian_filter1d(y, sigma=sigma)
    
    # Combine the smoothed x and y coordinates back into an array
    smoothed_contours = np.column_stack((x_smooth, y_smooth))
    
    # Plot the original and smoothed contours for comparison
    # plt.figure(figsize=(6, 6))
    # plt.plot(x, y, label='Original Contours', color='red', linewidth=2)
    # plt.plot(x_smooth, y_smooth, label='Smoothed Contours', color='blue', linewidth=2)
    # plt.legend()
    # plt.gca().set_aspect('equal')
    # plt.title('Original vs Smoothed Contours')
    # plt.show()
    return smoothed_contours
def close_contours(contours):
    """
    Turn a contour array into a closed loop by adding points between the start and end
    such that the distance between points is less than or equal to `min_spacing`.

    Parameters:
    -----------
    contours : np.ndarray
        The array of contour coordinates with shape (N, 2).
    min_spacing : float
        The minimal spacing between the points to be added.

    Returns:
    --------
    closed_contours : np.ndarray
        The array of contour coordinates with the newly added points to close the loop.
    """
    # Extract the start and end points
    end_point = contours[0]
    start_point = contours[-1]

 

    # Calculate the Euclidean distance between the start and end points
    distance = np.linalg.norm(end_point - start_point)
    
    min_spacing=0.005
    # Calculate how many points to add based on the minimal spacing
    num_points_to_add = int(np.ceil(distance / min_spacing))
 

    # Create the interpolation for the required number of points
    x_new = np.linspace(start_point[0], end_point[0], num_points_to_add + 2)  # +2 includes start and end points
    y_new = np.linspace(start_point[1], end_point[1], num_points_to_add + 2)

    # Remove the first and last points from the new points to avoid duplication with original points
    x_new = x_new[1:-1]
    y_new = y_new[1:-1]

    # Combine the new points into a new array
    new_points = np.column_stack((x_new, y_new))

    # Combine original contours with new points to form a closed contour
    closed_contours = np.vstack([contours, new_points])

    # Close the loop by adding the first point at the end
    # closed_contours = np.vstack([closed_contours, closed_contours[0]])

   
    
    return closed_contours

def sgn(x):
    """Sign function using NumPy."""
    return np.where(x > 0, 1, -1)  # Vectorized version





def make_cell(params):
    """
    Returns an array (N,2) of x,y coordinates of vertices of a polygon representing the cell
    """
    a=params[0][0]
    r=params[0][2] #spindle length=2*r
    FG_density=int(params[1])
    n_astro=int(params[2])
    b=params[0][1]
    spindle_angle=params[3]#np.random.uniform(0,np.pi)
    theta = np.linspace(0, 2*np.pi, config.number_of_sides+1)[:-1].copy()
    
    if (cell_type=='endo'):

        # Read the image
        image_path='./cells/cell_'+str(c)+'/Mask_'+str(starts[c-1])+'.jpg'
        # Get the raw contours and rescale them
        all_points=get_contours(image_path)
        cell_input=np.zeros((len(all_points),2))
        for i in range (len(cell_input)):
            cell_input[i,0]=all_points[i,0]/rescale[c-1]
            cell_input[i,1]=all_points[i,1]/rescale[c-1]
        # Yassify raw cell contours
        cell=enhance_cell(cell_input)

        # Get spindle size from its contours
        if (c>12):
            spindle_contour=get_contours('./spindles/spindle_'+str(c)+'/Mask_'+str(starts[c-1])+'.jpg')
        else:
            spindle_contour=get_contours('./spindles/spindle_'+str(c)+'/Spindle_'+str(starts[c-1])+'.jpg')
        spindle_input=np.zeros((len(spindle_contour),2))
        for i in range (len(spindle_input)):
            spindle_input[i,0]=spindle_contour[i,0]/rescale[c-1]
            spindle_input[i,1]=spindle_contour[i,1]/rescale[c-1]
   
        shapely_string = LineString(spindle_input)
        spindle_center=np.array([shapely_string.centroid.x,shapely_string.centroid.y])
    
    elif (cell_type  =='celegans'):
        a, b, n = 2.5, 1.5, 2#.2
        na = 2 / n  # Exponent adjustment
    
        cell=np.zeros((config.number_of_sides,2))
        cell[:,0] = (np.abs(np.cos(theta)) ** na) * a * sgn(np.cos(theta))
        cell[:,1] = (np.abs(np.sin(theta)) ** na) * b * sgn(np.sin(theta))
    else:
        cell=np.zeros((config.number_of_sides,2))
        cell[:,0] = a*np.cos(theta)
        cell[:,1] = b*np.sin(theta)

    # Making the spindle
    spindle_poles=np.zeros((2,2))

    # Choosing displacement depending on the cell type for debugging purposes
    if (cell_type=='celegans'):
        displacement=np.array([0,0])
    elif(cell_type=='endo'):
        displacement=0*spindle_center
    elif(cell_type=='FE'):
        displacement=0*np.array([-0.5,-0])

    spindle_poles[0]=[r*np.cos(spindle_angle)+displacement[0],r*np.sin(spindle_angle)+displacement[1]]#-0*(np.min(centered_cell[:,1]))]
    spindle_poles[1]=[r*np.cos(spindle_angle+np.pi)+displacement[0],r*np.sin(spindle_angle+np.pi)+displacement[1]]#-0*(np.min(centered_cell[:,1]))]

    if (cell_type=='endo'):
        cell=cell-spindle_center
        cell=smoothen_cell(cell)
        cell=close_contours(cell)
        
    #Sometimes the requested spindle angle doesnt fit so adjustment to the angle is made
    i=0
    while (point_in_polygon(spindle_poles[0], cell)==False or point_in_polygon(spindle_poles[1], cell)==False):
        spindle_angle=spindle_angle+np.pi/12
        spindle_poles[0]=[r*np.cos(spindle_angle)+displacement[0],r*np.sin(spindle_angle)+displacement[1]]
        spindle_poles[1]=[r*np.cos(spindle_angle+np.pi)+displacement[0],r*np.sin(spindle_angle+np.pi)+displacement[1]]
        i=i+1
        if(i>12):
            break
            
    # Making motors
    spots=make_spots(FG_density,a, b,cell, spindle_poles, 1)
    
    # Making astral microtubules
    astral_MTs, astral_angles, state,which_push,which_bind,free_spots,astral_which_spot, orig_length, df_list2=make_astral_MTs(params,cell,spindle_poles, spindle_angle,spots)
    
    return cell,astral_MTs,astral_angles,state,spindle_poles,spindle_angle,spots,which_push,which_bind,free_spots,astral_which_spot,orig_length,df_list2

def update_cell(params, t_time):
    """
    Updating cell shape according to live movies
    """
    FG_density=int(params[1])
    # Read the image
    image_path='./cells/cell_'+str(c)+'/Mask_'+str(int(starts[c-1]+t_time/frame_rates[c-1]))+'.jpg'
    # Get the raw contours and rescale them
    all_points=get_contours(image_path)

    cell_input=np.zeros((len(all_points),2))
    for i in range (len(cell_input)):
        cell_input[i,0]=all_points[i,0]/rescale[c-1]
        cell_input[i,1]=all_points[i,1]/rescale[c-1]
    cell=enhance_cell(cell_input)


    # Get spindle size from its contours
    if (c>12):
        spindle_contour=get_contours('./spindles/spindle_'+str(c)+'/Mask_'+str(starts[c-1])+'.jpg')
    else:
        spindle_contour=get_contours('./spindles/spindle_'+str(c)+'/Spindle_'+str(starts[c-1])+'.jpg')
    spindle_input=np.zeros((len(spindle_contour),2))
    for i in range (len(spindle_input)):
        spindle_input[i,0]=spindle_contour[i,0]/rescale[c-1]
        spindle_input[i,1]=spindle_contour[i,1]/rescale[c-1]
    spindle_length=max(spindle_input[:,1])-min(spindle_input[:,1])
    r=spindle_length/2
    shapely_string = LineString(spindle_input)
    spindle_center=np.array([shapely_string.centroid.x,shapely_string.centroid.y])

    cell=cell-spindle_center
    cell=smoothen_cell(cell)
    cell=close_contours(cell)
    return cell
    
def transfer_free_spots(new_length, b1):
    # Create array a1 of specified shape filled with zeroes
    
    a1 = np.zeros(new_length)

    # Determine the length of the smaller array
    min_length = min(new_length, len(b1))
    
    # Transfer elements from b1 to a1
    a1[:min_length] = b1[:min_length]
    
    return a1
def transfer_astro_which(new_length, b1):
    # Create array a1 of specified shape filled with zeroes
    
    a1 = np.zeros((new_length, 2))

    # Determine the length of the smaller array
    min_length = min(new_length, len(b1))
    
    # Transfer elements from b1 to a1
    a1[:min_length] = b1[:min_length]

    return a1
    
def bounded_normal_random(mean, stdev):
    while True:
        num = random.gauss(mean, stdev)  # Generate from normal distribution
        if 0 <= num <= 1:  # Check if within bounds
            return num
    
def make_astral_MTs(params,cell, spindle_poles, spindle_angle,spots):
    
    a=params[0][0]
    r=params[0][2] #spindle length=2*r
    FG_density=int(params[1])
    n_astro=int(params[2])
    b=params[0][1]
    
    # Initializing all arrays to keep MTs data
    
    # state=np.ones((2,(n_astro))) #two states: growing=1, shrinking=-1
    np.random.seed(42)
    free_spots=np.zeros((len(spots)))#array 0 if spot is free, 1 is taken
    astral_which_spot=np.zeros((len(spots),2))-1 #which astro occupies given spot by index (astral_which_spot[5]=[1,20] means the fifth spot is occupied by (1,20)), i dont remember where -1 is coming from, its coming from not confusing value 0,0 (-1,-1) with actual MT [0,0]
    which_bind=np.zeros((2,int(n_astro))) #astral MTs have binded
    which_push=np.zeros((2,int(n_astro))) #astral MT that push
    astral_MTs=np.zeros((2,int(n_astro),config.discr,2))
    
    # Astral MTs angles
    astral_angles=np.zeros((2,n_astro))
    astral_angles[0]=np.linspace(spindle_angle-spread/2,spindle_angle+spread/2, n_astro) #predefined angles
    astral_angles[1]=np.linspace(spindle_angle+np.pi-spread/2,spindle_angle+np.pi+spread/2, n_astro)
    
    # orig_length=astral_initial_length*np.ones((5,n_astro)) #to keep track of astral MTs lengthes in case of elongation
    
    # orig_length=np.zeros((2,n_astro))+0.05 
    # Generate the random array
    # orig_length = abs(np.random.normal(loc=mean, scale=stdev, size=(2, n_astro)))
    # orig_length=1*np.ones((2,n_astro)) #to keep track of astral MTs lengthes in case of elongation
    # max_length=np.zeros((2,n_astro))
    # for i in range (2):
    #     for j in range (n_astro):
    #         length_tc, _ =intersect_cell(a,b,astral_angles[i,j],spindle_poles[i],cell)
    #         max_length[i,j]=LA.norm(length_tc-spindle_poles[i])
    if (config.length_MTs=='gamma'):
        orig_length = np.random.gamma(AL, scale, size=(2, n_astro))#*max_length
    else:
        orig_length=AL*np.ones((2,n_astro))
    # if (config.length_MTs=='gamma'):
    #     orig_length = np.random.gamma(AL, scale, size=(2, n_astro))
    # else:
    #     orig_length=AL*np.ones((2,n_astro))

    if (config.state_MTs=='random'):
        state = np.random.choice([-1, 1], size=(2, n_astro), p=[0.5, 0.5])
    else:
        state = np.ones((2, n_astro))
    df_list2=[]  
      
    for i in range (2):
        for j in range (n_astro):
            
            # length_tc, _ =intersect_cell(a,b,astral_angles[i,j],spindle_poles[i],cell) 
            # orig_length[i,j]=LA.norm(length_tc-spindle_poles[i])*bounded_normal_random(mean, stdev)
            
            # print(f'astral=[{i,j}]')
            end=grow_astralMT(a,b,astral_angles[i,j],spindle_poles[i],cell,orig_length[i,j])
            # state[i,j]=1 if (np.random.rand()>0.5) else -1
            astral_MTs[i,j,:,0]=np.linspace(spindle_poles[i,0],end[0],config.discr)
            astral_MTs[i,j,:,1]=np.linspace(spindle_poles[i,1],end[1],config.discr)

            
            # which_bind[i,j],free_spots,astral_which_spot=check_bind(i, j,astral_MTs[i,j,-1],spindle_poles,spots,free_spots,astral_which_spot) #used to be check_bind_init
            # if (which_bind[i,j]==1):
            #     which_push[i,j]=0
            #     state[i,j]=-1
            # # state[i,j]=1 if which_bind[i,j]==0 else -1
            # which_push[i,j]=check_push(a,b,astral_MTs[i,j,-1],state[i,j],astral_angles[i,j],spindle_poles[i],cell)
            # which_push[i,j]=check_push_junc(a,b,astral_MTs[i,j,-1],which_bind[i,j],state[i,j],astral_angles[i,j],spindle_poles[i],cell, spots)
            
            which_bind[i,j], which_push[i,j], state[i,j], free_spots,astral_which_spot=check_push_bind_init(i, j, astral_MTs[i,j,-1], spindle_poles, spots, free_spots, astral_which_spot, state[i,j], astral_angles[i,j], cell)

            orig_length[i,j]=LA.norm(astral_MTs[i,j,-1]-spindle_poles[i]) 

            # List to keep MTs data
            df_list=[]
            df_list.append(-1)
            df_list.append([i,j])
            df_list.append('Not dead')
            df_list.append('Neither R or C')
            df_list.append('NOT TOO SHORT')
            df_list.append(which_bind[i,j])
            df_list.append(which_push[i,j])
            df_list.append(state[i,j])
            df_list.append(0)
            df_list.append(LA.norm(astral_MTs[i,j,-1]-spindle_poles[i]))
            df_list.append(orig_length[i,j])
            df_list.append(math.sqrt(astral_MTs[i,j,-1,0]*astral_MTs[i,j,-1,0]+astral_MTs[i,j,-1,1]*astral_MTs[i,j,-1,1]))
            df_list.append(math.degrees((astral_angles[i,j])))
            df_list2.append(df_list)
            
    return astral_MTs, astral_angles, state,which_push,which_bind,free_spots,astral_which_spot, orig_length, df_list2


def find_force(astral_MTs, spindle_poles, which_bind, which_push, v_c):
    # Initialize force vectors
    push_vecs = np.zeros((2, len(astral_MTs[0]), 2))
    pull_vecs = np.zeros((2, len(astral_MTs[0]), 2))

    # Debug: Print input shapes and values
    # print(f"Debug: astral_MTs shape = {astral_MTs.shape}")
    # print(f"Debug: spindle_poles = {spindle_poles}")
    # print(f"Debug: which_bind = {which_bind}")
    # print(f"Debug: which_push = {which_push}")
    # print(f"Debug: v_c = {v_c}")

    # Nested loop to go through each MT
    for i in range(2):
        for j in range(len(astral_MTs[0])):
            # Debug: Print current MT and pole indices
            # print(f"\nDebug: Processing MT {j} on pole {i}")

            # Each force vector is collinear with astral MT body alignment
            vec = np.multiply(astral_MTs[i, j, -1] - spindle_poles[i], 1 / LA.norm(astral_MTs[i, j, -1] - spindle_poles[i]))
            
            # Debug: Print the computed vector
            # print(f"Debug: vec = {vec}")

            # Case 1: MT is being pulled on
            if which_bind[i, j] == 1 and which_push[i, j] == 0:
                # Debug: Print case 1 condition
                # print("Debug: Case 1 - MT is being pulled")

                if config.force_velocity == 1:
                    a_vec = (astral_MTs[i, j, -1] - astral_MTs[i, j, 0]) / LA.norm(astral_MTs[i, j, -1] - astral_MTs[i, j, 0])
                    force = config.pull * (1 - np.dot(v_c, a_vec) / config.v_0)
                    
                    # Debug: Print force calculation details
                    # print(f"Debug: a_vec = {a_vec}")
                    # print(f"Debug: np.dot(v_c, a_vec) = {np.dot(v_c, a_vec)}")
                    # print(f"Debug: force = {force}")
                else:
                    force = config.pull
                    # Debug: Print constant force
                    # print(f"Debug: Constant force (force_velocity=0) = {force}")

                pull_vecs[i, j] = np.multiply(vec, force)
                # Debug: Print updated pull_vecs
                # print(f"Debug: pull_vecs[{i}, {j}] = {pull_vecs[i, j]}")

            # Case 2: MT is pushing against the cortex
            elif which_bind[i, j] == 0 and which_push[i, j] == 1:
                # Debug: Print case 2 condition
                # print("Debug: Case 2 - MT is pushing")

                # sp_angle=np.rad2deg(get_spindle_angle(spindle_poles))
                # # print(f"Debug: Acute angle between spindle poles = {sp_angle} degrees")
                # # print(f'sp_angle<cutoff_angle={sp_angle}<{cutoff_angle}')
                
                # if (sp_angle<cutoff_angle or sp_angle>180-cutoff_angle):
                #     print(f'cutoff pushing satisfied')
                #     push=0
                # else:
                #     print(f'cutoff pushing NOT satisfied')
                push=config.push
                astral_len = LA.norm(astral_MTs[i, j, -1] - spindle_poles[i])
                force = -min(push, config.EI * np.pi * np.pi / astral_len / astral_len)
                
                # Debug: Print astral_len and force calculation
                # print(f"Debug: astral_len = {astral_len}")
                # print(f"Debug: force = {force}")

                push_vecs[i, j] = np.multiply(vec, force)
                # Debug: Print updated push_vecs
                # print(f"Debug: push_vecs[{i}, {j}] = {push_vecs[i, j]}")

            # Case 3: any other MT exerts 0 force
            # else:
                # Debug: Print case 3 condition
                # print("Debug: Case 3 - MT exerts 0 force")

    # Debug: Print final force vectors
    # print("\nDebug: Final push_vecs =")
    # print(push_vecs)
    # print("Debug: Final pull_vecs =")
    # print(pull_vecs)

    return push_vecs, pull_vecs



def check_push_bind_init(i, j, astral, spindle_poles, spots, free_spots, astral_which_spot, state, astral_angles, cell):
    """
    Combined function that checks both binding and pushing conditions for an astral MT.
    Reordered workflow: first checks proximity to cell end, then binding distance.
    
    Parameters:
        i, j: MT indices
        astral: Astral MT tip coordinates
        spindle_poles: Spindle poles coordinates
        spots: FG spot coordinates
        free_spots: Array indicating available spots
        astral_which_spot: Tracking array
        state: MT state (1=growth, -1=shrink)
        astral_angles: MT angles
        cell: Cell geometry
    
    Returns:
        bind: 1 if bound, 0 otherwise
        push: 1 if pushing, 0 otherwise
        free_spots: Updated free spots array
        astral_which_spot: Updated tracking array
    """
    # Initialize outputs
    bind = 0
    push = 0

    dist = distance_matrix(np.array([astral]), spots)
    #check if the MT is close enough to a FG spot
    # and if the spot is free
    if (np.min(dist[0])<=config.max_interact_dist and free_spots[np.argmin(dist[0])]==0):
    
        free_spots[np.argmin(dist[0])] = 1
        bind = 1
        state = -1
        astral_which_spot[np.argmin(dist[0]), 0] = i
        astral_which_spot[np.argmin(dist[0]), 1] = j
    else:
        end, _ = intersect_cell(a, b, astral_angles, spindle_poles[i], cell)
        # Check if the astral MT is close enough to the cortex
        if (math.isclose(abs(LA.norm(end) - LA.norm(astral)), 0, abs_tol=config.push_dist)):
            # If not binding, check if the MT is pushing
            push = 1
            state = 1
        
    return bind, push, state, free_spots, astral_which_spot
    # """
    # Find the distance between the astral MT tip and the closest FG. If they are close enough, they bind together.
    # """
    # dist=distance_matrix(np.array([astral]),spots)
    # if (np.min(dist[0])<=config.max_interact_dist and free_spots[np.argmin(dist[0])]==0):
    #     free_spots[np.argmin(dist[0])]=1
    #     bind=1
    #     astral_which_spot[np.argmin(dist[0]),0]=i
    #     astral_which_spot[np.argmin(dist[0]),1]=j
    # else:
    #     bind=0
    # return bind,free_spots,astral_which_spot

# def check_push_bind(i, j, astral, spindle_poles, spots, free_spots, astral_which_spot, state, astral_angles, cell):
#     """
#     Combined function that checks both binding and pushing conditions for an astral MT.
#     Reordered workflow: first checks proximity to cell end, then binding distance.
    
#     Parameters:
#         i, j: MT indices
#         astral: Astral MT tip coordinates
#         spindle_poles: Spindle poles coordinates
#         spots: FG spot coordinates
#         free_spots: Array indicating available spots
#         astral_which_spot: Tracking array
#         state: MT state (1=growth, -1=shrink)
#         astral_angles: MT angles
#         cell: Cell geometry
    
#     Returns:
#         bind: 1 if bound, 0 otherwise
#         push: 1 if pushing, 0 otherwise
#         free_spots: Updated free spots array
#         astral_which_spot: Updated tracking array
#     """
#     # Initialize outputs
#     bind = 0
#     push = 0
    
#     # First check pushing condition (proximity to cell end)
#     end, _ = intersect_cell(a, b, astral_angles, spindle_poles[i], cell)
    
#     # Check if the astral MT is close enough to the cortex
#     if (math.isclose(abs(LA.norm(end) - LA.norm(astral)), 0, abs_tol=config.min_connect_dist)):
        

#         dist = distance_matrix(np.array([astral]), spots)
#         #check if the MT is close enough to a FG spot
#         # and if the spot is free
#         if (np.min(dist[0])<=config.max_interact_dist and free_spots[np.argmin(dist[0])]==0 and random.uniform(0, 1)<=prob_dyn_bind):
       
#             free_spots[np.argmin(dist[0])] = 1
#             bind = 1
#             state = -1
#             astral_which_spot[np.argmin(dist[0]), 0] = i
#             astral_which_spot[np.argmin(dist[0]), 1] = j
#         else:
#             # If not binding, check if the MT is pushing
#             push = 1
#             state = 1
        
#     return bind, push, state, free_spots, astral_which_spot
def check_push_bind(i, j, astral, spindle_poles, spots, free_spots, astral_which_spot, state, astral_angles, cell):
    """
    Combined function that checks both binding and pushing conditions for an astral MT.
    Reordered workflow: first checks proximity to cell end, then binding distance.
    
    Parameters:
        i, j: MT indices
        astral: Astral MT tip coordinates
        spindle_poles: Spindle poles coordinates
        spots: FG spot coordinates
        free_spots: Array indicating available spots
        astral_which_spot: Tracking array
        state: MT state (1=growth, -1=shrink)
        astral_angles: MT angles
        cell: Cell geometry
    
    Returns:
        bind: 1 if bound, 0 otherwise
        push: 1 if pushing, 0 otherwise
        free_spots: Updated free spots array
        astral_which_spot: Updated tracking array
    """
    # Initialize outputs
    bind = 0
    push = 0

    dist = distance_matrix(np.array([astral]), spots)
    #check if the MT is close enough to a FG spot
    # and if the spot is free
    if (np.min(dist[0])<=config.max_interact_dist and free_spots[np.argmin(dist[0])]==0 and random.uniform(0, 1)<=prob_dyn_bind):
    
        free_spots[np.argmin(dist[0])] = 1
        bind = 1
        state = -1
        astral_which_spot[np.argmin(dist[0]), 0] = i
        astral_which_spot[np.argmin(dist[0]), 1] = j
    else:
        end, _ = intersect_cell(a, b, astral_angles, spindle_poles[i], cell)
        # Check if the astral MT is close enough to the cortex.
        # print(f'astral to cortex= {abs(LA.norm(end) - LA.norm(astral))}')
        # if (math.isclose(abs(LA.norm(end) - LA.norm(astral)), 0, abs_tol=config.push_dist)):
        if (math.isclose(abs(LA.norm(end-astral)), 0, abs_tol=config.push_dist)):
            # If not binding, check if the MT is pushing
            push = 1
            state = 1
        
    return bind, push, state, free_spots, astral_which_spot
# def check_push_bind(i, j, astral, spindle_poles, spots, free_spots, astral_which_spot, state, astral_angles, cell):
#     """
#     Combined function that checks both binding and pushing conditions for an astral MT.
#     Reordered workflow: first checks proximity to cell end, then binding distance.
    
#     Parameters:
#         i, j: MT indices
#         astral: Astral MT tip coordinates
#         spindle_poles: Spindle poles coordinates
#         spots: FG spot coordinates
#         free_spots: Array indicating available spots
#         astral_which_spot: Tracking array
#         state: MT state (1=growth, -1=shrink)
#         astral_angles: MT angles
#         cell: Cell geometry
    
#     Returns:
#         bind: 1 if bound, 0 otherwise
#         push: 1 if pushing, 0 otherwise
#         free_spots: Updated free spots array
#         astral_which_spot: Updated tracking array
#     """
#     # Initialize outputs
#     bind = 0
#     push = 0

#     dist = distance_matrix(np.array([astral]), spots)
#     #check if the MT is close enough to a FG spot
#     # and if the spot is free
#     if (np.min(dist[0])<=config.max_interact_dist and free_spots[np.argmin(dist[0])]==0 and random.uniform(0, 1)<=prob_dyn_bind):
    
#         free_spots[np.argmin(dist[0])] = 1
#         bind = 1
#         state = -1
#         astral_which_spot[np.argmin(dist[0]), 0] = i
#         astral_which_spot[np.argmin(dist[0]), 1] = j
#     else:
#         end, _ = intersect_cell(a, b, astral_angles, spindle_poles[i], cell)
#         # Check if the astral MT is close enough to the cortex
#         if (math.isclose(abs(LA.norm(end) - LA.norm(astral)), 0, abs_tol=config.push_dist)):
#             # If not binding, check if the MT is pushing
#             push = 1
#             state = 1
        
#     return bind, push, state, free_spots, astral_which_spot

# def check_push_bind(i, j, astral, spindle_poles, spots, free_spots, astral_which_spot, state, astral_angles, cell):
#     """
#     Combined function that checks both binding and pushing conditions for an astral MT.
    
#     Parameters:
#         i, j: MT indices
#         astral: Astral MT tip coordinates
#         spindle_poles: Spindle poles coordinates
#         spots: FG spot coordinates
#         free_spots: Array indicating available spots
#         astral_which_spot: Tracking array
#         state: MT state (1=growth, -1=shrink)
#         astral_angles: MT angles
#         cell: Cell geometry
    
#     Returns:
#         bind: 1 if bound, 0 otherwise
#         push: 1 if pushing, 0 otherwise
#         free_spots: Updated free spots array
#         astral_which_spot: Updated tracking array
#     """
    
#     # Initialize outputs
#     bind = 0
#     push = 0
    
#     # Check binding condition
#     dist = distance_matrix(np.array([astral]), spots)
    
#     if (np.min(dist[0])<=config.max_interact_dist and free_spots[np.argmin(dist[0])]==0 and random.uniform(0, 1)<=prob_dyn_bind):
        
#         free_spots[np.argmin(dist[0])]=1
#         bind=1
#         state=-1
#         astral_which_spot[np.argmin(dist[0]),0]=i
#         astral_which_spot[np.argmin(dist[0]),1]=j
        
#         # print(f'CHECK BIND->{bind>0}')
    
#     # Check pushing condition (only if not bound)
#     if bind == 0:
        
#         end, _ = intersect_cell(a,b,astral_angles,spindle_poles[i],cell)
        
#         # print(f'close enough? {math.isclose(abs(LA.norm(end) - LA.norm(astral)), 0, abs_tol=1e-15)}')
        
#         # print(f'diff={LA.norm(end) - LA.norm(astral)}')
#         # print(f'state={state}')
        
#         if (math.isclose(abs(LA.norm(end) - LA.norm(astral)), 0, abs_tol=1e-15) and state == 1):
            
#             # print(f'close enough? {math.isclose(abs(LA.norm(end) - LA.norm(astral)), 0, abs_tol=1e-15)}')
#             # print(f'state={state}')
#             push = 1
        
#         # print(f'CHECK PUSH->{push>0}')
        
#     return bind, push, state, free_spots, astral_which_spot

def check_bind(i, j,astral,spindle_poles,spots,free_spots,astral_which_spot):
    """
    Find the distance between the astral MT tip and the closest FG. If they are close enough, they bind together.
    """
    dist=distance_matrix(np.array([astral]),spots)
    if (np.min(dist[0])<=config.max_interact_dist and free_spots[np.argmin(dist[0])]==0 and random.uniform(0, 1)<=prob_dyn_bind):
        free_spots[np.argmin(dist[0])]=1
        bind=1
        astral_which_spot[np.argmin(dist[0]),0]=i
        astral_which_spot[np.argmin(dist[0]),1]=j
    else:
        bind=0
    return bind,free_spots,astral_which_spot
    
def check_push(a,b,astral,state,astral_angles,spindle_poles,cell):
    """
    Determines if a given MT is long enough to push
    or not.
    """
    end, _ =intersect_cell(a,b,astral_angles,spindle_poles,cell)

    # if (bind==0 and abs(LA.norm(end)-LA.norm(astral))<=time_step*growth_rate/50 and state==1):
        
    # math.isclose(abs(LA.norm(end)-LA.norm(astral)), 0, abs_tol=1e-12)
    # if (bind==0 and abs(LA.norm(end)-LA.norm(astral))<=min_push_dist and state==1):
    if (math.isclose(abs(LA.norm(end)-LA.norm(astral)), 0, abs_tol=1e-15) and state==1):
        push=1
    else:
        push=0

    return push
    

def check_push_junc(a,b,astral,bind,state,astral_angles,spindle_poles,cell, spots):
    """
    For endo cells, no pushing in the region of cell-cell junctions.
    """
    end, _ =intersect_cell(a,b,astral_angles,spindle_poles,cell)
    astral_angles=normalize_angles(astral_angles)
    
    opening_spot=spots[0]-spindle_poles
    closing_spot=spots[-1]-spindle_poles
    
    opening_angle=normalize_angles(np.arctan2(opening_spot[1], opening_spot[0]))
    
    closing_angle=normalize_angles(np.arctan2(closing_spot[1], closing_spot[0]))


    if (bind==0 and abs(LA.norm(end)-LA.norm(astral))<=time_step*config.growth_rate/2 and state==1 and (not opening_angle<astral_angles<closing_angle)):#somnitel'no
        push=1
    else:
        push=0

    return push

def normal_vector_to_ellipse(a,b, coordinate):
    # Unpack ellipse parameters


    # Calculate the gradient of the ellipse equation
    x, y = coordinate[0], coordinate[1]
    gradient_x = (2 * x) / (a**2)
    gradient_y = (2 * y) / (b**2)

    # Normalize the gradient vector to get the unit normal vector
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    normal_vector = np.array([gradient_x / magnitude, gradient_y / magnitude])

    return normal_vector/LA.norm(normal_vector)
    
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle)
    return angle_degrees

def slipping_f(i,j,a,b,cell, astral_MTs,pole,orig_length,spindle_angle,n_astro,cut_part):
    old_astral_MTs=astral_MTs
    #print(f'orig length={orig_length}, norm = {LA.norm(pole-astral_MTs[-1])}')
    astral_angles=np.zeros((2,n_astro))
    astral_angles[0]=np.linspace(spindle_angle-spread/2,spindle_angle+spread/2, n_astro)
    astral_angles[1]=np.linspace(spindle_angle+np.pi-spread/2,spindle_angle+np.pi+spread/2, n_astro)
#     og_angle=astral_angles[i,j]
    dist=distance_matrix(np.array([astral_MTs[-1]]),cell)[0]
    k = 2
    result = np.argpartition(dist, k)
    cell_small1,cell_small2=result[:k]
    small_dist1,small_dist2=dist[result[:k]]
    #Find tangent vector
    c=cell[cell_small1]-cell[cell_small1-1] #Tangent is always points counterclockwise from i to i+1
    c=c/LA.norm(c)
    #Find the normal vector
    n=-normal_vector_to_ellipse(a,b, astral_MTs[-1])
    #Find astral vector
    a_vec=astral_MTs[-1]-astral_MTs[0] #astral vector
    F_push=min(push, config.EI*np.pi*np.pi/LA.norm(a_vec)/LA.norm(a_vec))
    angle = math.atan2(np.linalg.det([n,a_vec]),np.dot(n,a_vec))
    slip=(F_push/config.mu_fric)*abs(np.sin(angle))*time_step + cut_part#total slip distance
    
    #slip=max(growth_rate*time_step,cut_part) #Slip happens due to MT growth and subsequent pushing, it cant be greater than max growth amount. Exception: Pole moves towards cortex
    # print(f'slip={slip}')
    while (slip>=0.01*config.natural_spacing):
        # print(f'slip I={i}')
        dist=distance_matrix(np.array([astral_MTs[-1]]),cell)[0]
        k = 2
        result = np.argpartition(dist, k)
        cell_small1,cell_small2=result[:k]
        small_dist1,small_dist2=dist[result[:k]]
        #Find tangent vector
        c=cell[cell_small1]-cell[cell_small1-1] #Tangent is always points counterclockwise from i to i+1
        c=c/LA.norm(c)
        #Find the normal vector
        n=-normal_vector_to_ellipse(a,b, astral_MTs[-1])
        #Find astral vector
        a_vec=astral_MTs[-1]-astral_MTs[0] #astral vector
        """
        If angle between a_vec and c is acute, it means slip is along c. If not, the slip is along -c.
        """
        #determine angle between astral vector and tangent
        ac_angle=angle_between_vectors(a_vec,c)
        if (ac_angle>90):
            slip_t_unit=-c
        else:
            slip_t_unit=c

        astral_MTs[-1]=astral_MTs[-1]+config.natural_spacing*slip_t_unit
        dummy=astral_MTs
        # print("after slip", LA.norm(astral_MTs[-1]-pole))
        virtual, _ =intersect_cell(a,b,get_astral_angle(dummy), pole, cell)
        if (LA.norm(virtual-pole)>orig_length+config.natural_spacing): 
            dist=distance_matrix(np.array([astral_MTs[-1]]),cell)[0]
            k = 2
            result = np.argpartition(dist, k)
            cell_small1,cell_small2=result[:k]
            small_dist1,small_dist2=dist[result[:k]]
            # Closest point on a cell
            astral_MTs[-1]=cell[cell_small1]

            # print("after tang", LA.norm(astral_MTs[-1]-pole))
        else:
            astral_MTs[-1], _ =intersect_cell(a,b,get_astral_angle(dummy), pole, cell)
            # print("after intersect", LA.norm(astral_MTs[-1]-pole))
        slip=slip-1*config.natural_spacing
        i=i+1
        if (i>20):
            break
        # print(f'SLIP FINAL={LA.norm(astral_MTs[-1]-pole)}, coords={astral_MTs[-1]}')
    return astral_MTs
    
def restructure(beam):
    restructured=np.ones(np.shape(beam))
    restructured[:,0]=np.linspace(beam[0,0],beam[-1,0],config.discr)
    restructured[:,1]=np.linspace(beam[0,1],beam[-1,1],config.discr)
    return restructured


def normalize_angles(angles):
    """
    Normalize angles to the range [0, 2π].

    Parameters:
    angles: scalar, 1D array, or (2, N) array containing angles in radians

    Returns:
    normalized_angles: object of the same shape as input with angles normalized to [0, 2π]
    """
    # Convert to numpy array for ease of operations
    angles = np.asarray(angles)
    
    # Normalize angles to the range [0, 2π]
    normalized_angles = angles % (2 * np.pi)
    
    return normalized_angles


def get_astral_angle(astral):
    angle=np.arctan2(astral[-1,1]-astral[0,1], astral[-1,0]-astral[0,0])
    if (-np.pi<=angle<=0):
        angle=angle+2*np.pi
    return angle

def get_spindle_angle(spindle):
    angle=np.arctan2(spindle[0,1]-spindle[1,1], spindle[0,0]-spindle[1,0])
    if (-np.pi<=angle<=0):
        angle=angle+2*np.pi
    return angle

def grow_astralMT(a,b,angle,C,cell,orig_length):
    
    # print("grow_astralMT")
    #Returns MT end point after growrth and if it touches the cortex (which_push) since state=1 because its growing 
    intersect, _ =intersect_cell(a,b,angle,C,cell)
    
    end=C+(orig_length+config.growth_rate*time_step)*np.array([np.cos(angle),np.sin(angle)])
    
    """
    Checks if MT tip ends up beyond cell cortex after growth. In this case, the MT end will be set to the intersection with cell cortex
    """
    len1=LA.norm([C[0]-intersect[0],C[1]-intersect[1]])
    len2=LA.norm([C[0]-end[0],C[1]-end[1]])
    # if (orig_length+config.growth_rate*time_step>=config.MT_max_length):
    #     if (point_in_polygon(C+(orig_length)*np.array([np.cos(angle),np.sin(angle)]), cell)==True):
    #         return C+(orig_length)*np.array([np.cos(angle),np.sin(angle)]), 0
    #     else:
    #         return intersect,1
   
    if (len2<len1):# and point_in_polygon(end, cell)==True): #If grown end is closer to the pole than intersect with the cortex
        # print(f'grow end={end}, start_point={C}')
        # print(f'new MT length= {LA.norm(C-end)}')
        return end
    else:
        # print(f'grow intersect={end}, start_point={C}')
        # print(f'new MT length= {LA.norm(C-end)}')
        return intersect

def make_new_MT(i,j,a,b,MT,spindle_poles,astral_angles,cell,free_spots,astral_which_spot, orig_length):
    free_spots[np.argwhere((astral_which_spot[:,0]==i) & (astral_which_spot[:,1]==j))]=0
    orig_length=0
    MT[-1]=grow_astralMT(a,b,astral_angles[i,j],spindle_poles[i],cell,orig_length)
    
    MT=restructure(MT)

    # print(f'new MT length= {LA.norm(MT[-1]-spindle_poles[i])}')
    # print(f'new MT angle={astral_angles[i,j]}')
    return MT

def update_astral_MTs(params,cell,spindle_poles,spindle_angle,delta_spindle_angle,astral_MTs,astral_angles,state, which_push, which_bind, spots,free_spots,astral_which_spot,orig_length,force_vector_1,force_vector_2, run):
    a=params[0][0]
    r=params[0][2] #spindle length=2*r
    FG_density=int(params[1])
    n_astro=int(params[2])
    b=params[0][1]
    t_time=run*time_step

    # Updating astral MTs angles
    """
    Pivoting is the process of changing the angle of the astral MTs from its original angle. 
    If pivoting is on, astral microtubule will stay at the same angle. 
    If pivoting is off, the astral MTs will return to their original angle.
    """

    """
    Refreshing astral MTs angles for a new time step. Used as a placeholder for the new angles.
    """
    if (config.pivoting==0):
        # Astral MTs angles are updated to the new spindle angle and they are constant in reference to the spindle.
        astral_angles[0]=np.linspace(spindle_angle-spread/2,spindle_angle+spread/2, n_astro)
        astral_angles[1]=np.linspace(spindle_angle+np.pi-spread/2,spindle_angle+np.pi+spread/2, n_astro)
    else:
        # Astral MTs angles change together with the spindle angle.
        astral_angles[0]=astral_angles[0]+delta_spindle_angle
        astral_angles[1]=astral_angles[1]+delta_spindle_angle

    # List to keep MTs data
    # df_list=[]#append 'Number','Death', 'Switch','short','Push','State','Length','end if out','astral angle'
    df_list2=[]
    
    for i in range (2):
        for j in range (n_astro):
            # print("astral--------+++++++++++++++++++++++++++++++-",i,j)#, check_push(a,b,astral_MTs[i,j],which_bind[i,j],state[i,j],astral_angles[i,j],spindle_poles[i],cell)>0.5)
            df_list=[]#append 'Number','Death', 'Switch','short','Push','State','Length','end if out','astral angle'
            df_list.append(run)
            df_list.append([i,j])
            astral_MTs[i,j,0]=spindle_poles[i] #Updating MT start, but this stretches microtubule because the plus end is not updated  
            # Ways MT length can change:
            # 1. pulling MTs stretched
            # 2. growing MTs +end grow
            # 3. shrinking MTs -end shrink
            
            #Check if any has unbinded            
            
            if(which_bind[i,j]==1): 
                
                astral_MTs[i,j,0]=spindle_poles[i] #Updating MT start, but this stretches microtubule because the plus end is not updated
                
                # Check if angle between spindle pole vector (12 or 21) and astral MT is larger than 90 degrees.
                

                # Condition that astral_angle[i,j] must be either within spindle_angle-spread/2,spindle_angle+spread/2 for pole 0 or within spindle_angle+np.pi-spread/2,spindle_angle+np.pi+spread/2 for pole 1
                # Wrap angles to [0, 2π]
                # angle = np.mod(astral_angles[i, j], 2 * np.pi)
                # spindle_angle_wrapped = np.mod(spindle_angle, 2 * np.pi)

                # # Check within spread near the spindle pole (side 0)
                # mask_1 = (angle >= spindle_angle_wrapped - spread/2) & (angle <= spindle_angle_wrapped + spread/2)

                # # Check near opposite pole (side 1)
                # mask_2 = (angle >= np.mod(spindle_angle_wrapped + np.pi, 2 * np.pi) - spread/2) & \
                #         (angle <= np.mod(spindle_angle_wrapped + np.pi, 2 * np.pi) + spread/2)

                # if (i == 0 and not mask_1) or (i == 1 and not mask_2):
                #     which_bind[i,j] = 0  # unbind
                #     free_spots[np.argwhere((astral_which_spot[:,0]==i) & (astral_which_spot[:,1]==j))] = 0  # bookkeeping
                #     state[i,j] = -1  # set as unbound

                

                if(config.pivoting==1):
                    """Astral MTs angles are updated to the new spindle angle after spindle has moved (and microtubule minus end correspondingly)."""
                    astral_angles[i,j]=get_astral_angle(astral_MTs[i,j]) #config.pivoting allowed
                astral_intersect, geom_type=intersect_cell(a,b,astral_angles[i,j],spindle_poles[i],cell)

                # Check if 
                if (random.uniform(0, 1)<=prob_dyn_unbind or geom_type=='MultiPoint'):
                    which_bind[i,j]=0
                    free_spots[np.argwhere((astral_which_spot[:,0]==i) & (astral_which_spot[:,1]==j))]=0 #book keeping for available FG slots
                    state[i,j]=-1
                    # if(config.pivoting==1):
                    #     astral_angles[i,j]=get_astral_angle(astral_MTs[i,j]) #config.pivoting allowed
                        #print(f'astral[{i,j}] had angle [{np.rad2deg(astral_angles[i,j])}]')
                    if (geom_type=='MultiPoint'):
                        astral_MTs[i,j,-1]=astral_intersect
                        astral_MTs[i,j]=restructure(astral_MTs[i,j])
                    df_list.append('Unbinded')
                    df_list.append('Neither R or C')
                    df_list.append('unbinded')
                else:
                    if (cell_type=='endo'):
                        if (int(t_time)%frame_rates[c-1]==0 and i>int(0.9*frame_rates[c-1]) and i<total_time/time_step):
                            
                            if (config.mobile_motors==1): #if the cell shape changes and new spots forms keep the astral MT glued to the same index spot
                                astral_MTs[i,j,-1]=spots[np.where((astral_which_spot == np.array([i,j]) ).all(axis=1))[0]][0]
                            
                            else:
                                free_spots[np.argwhere((astral_which_spot[:,0]==i) & (astral_which_spot[:,1]==j))]=0 #release old spot
                                
                                dist=distance_matrix(np.array([astral_intersect]),spots) #find new closest spot
                                
                                astral_MTs[i,j, -1] = spots[np.argmin(dist[0])] #put the end to the new spot
                                which_bind[i,j],free_spots,astral_which_spot=check_bind_init(i, j,astral_MTs[i,j,-1],spindle_poles,spots,free_spots,astral_which_spot)
                                
                            astral_MTs[i,j]=restructure(astral_MTs[i,j])

                    df_list.append('Not Dead not binded')
                    df_list.append('Neither R or C')
                    df_list.append('stays binded')
                    
            #NOT BINDED 
            elif(which_bind[i,j]==0): # didn't bind (which_bind[i,j]=0)
                df_list.append('Not Dead not binded')
                rand_n=random.uniform(0, 1)
                if (state[i,j]>0 and rand_n<=prob_catastr):  #if catastrophe happens
                    state[i,j] = -1
                    rate=config.shrink_rate
                    df_list.append('Catastrophe')
                elif (state[i,j]<0 and rand_n<=prob_rescue): #if rescue happens
                    state[i,j] = 1
                    rate=config.growth_rate
                    df_list.append('Rescue')
                else:
                    df_list.append('Neither R or C')
                    rate=config.shrink_rate if state[i,j]==-1 else config.growth_rate

                #GROWING AND SHRINKING
                
                #Floating shrinking and growing MTs spin with the spindle
                
                #SHRINKING   
                if(state[i,j]==-1):
                    # print("SHRINKING")
                    old_astral_MTs=spindle_poles[i]+orig_length[i,j]*np.array([np.cos(astral_angles[i,j]),np.sin(astral_angles[i,j])]) # Shifting astral MTs due to new -end coordinate
                    length_delta=rate*np.array([np.cos(astral_angles[i,j]),np.sin(astral_angles[i,j])]) # shrink length
                    astral_MTs[i,j,-1]=old_astral_MTs+length_delta # Substracting
                    astral_MTs[i,j,0]=spindle_poles[i] #Updating MT start, but this stretches microtubule because the plus end is not updated
                    # print(f'old={old_astral_MTs+length_delta}, new={astral_MTs[i,j,-1]}')
                    astral_MTs[i,j]=restructure(astral_MTs[i,j])
                    astral_intersect, _ =intersect_cell(a,b,astral_angles[i,j],spindle_poles[i],cell)
                    #Comparing which point is further from the spindle pole (intersect or shrinking end)
                    if (LA.norm(astral_MTs[i,j,-1]-spindle_poles[i])>LA.norm(astral_intersect-spindle_poles[i])):
                        astral_MTs[i,j,-1]=astral_intersect
                        astral_MTs[i,j]=restructure(astral_MTs[i,j])
                        
                    if(LA.norm(astral_MTs[i,j,-1]-spindle_poles[i])<config.MT_min_length): #Check if astral MT is too short and is going to be replaced
                        #Angle Options for new astral
                        opt_astral_angles=np.zeros((2,n_astro))
                        opt_astral_angles[0]=np.linspace(spindle_angle-spread/2,spindle_angle+spread/2, n_astro)
                        opt_astral_angles[1]=np.linspace(spindle_angle+np.pi-spread/2,spindle_angle+np.pi+spread/2, n_astro)
                        
                        
                        astral_MTs[i,j]=make_new_MT(i,j,a,b, astral_MTs[i,j],spindle_poles,opt_astral_angles,cell,free_spots,astral_which_spot, orig_length[i,j])
                        astral_angles[i,j]=get_astral_angle(astral_MTs[i,j])
                        orig_length[i,j]==LA.norm(astral_MTs[i,j,-1]-spindle_poles[i])
                        state[i,j]=1
                        df_list.append('short and reborn')
                    else:
                        df_list.append('not short shrinking')
                    #Update length and check for a new pulling or pushing MT emergence
                    orig_length[i,j]=LA.norm(astral_MTs[i,j,-1]-spindle_poles[i])
                    
                    which_bind[i,j], which_push[i,j], state[i,j], free_spots,astral_which_spot=check_push_bind(i, j, astral_MTs[i,j,-1], spindle_poles, spots, free_spots, astral_which_spot, state[i,j], astral_angles[i,j], cell)

                #GROWING    
                else: 
                    # print("GROWING")
                    old_angle=astral_angles[i,j]
                    #print(f'orig_length[i,j]={orig_length[i,j]},LA.norm={LA.norm(astral_MTs[i,j,-1]-spindle_poles[i])}')
                    #astral_angles[i,j]=get_astral_angle(astral_MTs[i,j])
                    virtual_astral=grow_astralMT(a,b,astral_angles[i,j],spindle_poles[i],cell,orig_length[i,j]+config.growth_rate*time_step) #growing astral MTs without considering config.slipping behaviour
                    _ , virtual_push, _, _ ,_ =check_push_bind(i, j, astral_MTs[i,j,-1], spindle_poles, spots, free_spots, astral_which_spot, state[i,j], astral_angles[i,j], cell)
                        
                    # print(f'real={astral_MTs[i,j,-1]}, virtual={virtual_astral}')
                    if (config.slipping==1 and virtual_push==1): #Figure out if the MT is touching the cortex before growing (prerequisite for config.slipping)
                    # if (config.slipping==1 and check_push(a,b,virtual_astral,which_bind[i,j],state[i,j],astral_angles[i,j],spindle_poles[i],cell)==1): #Figure out if the MT is touching the cortex before growing (prerequisite for config.slipping)
                        
                        # print("astral config.slipping--------+++++++++++++++++++++++++++++++-",i,j)
                        virtual_astral=grow_astralMT(a,b,astral_angles[i,j],spindle_poles[i],cell,orig_length[i,j])
                        # print(f'L={orig_length[i,j]}, virtual L={LA.norm(virtual_astral-spindle_poles[i])}')
                        # print(f'astral={astral_MTs[i,j,-1]}, virtual={virtual_astral}')
                        center=np.array([1/2*(spindle_poles[0,0]+spindle_poles[1,0]),1/2*(spindle_poles[0,1]+spindle_poles[1,1])])
                        spin_vec=center-spindle_poles[i]
                        a_vec=astral_MTs[i,j,-1]-astral_MTs[i,j,0]
                        as_angle=angle_between_vectors(a_vec,spin_vec)
                        
                        # if (as_angle<=90):
                        #     print("slip angle is obtuse")
                        #     cut_part=0
                        # else:
                        cut_part=max(0,(orig_length[i,j]+config.growth_rate*time_step)-LA.norm(virtual_astral-spindle_poles[i]))
                        # print(f'cut={cut_part:.3f}')
                        astral_MTs[i,j]=slipping_f(i,j,a,b,cell,astral_MTs[i,j],spindle_poles[i],orig_length[i,j], spindle_angle, n_astro, cut_part) #cell, astral_MTs,pole,orig_length
                        astral_angles[i,j]=get_astral_angle(astral_MTs[i,j])
                        which_bind[i,j], which_push[i,j],state[i,j], free_spots,astral_which_spot=check_push_bind(i, j, astral_MTs[i,j,-1], spindle_poles, spots, free_spots, astral_which_spot, state[i,j], astral_angles[i,j], cell)
            
                        df_list.append('not short config.slipping') 
                    else:
                        # print("astral--------+++++++++++++++++++++++++++++++-",i,j)
                        astral_MTs[i,j,-1]=grow_astralMT(a,b,astral_angles[i,j],spindle_poles[i],cell,orig_length[i,j])
                        which_bind[i,j], which_push[i,j],state[i,j], free_spots,astral_which_spot=check_push_bind(i, j, astral_MTs[i,j,-1], spindle_poles, spots, free_spots, astral_which_spot, state[i,j], astral_angles[i,j], cell)
                        df_list.append('not short growing') 
            orig_length[i,j]=LA.norm(astral_MTs[i,j,-1]-spindle_poles[i])
            
            astral_intersect,_ =intersect_cell(a,b,astral_angles[i,j],spindle_poles[i],cell)
            #Comparing which point is further from the spindle pole (intersect or shrinking end)
            # if (LA.norm(astral_MTs[i,j,-1]-spindle_poles[i])>LA.norm(astral_intersect-spindle_poles[i])):
            #     print(f'astral[{i,j}] is outside and bypasses')
            
            # Recording data to excel sheet
            df_list.append(which_bind[i,j])
            df_list.append(which_push[i,j])
            df_list.append(state[i,j])
            if (i==0):
                df_list.append(np.linalg.norm(force_vector_1[j]))
            else:
                df_list.append(np.linalg.norm(force_vector_2[j]))

            df_list.append(LA.norm([astral_MTs[i,j,-1,0]-spindle_poles[i,0],astral_MTs[i,j,-1,1]-spindle_poles[i,1]]))
            df_list.append(orig_length[i,j])
            df_list.append(math.sqrt(astral_MTs[i,j,-1,0]*astral_MTs[i,j,-1,0]+astral_MTs[i,j,-1,1]*astral_MTs[i,j,-1,1]))
            df_list.append(math.degrees((astral_angles[i,j])))
            df_list2.append(df_list)
            #print(f'end of update')
    return astral_MTs,astral_angles, state, which_push, which_bind,free_spots,astral_which_spot,orig_length, df_list2



def distance_to_boundary(point, shape_coords):
    """
    Calculate the minimum distance from a point to the boundary defined by shape_coords.

    Parameters:
    point (np.array): The point (x, y) for which to calculate the distance.
    shape_coords (np.array): The coordinates of the boundary shape (N, 2).

    Returns:
    float: The minimum distance from the point to the shape's boundary.
    """
    # Calculate the Euclidean distance from the point to all points in shape_coords
    distances = np.sqrt((shape_coords[:, 0] - point[0])**2 + (shape_coords[:, 1] - point[1])**2)
    
    return np.min(distances)

def rotate_points(points, spindle_angle):
    centroid = np.mean(points, axis=0)
    # Create the 2D rotation matrix
    rotation_matrix = np.array([
        [np.cos(spindle_angle), np.sin(spindle_angle)],
        [-np.sin(spindle_angle), np.cos(spindle_angle)]
    ])
    # Translate points to origin (subtract centroid)
    translated_points = points - centroid
    # Apply the rotation matrix
    rotated_points = np.dot(translated_points, rotation_matrix)
    rotated_points += centroid
    return rotated_points

def compute_resistance_functions(a, b):
    """
    Computes the translational and rotational resistance functions (Table 3.4).
    Now handles circular cross-sections (a == b) with analytical solutions.
    
    Parameters:
    a : float - Semi-major axis
    b : float - Semi-minor axis
    
    Returns:
    X_A, Y_A, X_C, Y_C : float - Resistance functions for translation & rotation
    """
    
    # Check for circular case first
    if a == b:
        # Analytical solutions for circle (all equal)
        resistance = 16/3
        return resistance, resistance, resistance, resistance
    
    # Original check for invalid input
    if a < b:
        raise ValueError("Semi-major axis (a) must be greater than or equal to semi-minor axis (b).")
    
    # Compute eccentricity
    e = np.sqrt(1 - (b**2 / a**2))
    
    # Compute logarithmic correction term
    L = np.log((1 + e) / (1 - e))
    
    # Compute resistance functions
    X_A = (8/3) * (e**3 / (-2 * e + (1 + e**2) * L))
    Y_A = (16/3) * (e**3 / (2 * e + (1 + e**2) * L))
    X_C = (4/3) * (e**3 * (1 - e**2) / (2 * e - (1 - e**2) * L))
    Y_C = (4/3) * (e**3 * (2 - e**2) / (-2 * e + (1 + e**2) * L))

    return X_A, Y_A, X_C, Y_C

def generate_spindle(spindle_poles, spindle_angle, r, w):
    spindle=np.zeros((359,2))
    theta = np.linspace(0, 2*np.pi, 360)[:-1].copy()

    com=np.array([(spindle_poles[0,0]+spindle_poles[1,0])/2,(spindle_poles[0,1]+spindle_poles[1,1])/2]) #rotation is about com(centre of mass)
    spindle[:,0]=r*np.cos(theta)+com[0]
    spindle[:,1]=w*np.sin(theta)+com[1]

    spindle = rotate_points(spindle, spindle_angle) #rotate spindle to the right angle

    return spindle

def check_spindle(spindle_poles, spindle_angle, cell, r, w):

   
    # cell_polygon = Polygon(cell)

    # dist_pole1 = distance_to_boundary(spindle_poles[0], cell)
    # dist_pole2 = distance_to_boundary(spindle_poles[1], cell)
    # min_dist = min(dist_pole1, dist_pole2)
    r=LA.norm(spindle_poles[0]-spindle_poles[1])/2 #radius of the spindle
    spindle_body=generate_spindle(spindle_poles, spindle_angle, r, w)
    # spindle_polygon=Polygon(spindle_body)
    
    from shapely.geometry import Polygon
    from shapely.validation import make_valid

    min_distance=0.01
    """
    More robust version with:
    - Polygon validation
    - Edge case handling
    - Distance verification
    """
    try:
        cell = make_valid(Polygon(cell))
        spindle = make_valid(Polygon(spindle_body))
        
        if not cell.is_valid or not spindle.is_valid:
            return False
            
        # Verify minimum distance
        if cell.boundary.distance(spindle) < min_distance:
            print(f"Debug: Boundary condition violated! - Spindle too close to cell boundary: {cell.boundary.distance(spindle)} < {min_distance}")
            return False
            
        # Verify complete containment
        return cell.buffer(-min_distance).contains(spindle)
    except:
        return False

# def check_spindle(spindle_poles, spindle_angle, cell, r, w):

   
#     cell_polygon = Polygon(cell)

#     dist_pole1 = distance_to_boundary(spindle_poles[0], cell)
#     dist_pole2 = distance_to_boundary(spindle_poles[1], cell)
#     min_dist = min(dist_pole1, dist_pole2)
#     spindle_body=generate_spindle(spindle_poles, spindle_angle, r, w)
#     spindle_polygon=Polygon(spindle_body)
    
#     # Check boundary conditions
#     if cell_type=='endo' and cell_polygon.intersects(spindle_polygon):
#         print(f"Debug: Boundary condition violated! - Spindle outside")
#         return False
    
#     if min_dist <= config.min_cortex_dist:# or cell_polygon.intersects(spindle_polygon):
#         print(f"Debug: Boundary condition violated!")
#         # Debug: Print boundary violation details
#         if min_dist <= config.min_cortex_dist:
#             print(f"Debug: min_dist <= config.min_cortex_dist: {min_dist - config.min_cortex_dist}")
        
#         return False  # Boundary condition violated
    
#     for pole in spindle_poles:
#         point=pole
#         if not cell_polygon.contains(Point(point)):
#             # Debug: Print boundary violation
#             print(f"[DEBUG] Boundary violation [POLE OUTSIDE] at point: {point}")
#             return False  # Boundary condition violated

#     # Debug: Boundary conditions satisfied
#     print(f"Debug: Boundary conditions satisfied!")
#     return True  # Boundary conditions satisfied



def find_torque(spindle_poles, r, force1, force2):
    r=np.linalg.norm(spindle_poles[0]-spindle_poles[1])/2 #radius of the spindle
    poles12_unit=(spindle_poles[0]-spindle_poles[1])/(2*r)
    
    force1_t=force1-force1.dot(poles12_unit)*poles12_unit #transverse to the spindle
    force2_t=force2-force2.dot(-poles12_unit)*(-poles12_unit)
    
    sign1=np.sign(np.cross(poles12_unit,force1_t))#sign for torque using sign of cross product
    sign2=np.sign(np.cross(-poles12_unit,force2_t))

    arm1,arm2 = r, r

    # Floating point error may cause dot product to go outside [-1,1]
    
    if (np.linalg.norm(force1)!=0):

        input_value_safe = np.clip(np.dot(poles12_unit, force1/np.linalg.norm(force1)), -1, 1)  # Clip the value to be within [-1, 1]
        angle1 = np.arccos(input_value_safe) 

    if (np.linalg.norm(force2)!=0):

        input_value_safe = np.clip(np.dot(-poles12_unit, force2/np.linalg.norm(force2)), -1, 1)  # Clip the value to be within [-1, 1]
        angle2 = np.arccos(input_value_safe) 
    
    torque=sign1*LA.norm(force1_t)*arm1+sign2*LA.norm(force2_t)*arm2

    return torque

def new_spindle_poles(spindle_poles, spindle_angle, V, Omega):
    r=np.linalg.norm(spindle_poles[0]-spindle_poles[1])/2 #radius of the spindle
    new_poles = spindle_poles + (V * time_step) 
    new_angle = spindle_angle + (Omega * time_step) 

    com=np.array([(new_poles[0,0]+new_poles[1,0])/2,(new_poles[0,1]+new_poles[1,1])/2]) #rotation is about com(centre of mass)
        
    new_poles[0]=[r*np.cos(new_angle)+com[0],r*np.sin(new_angle)+com[1]]
    new_poles[1]=[r*np.cos(new_angle+np.pi)+com[0],r*np.sin(new_angle+np.pi)+com[1]]
    return new_poles, new_angle

def drags(r, w):
    b=w
    a=r
    ec = np.sqrt(1 - (b**2 / a**2))  # Eccentricity
    L=np.log((1 + ec) / (1 - ec))  # Logarithmic correction term
    # print(f'[DEBUG] Eccentricity={ec}, Logarithmic correction={L}')

    t_drag_par=6*np.pi*config.visc*a*((8/3)*ec**3/(-2*ec+(1+ec**2)*L)) #viscous resistance for translation along the spindle axis
    t_drag_perp=6*np.pi*config.visc*a*((16/3)*ec**3/(2*ec+(1+ec**2)*L)) #viscous resistance for translation perpendicular to the spindle axis
    # rot_drag = (16 * np.pi*config.visc* a**3 * ec**3)/(3 * ((2 - ec**2) * np.log((1+ec)/(1-ec)) - 2*ec))
    rot_drag = ((32/3) * np.pi*config.visc* a**3 * ec**3)*(2-ec**2)/(((1 + ec**2) * np.log((1+ec)/(1-ec)) - 2*ec))
    return t_drag_par, t_drag_perp, rot_drag

def elongate_spindle(spindle_poles, spindle_angle, r, w):
    # Calculate the elongation of the spindle
    # spindle_poles=np.array((2,2))
    delta_length = 0.002 * time_step  # Example elongation value, adjust as needed
    
    # print(f'[DEBUG] Elongating spindle poles by {delta_length} units.')
    # Displace each spindle pole by half the elongation in the direction of the each pole

    # Update the spindle poles based on the elongation
    vec12 = spindle_poles[1] - spindle_poles[0]  # Vector from pole1 to pole2
    vec12=vec12 / LA.norm(vec12)  # Normalize the vector
    vec21 = spindle_poles[0] - spindle_poles[1]  # Vector from pole2 to pole1
    vec21=vec21 / LA.norm(vec21)  # Normalize the vector
    new_pole1 = spindle_poles[0] + vec21 * (delta_length )  # Move pole1 away from pole2
    new_pole2 = spindle_poles[1] #+ vec12 * (delta_length / 2)  # Move pole2 away from pole1

    # Return new poles
    spindle_poles= np.array([new_pole1, new_pole2])
    # New spindle length
    spindle_length = LA.norm(spindle_poles[0] - spindle_poles[1])  # Spindle length
    # print(f'[DEBUG] Elongated spindle length and difference={spindle_length, spindle_length - 2*r}')
    return spindle_poles

def move_spindle(params, astral_MTs,astral_angles, state,spindle_poles,spindle_angle,cell,spots,which_push, which_bind,free_spots,astral_which_spot,orig_length, v_c, i):
    
    #Parameters
    a=params[0][0]
    b=params[0][1]
    r=params[0][2] #spindle length=2*r
    n_astro=int(params[2])
    t_time=i*time_step
    spindle_length=LA.norm(spindle_poles[0]-spindle_poles[1]) #Spindle length
    r=spindle_length/2
    
    
    dum_spot=astral_which_spot
    if (cell_type=='endo'):
        if (i%(1/time_step)==0 and i<total_time/time_step):
        # if (int(t_time)%frame_rates[c-1]==0 and i>int(0.9*frame_rates[c-1]) and i<total_time/time_step):
            # print("CELL SHAPE CHANGE")
            cell=update_cell(t_time)
            spots=make_spots(int(params[1]),a, b,cell,spindle_poles, int(t_time/frame_rates[c-1]))
            free_spots=transfer_free_spots(len(spots), free_spots)
            astral_which_spot=transfer_astro_which(len(spots), dum_spot)



    # New spindle length
    # spindle_length=LA.norm(spindle_poles[0]-spindle_poles[1]) #Spindle length
    # spindle_poles=adjust_spindle_poles(spindle_poles, spindle_angle, cell) #Ch
    #Calculating repel force
    dist_to_cort=distance_matrix(spindle_poles,cell)
    repel_force1=np.array([0,0])
    repel_force2=np.array([0,0])
    
    # Astral MTs correction
    # t_drag_par=t_drag_par*(1+0.12*2*n_astro) # Astral MTs correction for translational drag
    # t_drag_perp=t_drag_perp*(1+0.12*2*n_astro) # Astral MTs correction for translational drag
    # rot_drag=rot_drag*(1+0.17*2*n_astro) # Astral MTs correction for rotational drag
    
    push_force, pull_force=find_force(astral_MTs,spindle_poles,which_bind,which_push, v_c)

    # Find TOTAL pulling and pushing forces
    pull_t=np.sum(pull_force[0]+pull_force[1], axis=0)
    push_t=np.sum(push_force[0]+push_force[1], axis=0)

    #Calculating ratio of pulling and pushing forces
    
    if((LA.norm(pull_t)+LA.norm(push_t))==0):
        ratio=0
    else:
        ratio=100*LA.norm(pull_t)/(LA.norm(pull_t)+LA.norm(push_t))
    
    force_vector_1=pull_force[0]+push_force[0]
    force_vector_2=pull_force[1]+push_force[1]

    # Adding repel force
    force1=np.sum(force_vector_1, axis=0)
    force2=np.sum(force_vector_2, axis=0)

    force_net=(force1+force2)

    spindle_dir = np.array([np.cos(spindle_angle), np.sin(spindle_angle)]) 
    delta_spindle_angle=0
    mu=config.visc
    torque=find_torque(spindle_poles, r, force1, force2) 
    # print(f'[DEBUG] Spindle angle={np.rad2deg(spindle_angle)}, Spindle direction={spindle_dir}')
    if (r!=w):
        F_parallel = np.dot(force_net, spindle_dir) * spindle_dir
        F_perpendicular = force_net - F_parallel
        t_drag_par, t_drag_perp, rot_drag = drags(r, w) # Translational and rotational drag coefficients
        V= F_perpendicular / t_drag_perp + F_parallel / t_drag_par  # Translational velocity
        Omega = torque / rot_drag  # Rotational velocity
    else:
        # Circular case
        # r=2*r #Kimura effective drag
        V= force_net / (6 * np.pi * mu * r)  # Translational velocity for circular cross-section
        Omega = torque / (8 * np.pi * mu * r**3)  #
    # print(f'np.cross(spindle_dir, force_net)={np.cross(spindle_dir, force_net)}')
    # print(f'[DEBUG] Torque={torque}')
    # print(f'[DEBUG] Force net={force_net}, V={V}, Omega={Omega}')

    old_poles=spindle_poles
    
    
    # print(f'Net force={np.linalg.norm(force_net)}, torque={torque}')
    if (np.linalg.norm(force_net)!=0 and torque!=0):

        
        
        # Adjustment loop
        
            
        # print(f"[DEBUG] ROTATION Adjustment iteration {k + 1} with scale: {scale}")
        new_poles, new_angle = new_spindle_poles(spindle_poles, spindle_angle, V, Omega)
        
        new_poles_R, new_angle_R = new_spindle_poles(spindle_poles, spindle_angle, 0, Omega)
        
        new_poles_T, new_angle_T = new_spindle_poles(spindle_poles, spindle_angle, V, 0)

        if check_spindle(new_poles, new_angle, cell, r, w):
            
            spindle_angle=new_angle
        
            delta_spindle_angle=time_step*Omega
            
            spindle_poles=new_poles

            # print("[DEBUG] Update succesful - No Adjustment")
            

        
        elif check_spindle(new_poles_R, new_angle_R, cell, r, w):
            
            # print("Rotation path")
            
            spindle_angle=new_angle
        
            delta_spindle_angle=time_step*Omega
            
            spindle_poles=new_poles
            
            scale=0.5
            max_iter=20
            for k in range(max_iter):
                
                test_poles=new_poles_R+V*time_step*scale

                if check_spindle(test_poles, spindle_angle, cell, r, w):
                    
                    spindle_poles=test_poles

                    # print(f"[DEBUG] Rotation path: TRANSLATION Adjustment successful at iteration {k + 1} with scale: {scale}")
                    
                    break
        
                if (k==19):

                    # print("[DEBUG] Rotation path: TRANSLATION Adjustment failed after maximum iterations. Returning original state.")
                    break
                
                scale *= 0.5
                
        
        elif check_spindle(new_poles_T, new_angle_T, cell, r, w):
            
            # print("Translation path")
            scale=0.5
            max_iter=20
            for k in range(max_iter):
                

                test_poles, test_angle = new_spindle_poles(spindle_poles, spindle_angle, V*scale, Omega*scale)

                if check_spindle(test_poles, test_angle, cell, r, w):
                    
                    spindle_poles=test_poles
                    
                    spindle_angle=test_angle
                                
                    delta_spindle_angle=time_step*Omega*scale
                    
                    # print(f"[DEBUG] Translation path: ROTATION Adjustment successful at iteration {k + 1} with scale: {scale}")
                    
                    break
        
                if (k==19):

                    # print("[DEBUG] Translation path: ROTATION Adjustment failed after maximum iterations. Returning original state.")
                    break
                
                scale *= 0.5
                
        else:
            
            # print("Naruto path")
            
            scale=0.5
            max_iter=20
            for k in range(max_iter):
                
                test_angle=spindle_angle+time_step*Omega*scale
                test_poles, _ = new_spindle_poles(spindle_poles, test_angle, 0, 0)

                if check_spindle(test_poles, test_angle, cell, r, w):
                    
                    spindle_poles=test_poles
                    
                    spindle_angle=test_angle
                                
                    delta_spindle_angle=time_step*Omega*scale
                    
                    # print(f"[DEBUG] Naruto path: Adjustment successful at iteration {k + 1} with scale: {scale}")
                    
                    break
        
                if (k==19):

                    # print("[DEBUG] Naruto path: Adjustment failed after maximum iterations. Returning original state.")
                    break
                
                scale *= 0.5

          
    
    print(f'[DEBUG] Angle change={np.rad2deg(delta_spindle_angle)}')
    print(f'[DEBUG] Position change={spindle_poles-old_poles, LA.norm(spindle_poles-old_poles)}')
    
    v_c=(spindle_poles-old_poles)/time_step
    # print(f'v_c={v_c}')
    r=np.linalg.norm(spindle_poles[0]-spindle_poles[1])/2
    if (spindle_length<1.6):
        # Debug
        print(f"[DEBUG] Spindle length too short: {spindle_length}. Elongating spindle poles.")
        spindle_poles=elongate_spindle(spindle_poles, spindle_angle, r, w) #Elongate spindle poles
    r=np.linalg.norm(spindle_poles[0]-spindle_poles[1])/2 #radius of the spindle
    astral_MTs, astral_angles, state, which_push,which_bind,free_spots,astral_which_spot,orig_length, df_list2=update_astral_MTs(params,cell,spindle_poles,spindle_angle,delta_spindle_angle,astral_MTs,astral_angles, state,which_push, which_bind,spots,free_spots,astral_which_spot,orig_length,force_vector_1,force_vector_2,i)
    
    return cell, spots, spindle_poles, spindle_angle, astral_MTs, astral_angles, state, which_push,which_bind,free_spots,astral_which_spot,orig_length, df_list2, ratio, push_force, pull_force, v_c#force1_next,force2_next


# def move_spindle(params, astral_MTs,astral_angles, state,spindle_poles,spindle_angle,cell,spots,which_push, which_bind,free_spots,astral_which_spot,orig_length, v_c, i):
    
#     #Parameters
#     a=params[0][0]
#     b=params[0][1]
#     r=params[0][2] #spindle length=2*r
#     t_time=i*time_step

#     spindle_b=0.8*a

#     # X_A, Y_A, X_C, Y_C = compute_resistance_functions(r/2, spindle_b/2)
#     X_A, Y_A, X_C, Y_C = compute_resistance_functions(max(r, w), min(r, w))
#     # Projection operators
#     r=params[0][2] #spindle length=2*r
#     drag=params[0][3]
#     t_time=i*time_step


#     #Calculating repel force
#     dist_to_cort=distance_matrix(spindle_poles,cell)
#     repel_force1=np.array([0,0])
#     repel_force2=np.array([0,0])
    
#     # if (np.min(dist_to_cort[0])<=1.5*config.min_cortex_dist):

#     #     repel_vec1=-spindle_poles[0]/LA.norm(spindle_poles[0])
#     #     repel_force1=(repel/np.min(dist_to_cort[0])**1)*repel_vec1

#     # if (np.min(dist_to_cort[1])<=1.5*config.min_cortex_dist):

#     #     repel_vec2=-spindle_poles[1]/LA.norm(spindle_poles[1])
#     #     repel_force2=(repel/np.min(dist_to_cort[1])**1)*repel_vec2

    
#     # print(f'repel force={repel_force1, repel_force2}')
    
#     push_force, pull_force=find_force(astral_MTs,spindle_poles,which_bind,which_push, v_c)

#     # Find TOTAL pulling and pushing forces
#     pull_t=np.sum(pull_force[0]+pull_force[1], axis=0)
#     push_t=np.sum(push_force[0]+push_force[1], axis=0)

#     #Calculating ratio of pulling and pushing forces
    
#     if((LA.norm(pull_t)+LA.norm(push_t))==0):
#         ratio=0
#     else:
#         ratio=100*LA.norm(pull_t)/(LA.norm(pull_t)+LA.norm(push_t))
    
#     force_vector_1=pull_force[0]+push_force[0]
#     force_vector_2=pull_force[1]+push_force[1]

#     # Adding repel force
#     force1=np.sum(force_vector_1, axis=0)#+repel_force1
#     force2=np.sum(force_vector_2, axis=0)#+repel_force2

#     force_net=(force1+force2)

#     v_c=force_net*time_step/drag

#     delta_spindle_angle=0
    
#     mu=config.visc
    
#     F=np.array([force_net[0],force_net[1], 0])
    
#     torque=find_torque(spindle_poles, r, force1, force2)
    
#     if (np.linalg.norm(force_net)!=0 and torque!=0):

#         T=np.array([0,0,torque])
#         d=(spindle_poles[0]-spindle_poles[1])/(2*r)
#         d=np.array([d[0], d[1], 0])
#         d_outer = np.outer(d, d)  # d_i d_j (parallel projection)
#         I = np.eye(3)  # Identity tensor
#         P_perp = I - d_outer  # Projection perpendicular to the symmetry axis
#         # Compute translational velocity
#         U = (F / (6 * np.pi * mu * a)) @ (X_A * d_outer + Y_A * P_perp)
#         # Compute rotational velocity
#         Omega = (T / (8 * np.pi * mu * a**3)) @ (X_C * d_outer + Y_C * P_perp)
#         # print(f'Ang velo={Omega}')

#         Omega=Omega[2]

#         og_poles=spindle_poles       
        
         
#         # ROTATION
#         test_poles=np.zeros((2,2))
#         scale=1
#         max_iter=20
#         for k in range(max_iter):
            
#             # print(f"[DEBUG] ROTATION Adjustment iteration {k + 1} with scale: {scale}")
#             test_angle=spindle_angle+time_step*Omega*scale
            
#             com=np.array([(spindle_poles[0,0]+spindle_poles[1,0])/2,(spindle_poles[0,1]+spindle_poles[1,1])/2]) #rotation is about com(centre of mass)
            
#             test_poles[0]=[r*np.cos(test_angle)+com[0],r*np.sin(test_angle)+com[1]]
#             test_poles[1]=[r*np.cos(test_angle+np.pi)+com[0],r*np.sin(test_angle+np.pi)+com[1]]

#             if check_spindle(test_poles, test_angle, cell, r, w):
                
#                 spindle_angle=test_angle
            
#                 delta_spindle_angle=time_step*Omega*scale
                
#                 spindle_poles=test_poles

#                 print(f"[DEBUG] ROTATION Adjustment successful at iteration {k + 1} with scale: {scale}")
                
#                 break
    
#             if (k==9):
#                 spindle_poles=og_poles#poles_after_translation#

#                 print("[DEBUG] ROTATION Adjustment failed after maximum iterations. Returning original state.")
                
#                 break

#             scale *= 0.5
#         # saving
#         print(f'[DEBUG] Angle change={np.rad2deg(delta_spindle_angle)}')

#         poles_after_rotation=spindle_poles
        
#         # TRANSLATION
        
#         sp_velo=U[:2]
#         spindle_poles=spindle_poles+sp_velo*time_step
#         k=0
#         scale=1
#         max_iter=20

#         for k in range(max_iter):
#             # print(f"[DEBUG] TRANSLATION Adjustment iteration {k + 1} with scale: {scale}")
#             test_poles=spindle_poles+sp_velo*time_step*scale

#             if check_spindle(spindle_poles, spindle_angle, cell, r, w):
                
#                 spindle_poles=test_poles

#                 print(f"[DEBUG] TRANSLATION Adjustment successful at iteration {k + 1} with scale: {scale}")
                
#                 break
    
#             if (k==9):
#                 spindle_poles=poles_after_rotation#og_poles#
#                 print("[DEBUG] TRANSLATION Adjustment failed after maximum iterations. Returning original state.")
#                 break
            
#             scale *= 0.5
        
#         poles_after_translation=spindle_poles
        
    
#     # print(f'[DEBUG] Position change={sp_velo*time_step*scale, LA.norm(sp_velo*time_step*scale)}')
        
        
#     astral_MTs, astral_angles, state, which_push,which_bind,free_spots,astral_which_spot,orig_length, df_list2=update_astral_MTs(params,cell,spindle_poles,spindle_angle,delta_spindle_angle,astral_MTs,astral_angles, state,which_push, which_bind,spots,free_spots,astral_which_spot,orig_length,force_vector_1,force_vector_2,i)
    
#     return cell, spots, spindle_poles, spindle_angle, astral_MTs, astral_angles, state, which_push,which_bind,free_spots,astral_which_spot,orig_length, df_list2, ratio, push_force, pull_force, v_c#force1_next,force2_next


def move_severed_spindle(params, astral_MTs, astral_angles, state, spindle_poles, spindle_angle, cell, spots, which_push, which_bind, free_spots, astral_which_spot, orig_length, v_c, i):
    # Parameters
    a = params[0][0]  # Sphere radius (formerly semi-major axis)
    b = params[0][1]  # Unused now (was semi-minor axis)
    r = params[0][2]  # Original spindle length (now used for distance checks)
    t_time = i * time_step

    # Each centrosome is now an independent sphere with radius 'a'
    mu = config.visc

    # Calculate forces for each centrosome independently
    push_force, pull_force = find_force(astral_MTs, spindle_poles, which_bind, which_push, v_c)
    
    # Force vectors for each centrosome
    force1 = np.sum(pull_force[0] + push_force[0], axis=0)
    force2 = np.sum(pull_force[1] + push_force[1], axis=0)

    # Calculate movement for each centrosome independently (Stokes' law for spheres)
    def move_sphere(position, force, cell_boundary, sphere_radius):
        # Simple Stokes' drag for sphere
        velocity = force / (6 * np.pi * mu * sphere_radius)
        
        # Proposed new position
        new_position = position + velocity * time_step
        
        # Check boundary conditions (similar to original check_spindle but for single sphere)
        if LA.norm(new_position) < cell_boundary - sphere_radius:
            return new_position, velocity
        else:
            # If would hit boundary, scale down movement
            max_dist = cell_boundary - sphere_radius - LA.norm(position)
            if max_dist > 0:
                scale = max_dist / LA.norm(velocity * time_step)
                return position + velocity * time_step * scale, velocity * scale
            else:
                return position, np.zeros(2)

    # Move each centrosome independently
    cell_radius = LA.norm(cell[0])  # Assuming cell is circular
    new_pole1, v1 = move_sphere(spindle_poles[0], force1, cell_radius, 0.16)
    new_pole2, v2 = move_sphere(spindle_poles[1], force2, cell_radius, 0.16)

    # Update spindle poles (now independent)
    spindle_poles = np.array([new_pole1, new_pole2])
    v_c = np.array([0,0])#(spindle_poles - old_poles) / time_step

    # Update spindle angle based on new pole positions
    #vec = spindle_poles[1] - spindle_poles[0]
    new_angle = spindle_angle#np.arctan2(vec[1], vec[0]) if LA.norm(vec) > 0 else spindle_angle
    delta_spindle_angle = new_angle - spindle_angle
    spindle_angle = new_angle

    # Update MTs (same as before but now poles may be completely independent)
    astral_MTs, astral_angles, state, which_push, which_bind, free_spots, astral_which_spot, orig_length, df_list2 = update_astral_MTs(
        params, cell, spindle_poles, spindle_angle, delta_spindle_angle, astral_MTs, astral_angles, 
        state, which_push, which_bind, spots, free_spots, astral_which_spot, orig_length, 
        pull_force[0] + push_force[0], pull_force[1] + push_force[1], i)

    # Calculate force ratios (for reporting)
    pull_t = np.sum(pull_force[0]) + np.sum(pull_force[1])
    push_t = np.sum(push_force[0]) + np.sum(push_force[1])
    ratio = 100 * LA.norm(pull_t) / (LA.norm(pull_t) + LA.norm(push_t)) if (LA.norm(pull_t) + LA.norm(push_t)) > 0 else 0

    return cell, spots, spindle_poles, spindle_angle, astral_MTs, astral_angles, state, which_push, which_bind, free_spots, astral_which_spot, orig_length, df_list2, ratio, push_force, pull_force, v_c


def create_color_gradient(value):
    """
    Create a color gradient from green to white to red based on the provided value.
    The value should range from 0 to 1.
    """
    if value < 0:
        value = 0
    elif value > 1:
        value = 1

    if value <= 0.5:
        cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', ['green', 'white'])
        norm = mcolors.Normalize(vmin=0, vmax=0.5)
    else:
        cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', ['white', 'red'])
        norm = mcolors.Normalize(vmin=0.5, vmax=1)

    return cmap(norm(value))

def plot_cell(cell ,astral_MTs,astral_angles, state, spindle_poles,spindle_angle,push_force,pull_force,spots,i,which_bind, which_push, v_c,  n,ratio, params, borders):
    #plot_cell(cell,new_astral_MTs,new_state,new_poles,new_angle,push_force,pull_force,spots,i,new_which_bind,new_which_push,i, ratio,params)
    
    """The tricky part I sometimes forget about is that all the data in the legend such as total force, total pushing force and pulling force are for current iteration i, 
    but all the vectors (lines) are shown for future iteration. Basically, vectors predict the next step."""
    #cell ,astral_MTs, state, spindle_poles,spindle_angle,force1_next,force2_next,spots,i,which_bind, which_push=plot_list
    # Initialization
    tpoint=i
    a=params[0][0]
    b=params[0][1]
    r=LA.norm(spindle_poles[0]-spindle_poles[1])/2 #spindle length=2*r
    if (cell_type=='FE'):
        left_min = min(cell[:,0])-1.5*a#np.min([-2, 2*(np.min(center[:,0])-)])
        right_max =max(cell[:,0])+1.5*a#np.max([2, 2*(np.max(center[:,0])+)])
        top = max(cell[:,1])+0.5#1.5*b
        bottom =min(cell[:,1])-0.5# -1.5#*b
    elif (cell_type=='celegans'):
        left_min = min(cell[:,0])-.5*a#np.min([-2, 2*(np.min(center[:,0])-)])
        right_max =max(cell[:,0])+.5*a#np.max([2, 2*(np.max(center[:,0])+)])
        top = max(cell[:,1])+1.#.2#1.5*b
        bottom =min(cell[:,1])-1.# -1.5#*b
    else:
        left_min =borders[0]# min(cell[:,0])-1.5*a#np.min([-2, 2*(np.min(center[:,0])-)])
        right_max = borders[1] #max(forever_cell[:,0])+1.5*a#np.max([2, 2*(np.max(center[:,0])+)])
        top = borders[2]#max(forever_cell[:,1])+0.5#.2#1.5*b
        bottom =borders[3]#min(forever_cell[:,1])-0.5# -1.5#*b
        
        
    #save_name = kwargs.get('save_name', False)
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim([left_min,right_max])
    ax.set_ylim([bottom,top])

    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(True)
    #ax.set_facecolor('k')

    # Calculate force vectors for display
    
    count_pushing = np.sum(which_push == 1, axis=1)
    count_pulling = np.sum(which_bind == 1, axis=1)
    #Calculating vectors for next step to show given spindle and astral MTs position at the end of step (i) what force vectors look like during time step (i+1)
    #Calculate envelope repel force
    center=np.array([1/2*(spindle_poles[0,0]+spindle_poles[1,0]),1/2*(spindle_poles[0,1]+spindle_poles[1,1])])

    #Calculating repel force
    dist_to_cort=distance_matrix(spindle_poles,cell)
    repel_force1=np.array([0,0])
    repel_force2=np.array([0,0])
    
    if (np.min(dist_to_cort[0])<=1.5*config.min_cortex_dist):

        repel_vec1=-spindle_poles[0]/LA.norm(spindle_poles[0])
        repel_force1=(config.repel/np.min(dist_to_cort[0]))*repel_vec1

    if (np.min(dist_to_cort[1])<=1.5*config.min_cortex_dist):

        repel_vec2=-spindle_poles[1]/LA.norm(spindle_poles[1])
        repel_force2=(config.repel/np.min(dist_to_cort[1]))*repel_vec2

    
   
    
    # push_force, pull_force=find_force(astral_MTs,spindle_poles,which_bind,which_push, v_c)
    pull_t=np.sum(pull_force[0]+pull_force[1], axis=0)
    push_t=np.sum(push_force[0]+push_force[1], axis=0)

    force_vector_1=pull_force[0]+push_force[0]
    force_vector_2=pull_force[1]+push_force[1]
    force1=np.sum(force_vector_1, axis=0)+repel_force1
    force2=np.sum(force_vector_2, axis=0)+repel_force2
    # force1=force1#+repel_force1
    # force2=force2#+repel_force2  
    pull_1_t=np.sum(pull_force[0], axis=0)
    push_1_t=np.sum(push_force[0], axis=0)
    pull_2_t=np.sum(pull_force[1], axis=0)
    push_2_t=np.sum(push_force[1], axis=0)

    if(LA.norm(force1+force2)==0):
        f_net=np.array([0,0])
    else:
        f_net=(force1+force2)/LA.norm(force1+force2)
    #fnet is a normalized total force vector
        
    if((LA.norm(pull_t)+LA.norm(push_t))==0):
        ratio=0
    else:
        ratio=100*LA.norm(pull_t)/(LA.norm(pull_t)+LA.norm(push_t))
    # Sections force
    # sect_force_1=np.zeros((2,2))
    # sect_force_2=np.zeros((2,2))
    # sect_force_3=np.zeros((2,2))
    # for i in range (2):
    #     for j in range (len(astral_MTs[i])):
    #         if (spindle_angle-spread/2<=astral_angles[i,j]<=spindle_angle-spread/2+spread/3):
    #             #print(f'[{i,j}] is in region 1 {np.rad2deg(spindle_angle-spread/2),np.rad2deg(spindle_angle-spread/2+spread/3)}')
    #             sect_force_1[0]=sect_force_1[0]+pull_force[i,j]
    #             sect_force_1[1]=sect_force_1[1]+push_force[i,j]
    #         elif (spindle_angle-spread/2+spread/3<astral_angles[i,j]<=spindle_angle-spread/2+2*spread/3):
    #             #print(f'[{i,j}] is in region 2 {np.rad2deg(spindle_angle-spread/2+spread/3),np.rad2deg(spindle_angle-spread/2+2*spread/3)}')
    #             sect_force_2[0]=sect_force_2[0]+pull_force[i,j]
    #             sect_force_2[1]=sect_force_2[1]+push_force[i,j]
    #         elif (spindle_angle-spread/2+2*spread/3<astral_angles[i,j]<=spindle_angle+spread/2):
    #             #print(f'[{i,j}] is in region 3 {np.rad2deg(spindle_angle-spread/2+2*spread/3),np.rad2deg(spindle_angle+spread/2)}')
    #             sect_force_3[0]=sect_force_3[0]+pull_force[i,j]
    #             sect_force_3[1]=sect_force_3[1]+push_force[i,j]

    
    
    
    
    
    # ax.scatter([spindle_poles[0,0],spindle_poles[1,0]],[spindle_poles[0,1],spindle_poles[1,1]],color='w',label='Pull/Push = %.5f'%(ratio), s=100, edgecolors='k')
    # ax.plot([spindle_poles[0,0],spindle_poles[1,0]],[spindle_poles[0,1],spindle_poles[1,1]],color='m',linewidth=7)#,label='Pull/Push = %.5f'%(ratio))

    ax.plot(cell[:,0], cell[:,1], color='dimgrey',label='Step = %.1f'%(n), linewidth=5, zorder=10)
    # plt.fill(cell[:,0], cell[:,1], color='lavenderblush', edgecolor='black', linewidth=2)

    ax.scatter([spindle_poles[0,0],spindle_poles[1,0]],[spindle_poles[0,1],spindle_poles[1,1]],color='yellow', s=150, edgecolors='k',zorder=6, label=f'Spindle length = {20*r:.5f} μm')#'Pull/Push = %.5f'%(ratio))
    # ax.plot([spindle_poles[0,0],spindle_poles[1,0]],[spindle_poles[0,1],spindle_poles[1,1]],color='grey',linewidth=7,zorder=5)
    # CORTEX 
    
    theta = np.linspace(0, 2*np.pi, config.number_of_sides+1)[:-1].copy()
    cortex=np.zeros((config.number_of_sides,2))
    cortex[:,0] = 1.02*a*np.cos(theta)
    cortex[:,1] = 1.02*b*np.sin(theta)
    #ax.plot(cortex[:,0], cortex[:,1], color='crimson', linewidth=6,zorder=0)
    ax.plot([cell[0,0], cell[-1,0]], 
                [cell[0,1], cell[-1,1]], color = 'darkgrey',label=f'Angle ={math.degrees(spindle_angle):.2f}°')#,(math.degrees(spindle_angle)))#end of the cell lines

    # MOTORS
    
    ax.scatter(spots[:,0], spots[:,1], label='Number of astro MTs = %.1f,\n Number of spots= %.1f'%(2*len(astral_MTs[0]),len(spots)), color = 'salmon', s=55, edgecolors='dimgrey',marker="8", zorder=50)
    # ax.scatter(spots[:,0], spots[:,1], color='lightgrey', s=25, edgecolors='dimgrey',marker="8", zorder=50)
    
    # COLORCODING MTs
    for i in range (len(astral_MTs[0])):
        if (which_bind[0,i]==1):#binded
            ax.plot(astral_MTs[0,i,:,0],astral_MTs[0,i,:,1],'red', zorder=5)

        elif (state[0,i]==-1):#shrinking
            ax.plot(astral_MTs[0,i,:,0],astral_MTs[0,i,:,1],'tab:cyan', zorder=3)

        elif(which_push[0,i]==1):#pushing
            ax.plot(astral_MTs[0,i,:,0],astral_MTs[0,i,:,1],'darkgreen', zorder=4)

        else: #which_push=0, state=1,which_bind=0
            ax.plot(astral_MTs[0,i,:,0],astral_MTs[0,i,:,1],'slateblue', zorder=2)

    for i in range (len(astral_MTs[1])):  
        if (which_bind[1,i]==1):
            ax.plot(astral_MTs[1,i,:,0],astral_MTs[1,i,:,1],'tab:red', zorder=5)
        elif (state[1,i]==-1):
            ax.plot(astral_MTs[1,i,:,0],astral_MTs[1,i,:,1],'tab:cyan', zorder=3)
        elif(which_push[1,i]==1):
            ax.plot(astral_MTs[1,i,:,0],astral_MTs[1,i,:,1],'green', zorder=4)
        else:
            ax.plot(astral_MTs[1,i,:,0],astral_MTs[1,i,:,1],'slateblue', zorder=2)#'tab:purple')
   
    # Force vectors-------------------------------------------------------------------------------------------
    
    spindle_envelope=np.zeros((359,2))
    theta = np.linspace(0, 2*np.pi, 360)[:-1].copy()
    com=np.array([(spindle_poles[0,0]+spindle_poles[1,0])/2,(spindle_poles[0,1]+spindle_poles[1,1])/2])
    spindle_envelope[:,0]=r*np.cos(theta)+com[0]
    spindle_envelope[:,1]=w*np.sin(theta)+com[-1]
    spindle_envelope = rotate_points(spindle_envelope, spindle_angle)
    # if (cell_type=='celegans'):
    #     ax.plot(spindle_envelope[:,0], spindle_envelope[:,1], linewidth=7,color='grey', zorder=1)#,label='Spindle envelope')
    #     plt.fill(spindle_envelope[:,0], spindle_envelope[:,1], color='lightsteelblue', edgecolor='darkgrey', linewidth=2)

    # MEAN MTs LENGTH
    
    leng=np.zeros((2, len(state[0])))
    for i in range (2):
        for j in range (len(astral_MTs[0])):
            leng[i,j]=LA.norm([astral_MTs[i,j,-1,0]-spindle_poles[i,0],astral_MTs[i,j,-1,1]-spindle_poles[i,1]])
    mean_length=np.average(leng)
   
    # ANNOTATION
    
    if (config.show_vectors==1):
        kap=0.01
# #         ax.plot(spindle_envelope[:,0], spindle_envelope[:,1], color = 'r',label='Spindle envelope')
        # for j in range(len(spindle_poles)): #annotate poles
        #     ax.annotate(j+1,(spindle_poles[j,0],spindle_poles[j,1]))
        # for j in range(len(astral_MTs[0])):  # annotate astral MTs
        #     label = f"{j}({which_push[0,j]})"  # Creates label like "0(1)" or "1(-1)"
        #     ax.annotate(label, 
        #                 (astral_MTs[0,j,-1,0], astral_MTs[0,j,-1,1]))
        # for j in range(len(astral_MTs[1])):  # annotate astral MTs
        #     label = f"{j}({which_push[1,j]})"  # Creates label like "0(1)" or "1(-1)"
        #     ax.annotate(label, 
        #                 (astral_MTs[1,j,-1,0], astral_MTs[1,j,-1,1]))
        # for j in range (len(astral_MTs[0])): #annotate astral MTs
        #     ax.annotate(j, (astral_MTs[0,j,-1,0],astral_MTs[0,j,-1,1])) 
        # for j in range (len(astral_MTs[1])):
        #     ax.annotate(j, (astral_MTs[1,j,-1,0],astral_MTs[1,j,-1,1]))
        # for j in range(len(spots)): #annotate poles
        #     ax.annotate(j+1,(spots[j,0],spots[j,1]))
        # for j in range(len(cell)): #annotate poles
        #     if (j%100==0):
        #         ax.annotate(j,(cell[j,0],cell[j,1]))
        # EACH POLE SEPARATE PULL AND PUSH
        ax.plot([spindle_poles[0,0],spindle_poles[0,0]-kap*push_1_t[0]],[spindle_poles[0,1],spindle_poles[0,1]-kap*push_1_t[1]],'tab:red',linewidth=4,label='F_push 1 = %.2f'%(LA.norm(push_1_t)))
        ax.plot([spindle_poles[0,0],spindle_poles[0,0]+kap*pull_1_t[0]],[spindle_poles[0,1],spindle_poles[0,1]+kap*pull_1_t[1]],'tab:pink',linewidth=4,label='F_pull 1 = %.2f'%(LA.norm(pull_1_t)))
        ax.plot([spindle_poles[1,0],spindle_poles[1,0]-kap*push_2_t[0]],[spindle_poles[1,1],spindle_poles[1,1]-kap*push_2_t[1]],'tab:purple',linewidth=4,label='F_push 2 = %.2f'%(LA.norm(push_2_t)))
        ax.plot([spindle_poles[1,0],spindle_poles[1,0]+kap*pull_2_t[0]],[spindle_poles[1,1],spindle_poles[1,1]+kap*pull_2_t[1]],'tab:blue',linewidth=4,label='F_pull 2 = %.2f'%(LA.norm(pull_2_t)))
    
        ax.plot([spindle_poles[1,0],spindle_poles[1,0]-kap*repel_force1[0]],[spindle_poles[1,1],spindle_poles[1,1]-kap*repel_force1[1]],'indigo',linewidth=4,label='repel 1 = %.2f'%(LA.norm(repel_force1)))
        ax.plot([spindle_poles[1,0],spindle_poles[1,0]+kap*repel_force2[0]],[spindle_poles[1,1],spindle_poles[1,1]+kap*repel_force2[1]],'slategray',linewidth=4,label='repel 2 = %.2f'%(LA.norm(repel_force2)))
   
    else:
        kap=0
        fig.patch.set_visible(False)
        ax.axis('off')

    # NET FORCE EACH POLE
    
    # ax.plot([spindle_poles[0,0],spindle_poles[0,0]+kap*force1[0]],[spindle_poles[0,1],spindle_poles[0,1]+kap*force1[1]],'tab:orange',ls='--', linewidth=4, label=r'$F_{{\text{{pole1}}}} ={:.3f} \ pN $'.format(LA.norm(force1)),zorder=90)
    # ax.plot([spindle_poles[1,0],spindle_poles[1,0]+kap*force2[0]],[spindle_poles[1,1],spindle_poles[1,1]+kap*force2[1]],'k',ls='--', linewidth=4,label=r'$F_{{\text{{pole2}}}} ={:.3f} \ pN $'.format(LA.norm(force2)),zorder=90)

    # NET FORCE
    
    # ax.plot([com[0],com[0]+kap*f_net[0]],[com[1],com[1]+kap*f_net[1]],'k',linewidth=1,label='Total force = %.2f pN'%(LA.norm(force1+force2)))


   


    # TEXT
    

    if (cell_type=='endo' and c==3):
        plt.text(right_max-5.2, top-2.2, 'Mean astral MTs length = %.2f'%(mean_length), fontsize = 12)
        plt.text(right_max-5.2, top-2.3, 'Pull/Push = %.2f'%(ratio), fontsize = 12)
        plt.text(right_max-5.2, top-2.4, f'Pull,Push = {np.sum(count_pulling), np.sum(count_pushing)}', fontsize = 12)
        rect = patches.Rectangle((right_max-5.2, top-2.8), 0.3, 0.3, facecolor=create_color_gradient(ratio/100))
    elif(cell_type=='celegans'):

        plt.text(right_max-2.1, top-0.2, 'Mean astral MTs length = %.2f'%(mean_length), fontsize = 12)
        plt.text(right_max-2.1, top-0.3, 'Pull/Push = %.2f'%(ratio), fontsize = 12)
        plt.text(right_max-2.1, top-0.4, f'Pull,Push = {np.sum(count_pulling), np.sum(count_pushing)}', fontsize = 12)
        
        # plt.text(right_max-2.1, top-0.5, 'force1 pull=%.3f, force1 push=%.3f'%(LA.norm(sect_force_1[0]), LA.norm(sect_force_1[1])))
        # plt.text(right_max-2.1, top-0.6, 'force2 pull=%.3f, force2 push=%.3f'%(LA.norm(sect_force_2[0]), LA.norm(sect_force_2[1])))
        # plt.text(right_max-2.1, top-0.7, 'force3 pull=%.3f, force3 push=%.3f'%(LA.norm(sect_force_3[0]), LA.norm(sect_force_3[1])))
        rect = patches.Rectangle((right_max-1.2, top-1.1), 0.3, 0.3, facecolor=create_color_gradient(ratio/100))
    else:
        plt.text(right_max - 1, top - 0.3,  # Adjusted y-coordinate: top - 0.1 → top - 0.3
         f'Total force = {np.linalg.norm(LA.norm(force1+force2)):.3f}', 
         fontsize=10)
        #plt.text(right_max-0.9, top-0.3, 'Pull/Push = %.2f'%(ratio), fontsize = 10)  # Adjusted y-coordinate: top-0.3 → top-0.5

        plt.text(right_max-1, top-0.5, r'$F_{{\text{{pull}}}} / F_{{\text{{push}}}} = {:.3f}$'.format(ratio), fontsize=10)  # Adjusted y-coordinate: top-0.3 → top-0.5
        if (config.show_vectors==1):
            plt.text(right_max-1, top-0.4, r'$d_{{\text{{pole1}}}} = {:.3f}, d_{{\text{{pole2}}}} = {:.3f}$'.format(distance_to_boundary(spindle_poles[0], cell), distance_to_boundary(spindle_poles[1], cell)), fontsize=10)  # Adjusted y-coordinate: top-0.2 → top-0.4
       # 'Number of astro MTs = %.1f, Number of spots= %.1f'%(2*len(astral_MTs[0]),len(spots))
        variable1=np.sum(count_pulling)
        variable2=np.sum(count_pushing)
        plt.text(right_max-1, top-0.6, r'$N_{{\text{{pull}}}} = {}, N_{{\text{{push}}}} = {}$'.format(variable1, variable2), fontsize=10)  # Adjusted y-coordinate: top-0.4 → top-0.6
        # plt.text(right_max-1, top-0.3, f'angle={np.rad2deg(spindle_angle):.3f}, center={LA.norm(np.array([(spindle_poles[0,0]+spindle_poles[1,0])/2,(spindle_poles[0,1]+spindle_poles[1,1])/2])):.3f}')  # Adjusted y-coordinate: top-0.1 → top-0.3

        
        #plt.text(right_max-0.9, top-0.6, f'N_Pull/N_Push = {np.sum(count_pulling), np.sum(count_pushing)}', fontsize = 12)  # Adjusted y-coordinate: top-0.4 → top-0.6
        # plt.text(right_max-0.9, top-0.7, 'force1 pull=%.3f, force1 push=%.3f'%(LA.norm(sect_force_1[0]), LA.norm(sect_force_1[1])))  # Adjusted y-coordinate: top-0.5 → top-0.7
        # plt.text(right_max-0.9, top-0.8, 'force2 pull=%.3f, force2 push=%.3f'%(LA.norm(sect_force_2[0]), LA.norm(sect_force_2[1])))  # Adjusted y-coordinate: top-0.6 → top-0.8
        # plt.text(right_max-0.9, top-0.9, 'force3 pull=%.3f, force3 push=%.3f'%(LA.norm(sect_force_3[0]), LA.norm(sect_force_3[1])))  # Adjusted y-coordinate: top-0.7 → top-0.9
        rect = patches.Rectangle((right_max-0.7, top-1.0), 0.3, 0.3, facecolor=create_color_gradient(ratio/100))  # Adjusted y-coordinate: top-0.8 → top-1.0
    #label=r'$\sin (x)$'
    ax.add_patch(rect)
    # Parameters for the free body diagram
    if (config.show_vectors==3):
        rod_length = 0.6  # Length of the rod
        rod_angle = np.rad2deg(spindle_angle)  # Angle of the rod in degrees
        sphere_radius = 0.1  # Radius of the spheres at the ends of the rod
        
        # Convert angle to radians
        rod_angle_rad = np.radians(rod_angle)
        
        # Calculate the positions of the spheres
        center_x, center_y = 1.8, -0.2  # Center of the rod
        end1_x = center_x + (rod_length / 2) * np.cos(rod_angle_rad)
        end1_y = center_y + (rod_length / 2) * np.sin(rod_angle_rad)
        end2_x = center_x - (rod_length / 2) * np.cos(rod_angle_rad)
        end2_y = center_y - (rod_length / 2) * np.sin(rod_angle_rad)
        
        # Draw the rod
        rod = patches.FancyArrowPatch((end1_x, end1_y), (end2_x, end2_y), 
                                      arrowstyle='-', color='black', lw=2)
        ax.add_patch(rod)
        
        # Draw the spheres
        sphere1 = patches.Circle((end1_x, end1_y), sphere_radius, color='gold')
        sphere2 = patches.Circle((end2_x, end2_y), sphere_radius, color='gold')
        ax.add_patch(sphere1)
        ax.add_patch(sphere2)
        
        # Define the forces as NumPy arrays
        # Each row represents a force: [magnitude, angle in degrees]
        forces1 = np.array([
            push_1_t,   # Push1
            pull_1_t,  # Pull1
            repel_force1,  # Repel1
            force1    # Net1
        ])
        
        forces2 = np.array([
            push_2_t,   # Push1
            pull_2_t,  # Pull1
            repel_force2,  # Repel1
            force2    # Net1
        ])
        
        colors1 = ['tab:red', 'tab:pink', 'green', 'black']  # Colors for Push1, Pull1, Repel1, Net1
        colors2 = ['tab:purple', 'blue', 'green', 'black']  # Colors for Push2, Pull2, Repel2, Net2
        
        # Labels for the forces
        labels1 = ["Push1", "Pull1", "Repel1", "Net1"]
        labels2 = ["Push2", "Pull2", "Repel2", "Net2"]
        
        
        # Draw vectors for the first pole
        draw_vectors(ax, (end1_x, end1_y), forces1, labels1, colors1)
        
        # Draw vectors for the second pole
        draw_vectors(ax, (end2_x, end2_y), forces2, labels2, colors2)
    
    # CHROMOSOMES

    
    # if(cell_type!='celegans'):# and  tpoint*time_step<severance_time):
    if(cell_type=='celegans' and  tpoint*time_step<severance_time):
        chrom_angle=np.pi/2+spindle_angle
        n_chrom=4
        dd=0.7*w
        xs=np.linspace(com[0]-dd*np.cos(chrom_angle),com[0]+dd*np.cos(chrom_angle),n_chrom)
        ys=np.linspace(com[1]-dd*np.sin(chrom_angle),com[1]+dd*np.sin(chrom_angle),n_chrom)

        for i in range (n_chrom):
            line1,line2=chrom(xs[i],ys[i],chrom_angle)
            plt.plot([spindle_poles[0,0],xs[i]],[spindle_poles[0,1],ys[i]],color='g',linewidth=6)
            plt.plot([spindle_poles[1,0],xs[i]],[spindle_poles[1,1],ys[i]],color='g',linewidth=6)
            plt.plot(line1[0],line1[1],color='dodgerblue', linewidth=10)
            plt.plot(line2[0],line2[1], color='dodgerblue', linewidth=10)
    # elif (severance==1 and tpoint*time_step>=severance_time):
    #     plt.plot([0-1.2*w*np.cos(np.deg2rad(init_sp_angle)+np.pi/2),0+1.2*w*np.cos(np.deg2rad(init_sp_angle)+np.pi/2) ],[0-1.2*w*np.sin(np.deg2rad(init_sp_angle)+np.pi/2),0+1.2*w*np.sin(np.deg2rad(init_sp_angle)+np.pi/2)],color='r',linewidth=15)
            # plt.fill(
            # [spindle_poles[0, 0], spindle_poles[1, 0], xs[i], xs[i]],  # X-coordinates
            # [spindle_poles[0, 1], spindle_poles[1, 1], ys[i], ys[i]],  # Y-coordinates
            # color='mediumseagreen', 
            # alpha=1  # Transparency (adjust as needed)
            # )       
    # else:# (severance==1 and tpoint*time_step>=severance_time):
    #     # plt.plot([com[0]-1.2*w*np.cos(spindle_angle+np.pi/2),com[0]+1.2*w*np.cos(spindle_angle+np.pi/2) ],[com[1]-1.2*w*np.sin(spindle_angle+np.pi/2),com[1]+1.2*w*np.sin(spindle_angle+np.pi/2)],color='r',linewidth=15)
    #     plt.plot([0-1.2*w*np.cos(np.deg2rad(init_sp_angle)+np.pi/2),0+1.2*w*np.cos(np.deg2rad(init_sp_angle)+np.pi/2) ],[0-1.2*w*np.sin(np.deg2rad(init_sp_angle)+np.pi/2),0+1.2*w*np.sin(np.deg2rad(init_sp_angle)+np.pi/2)],color='r',linewidth=15)
          
    ax.legend(ncol=1,loc='upper left')

    # SAVING
    path=params[5]
    name=params[4]
    plt.savefig(path+'/'+name+'_' + str(1+n) + '.pdf', bbox_inches = 'tight', pad_inches = 1)
    plt.close(fig)
    # plt.show()

def draw_vectors(ax, origin, forces, labels, colors, normalized_length=0.2, text_offset=0.3):
    """
    Draw vectors defined by their endpoints.

    Parameters:
        ax: Matplotlib axis object.
        origin: The starting point of all vectors (e.g., [x, y]).
        forces: List of force vectors, where each vector is an endpoint [x, y].
        labels: List of labels for each vector.
        colors: List of colors for each vector.
        normalized_length: Desired length of the vectors (optional).
        text_offset: Offset for text labels to avoid overlap (optional).
    """
    for i, force_end in enumerate(forces):
        # Calculate the direction vector
        dx = force_end[0] - origin[0]
        dy = force_end[1] - origin[1]
        
        # Normalize the vector to the desired length
        if normalized_length is not None:
            magnitude = np.sqrt(dx**2 + dy**2)
            if magnitude > 0:  # Avoid division by zero
                scale = normalized_length / magnitude
                dx *= scale
                dy *= scale
        
        # Calculate the endpoint of the normalized vector
        end_x = origin[0] + dx
        end_y = origin[1] + dy
        
        # Draw the arrow
        ax.arrow(origin[0], origin[1], dx, dy, 
                 head_width=0.05, head_length=0.1, fc=colors[i], ec=colors[i])
        
        # Calculate text position with an offset
        text_x = end_x + dx * text_offset
        text_y = end_y + dy * text_offset
        
        # Add text label
        ax.text(text_x, text_y, f"{labels[i]}", fontsize=8, ha='center', color=colors[i])
def chrom(x,y, angle):
    if(cell_type=='FE'):
        rc=0.1
    else:
        rc=0.05

    line1=np.array([[x-rc*np.cos(angle-np.pi/4),x+rc*np.cos(angle-np.pi/4)],[y-rc*np.sin(angle-np.pi/4),y+rc*np.sin(angle-np.pi/4)]])
    line2=np.array([[x-rc*np.cos(angle+np.pi/4),x+rc*np.cos(angle+np.pi/4)],[y-rc*np.sin(angle+np.pi/4),y+rc*np.sin(angle+np.pi/4)]])
    return line1, line2

def rotate_and_shift_parabola(x, y, theta, xi, yi):
    # Rotation matrix
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
    
    # Stack x and y to create coordinate pairs
    coordinates = np.vstack((x, y))
    
    # Apply rotation
    rotated_coordinates = rotation_matrix @ coordinates
    
    # Shift the coordinates by (xi, yi)
    x_shifted = rotated_coordinates[0] + xi
    y_shifted = rotated_coordinates[1] + yi
    
    return x_shifted, y_shifted
    
def simulate(params):
    
    #before simulating create cell
    
    cell,astral_MTs,astral_angles,state,spindle_poles,spindle_angle,spots,which_push,which_bind,free_spots,astral_which_spot,orig_length,df_list2= make_cell(params)
    borders=[min(cell[:,0])-2, max(cell[:,0])+2, max(cell[:,1])+1, min(cell[:,1])-1 ]
    v_c=np.array([0,0])
    push_force, pull_force=find_force(astral_MTs,spindle_poles,which_bind,which_push, v_c)
    #plot intial cell and spindle
    plot_cell(cell,astral_MTs,astral_angles,state,spindle_poles,spindle_angle,push_force,pull_force,spots,-1,which_bind,which_push,v_c, -1,0, params, borders)
    #Dataframe
    df = pd.DataFrame(df_list2,columns=['Run','Number','Death', 'Switch','short' ,'bind status','Push','State','Force','Length','Orig length','end if out','astral angle'])#,'garb1','garb2','garb3'])
    #CUTOFF
    plot_list, spindle_center, RMS_angle_list, RMS_center_list=([] for i in range(4))
    i=0
    ratio_list=[]
    cutoff=0
    
    while(1>0):
        # print(f'{i}-----------------------=++++++++++++++++++++=')
        if (i>int(total_time/time_step)):
            cutoff=i
            break
        # MOVE SPINDLE
        if (i*time_step>severance_time and severance==1):
            new_cell, new_spots,new_poles,new_angle, new_astral_MTs,new_astral_angles,new_state,new_which_push, new_which_bind,new_free_spots,new_astral_which_spot,new_orig_length,df_list2, ratio, push_force, pull_force, new_v_c=move_severed_spindle(params,astral_MTs,astral_angles,state,spindle_poles,spindle_angle,cell,spots,which_push, which_bind,free_spots,astral_which_spot,orig_length,v_c,i)
        else: 
            new_cell, new_spots,new_poles,new_angle, new_astral_MTs,new_astral_angles,new_state,new_which_push, new_which_bind,new_free_spots,new_astral_which_spot,new_orig_length,df_list2, ratio, push_force, pull_force, new_v_c=move_spindle(params,astral_MTs,astral_angles, state,spindle_poles,spindle_angle,cell,spots,which_push, which_bind,free_spots,astral_which_spot,orig_length, v_c, i)
          
        new_center=np.array([1/2*(new_poles[0,0]+new_poles[1,0]),1/2*(new_poles[0,1]+new_poles[1,1])])
        # Check if spindle center is within 0.05 units of the cell center
       
        pull_t=np.sum(pull_force[0]+pull_force[1], axis=0)
        push_t=np.sum(push_force[0]+push_force[1], axis=0)

        for pole in new_poles:
            cell_polygon = Polygon(cell)
            point=pole
            if not cell_polygon.contains(Point(point)):
                # Debug: Print boundary violation
                # print(f"[DEBUG] POLE OUTSIDE at point: {point}")
                break
                return plot_list, df
                
        # print(f'spindle center={new_center}')
        # leng=np.zeros(np.shape(new_orig_length))
        # for q in range (2):
        #     for j in range (len(astral_MTs[0])):
        #         leng[q,j]=LA.norm([new_astral_MTs[q,j,-1,0]-new_poles[q,0],new_astral_MTs[q,j,-1,1]-new_poles[q,1]])
        # mean_length=np.average(leng)
        # DATA LIST
        if (severance==1):
            plot_list.append([cell,new_astral_MTs,new_state,new_poles,new_angle,new_poles[0],new_poles[1],spots,np.sum(new_which_bind == 1, axis=1),np.sum(new_which_push == 1, axis=1), 0, ratio, pull_t, push_t])
        else:
            plot_list.append([cell,new_astral_MTs,new_state,new_poles,new_angle,new_center,new_center[1],spots,np.sum(new_which_bind == 1, axis=1),np.sum(new_which_push == 1, axis=1), 0, ratio, pull_t, push_t])
        # 
        # PLOTTING
        #if (i%(int(1/time_step))==10):
        # if (i*time_step<(severance_time-5)):
        
        # if (i%20==0):

        save_interval_time = 1  # Save every 0.5 time units
        save_interval_steps = int(round(save_interval_time / time_step))  # Steps between saves
        if i % save_interval_steps == 0: 
        # print("saving pdf")
            plot_cell(new_cell,new_astral_MTs,new_astral_angles,new_state,new_poles,new_angle,push_force,pull_force,new_spots,i,new_which_bind,new_which_push, v_c,i, ratio,params, borders)
        # else:
        #     plot_cell(new_cell,new_astral_MTs,new_astral_angles,new_state,new_poles,new_angle,push_force,pull_force,new_spots,i,new_which_bind,new_which_push, v_c,i, ratio,params, borders)
            # print("saving pdf")
          
        cell, spots, astral_MTs, astral_angles, state, spindle_poles,spindle_angle,which_push, which_bind,free_spots,astral_which_spot,orig_length, v_c=new_cell, new_spots,new_astral_MTs,new_astral_angles, new_state, new_poles, new_angle,new_which_push,new_which_bind,new_free_spots,new_astral_which_spot,new_orig_length, new_v_c
        
        #Dataframes
        temporary_df=pd.DataFrame(df_list2,columns=['Run','Number','Death', 'Switch','short', 'bind status','Push','State','Force','Length','Orig length','end if out','astral angle'])#,'garb1','garb2','garb3'])
        df=pd.concat([df,temporary_df])
        
        # if (LA.norm(new_center-np.array([0,0]))>0.8):
        #     # Debug: Print center violation
        #     # print(f"[DEBUG] SPINDLE CENTER OUTSIDE at center: {new_center}")
        #     break
        #     return plot_list, df

        i+=1

    return plot_list, df#, RMS_angle_list, RMS_center_list, cutoff, final_RMS_angle, ratio_list

def plot_simulate(params):
    #pl_list, df, RMS_angle, RMS_center, cutoff, final_RMS_angle, ratio_list = simulate(params)#pl_list
    pl_list, df = simulate(params)#pl_list
    if task_id==1:
        file_name=params[4]+'.xlsx'
        file_path = os.path.join(params[5], file_name)
        df.head(10000).to_excel(file_path, index=False)#, columns =['Number','Death', 'Switch', 'Push','State','Length'],index=False, header=False)
 
    # EXTRACTING DATA
    df_angle = [np.rad2deg(row[4]) for row in pl_list]
    df_angle=flip_angles(df_angle)
    df_center=[row[5] for row in pl_list]
    df_ypos=[row[6] for row in pl_list]
    df_count_push=[np.sum(row[9]) for row in pl_list]
    df_count_pull=[np.sum(row[8]) for row in pl_list]
    df_ratio=[row[11] for row in pl_list]
    df_pull_t=[row[12] for row in pl_list]
    df_push_t=[row[13] for row in pl_list]
    # df_length=[row[10] for row in pl_list]
    return df_angle, df_count_push, df_count_pull, df_center,df_ypos, df_ratio, df_pull_t, df_push_t #, df_count_push, df_count_pull, df_length#, RMS_angle, RMS_center, cutoff, ratio_list

def flip_angles(df_angle):

    if(df_angle[-1]>90):
        for i in range(len(df_angle)):
            dummy=df_angle[i]
            df_angle[i]=180-dummy
    elif(df_angle[-1]<-90):
        for i in range(len(df_angle)):
            dummy=df_angle[i]
            df_angle[i]=180+dummy
    return df_angle

def flip_angles_NB(df_angle):

    if(df_angle[1]>160):
        for i in range(len(df_angle)):
            dummy=df_angle[i]
            df_angle[i]=180-dummy
    elif(df_angle[-1]<-90):
        for i in range(len(df_angle)):
            dummy=df_angle[i]
            df_angle[i]=180+dummy
    return df_angle

def extract_parameters(input_string):
    # Regular expressions to extract specific parameters
    motor_density_match = re.search(r'MUD_(\d+)', input_string)
    astral_MTs_match = re.search(r'MT_(\d+)', input_string)
    push_match = re.search(r'push_([\d.]+)', input_string)
    pull_match = re.search(r'pull_([\d.]+)', input_string)
    ts_match = re.search(r'ts_([\d.]+)', input_string)
    cell_match = re.search(r'cell_([A-Za-z]+)', input_string)
    length_MTs_match = re.search(r'AL_([A-Za-z]+)', input_string)
    state_MTs_match = re.search(r'state_([A-Za-z]+)', input_string)
    # repel_match = re.search(r'repel_([\d.]+)', input_string)
    angle_match = re.search(r'angle_([\d.]+)', input_string)
    SL_match = re.search(r'SL_([\d.]+)', input_string)
    AL_match = re.search(r'AL_([\d.]+)', input_string)
    mcd_match = re.search(r'mcd_([\d.]+)', input_string)

    if cell_match:
        cell_type = cell_match.group(1)  # Extract the matched string ("FE")
        print("Cell type found:", cell_type)  # Output: Cell type found: FE
    else:
        print("No cell type found.")
    
    if length_MTs_match:
        length_MTs = length_MTs_match.group(1)
    else:
        print("No length_MTs found.")
        length_MTs = "default_length"
    if state_MTs_match:
        state_MTs = state_MTs_match.group(1)
    else:
        print("No state_MTs found.")
        state_MTs = "default_state"
    # Extract values or set to None if not found
    motor_density = int(motor_density_match.group(1)) if motor_density_match else None
    astral_MTs = int(astral_MTs_match.group(1)) if astral_MTs_match else None
    push = float(push_match.group(1)) if push_match else None
    pull = float(pull_match.group(1)) if pull_match else None
    # repel = float(repel_match.group(1)) if repel_match else None
    angle = float(angle_match.group(1)) if angle_match else None
    SL = float(SL_match.group(1)) if SL_match else None
    AL = float(AL_match.group(1)) if AL_match else None
    mcd = float(mcd_match.group(1)) if mcd_match else None
    time_step = float(ts_match.group(1)) if ts_match else None

    # Return as a dictionary
    return motor_density, astral_MTs, push, pull, SL, time_step, cell_type, angle


def find_number_after_cell(input_string):
    # Regular expression to find a number after "cell"
    match = re.search(r'endo_(\d+)', input_string)
    if match:
        number = match.group(1)
    return int(number)

def create_simulation_directory(test_folder_path, name, task_id ):
    folder_name = f"{name}_{task_id}"
    new_dir_path = os.path.join(test_folder_path, folder_name)
    os.mkdir(new_dir_path)
    return folder_name, new_dir_path

class SimulationParameters:
    def __init__(self):
        # Model parameters
        self.number_of_sides = 480
        self.length_MTs ='gamma'
        
        self.state_MTs = 'random'

        self.max_interact_dist = 0.02

        self.push_dist=0.0005#1e-15#0.02
        self.min_cortex_dist = 0.025
        self.MT_min_length = 0.025
        self.MT_max_length = 10
        
        self.natural_spacing = 2 * np.pi / self.number_of_sides
        self.elongate_limit = 0.5
        self.repel_dist = 0.005
        self.discr = 2
        self.astro_lin_density = 1 / 5
        self.max_slip = np.pi / 6

        # Experiment parameters
        self.catastr_rate = 0.014
        self.rescue_rate = 0.044#29
        self.shrink_rate = -18 / 600
        self.growth_rate = 9 / 600

        # Motor protein binding/unbinding
        self.dyn_bind = 0.03
        self.dyn_unbind = 0.02

        # Force
        self.pull=pull
        self.push=push
        self.repel =  0
        # Rigidity & Viscosity
        self.EI = 20 * 0.1 * 0.1
        self.visc = 100
        self.mu_fric = 500

        # Features
        self.buckling = 0
        self.slipping = 0
        self.pivoting = 0
        self.mobile_motors = 0
        self.force_velocity = 0
        self.v_0 = 0.086
        self.show_vectors = 0
        self.uniform_status = 0
        self.top_status = 1

    def save_parameters_to_file(self, directory, filename, additional_params=None):
        """
        Save all parameters of the class and additional parameters to a text file.
        
        Args:
            directory (str): The directory where the file should be saved.
            filename (str): The name of the file to save the parameters to.
            additional_params (dict): Additional parameters to save, as a dictionary.
        """
        # Create the full file path
        filepath = os.path.join(directory, filename)
        
        # Check if the file already exists
        if os.path.exists(filepath):
            print(f"File '{filepath}' already exists. Skipping save.")
            return
        
        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save the parameters to the file
        with open(filepath, 'w') as file:
            # Write class attributes
            for attr, value in self.__dict__.items():
                file.write(f"{attr}: {value}\n")
            
            # Write additional parameters
            if additional_params:
                for key, value in additional_params.items():
                    file.write(f"{key}: {value}\n")
        
        print(f"Parameters saved to '{filepath}'.")



if __name__ == "__main__":
    
    test_folder_path = sys.argv[1]
    data_folder_path = sys.argv[2]
    name=sys.argv[3] #"spindle_9214_FE_AL_1_0.3_SL_1.8_ts_0.05_opt_forces_3.6_4_2.5_MUD_PINS_10_MT_50_push_1"
    #"spindle_9214_endo_cell_1_AL_1_0.3_SL_1.8_ts_0.05_opt_forces_3.6_4_2.5_MUD_PINS_10_MT_50_push_1"
    
    print(f"Running simulation with name: {name}")
    task_id =int(os.getenv('SLURM_ARRAY_TASK_ID'))
    job_id = os.getenv('SLURM_JOB_ID')
    motor_density, astral_number, push, pull, SL, time_step, cell_type, init_sp_angle = extract_parameters(name)    

    # SL=1.8 
    # motor_density, astral_number, push_status, time_step, cell_type = 40, 50, 1, 0.05, "FE"
    # if push_status==1:
    #     push=4
    # else:
    #     push=0
    # push=push_status
    # Model parameters
    
    config = SimulationParameters()
    

    
    # Experiment probabilities
    prob_catastr=1-np.exp(-config.catastr_rate*time_step)
    prob_rescue=1-np.exp(-config.rescue_rate*time_step)

    if pull!=0:
        prob_dyn_bind =1 - np.exp(-config.dyn_bind * time_step)
        prob_dyn_unbind = 1 - np.exp(-config.dyn_unbind * time_step)
    else:
        prob_dyn_bind =0#1 - np.exp(-config.dyn_bind * time_step)
        prob_dyn_unbind = 1# - np.exp(-config.dyn_unbind * time_step)
        

    severance_time=2500
    severance=0
    
    if cell_type == "celegans":
        
        total_time = 200
        # SL=1
        r=SL/2
        a, b, r, w = 2.5, 1.5, 0.5, 0.5 #1, 0.4
        # a, b , w = 2.5, 1.5, 0.5
        spread = np.deg2rad(270)# * np.pi / 22 * np.pi / 3
        mean, stdev = 1, 0.3
        AL,scale= 9, 0.167

        sp_angle=np.deg2rad(init_sp_angle)*np.ones(50)
    elif cell_type == "endo":
        c = find_number_after_cell(name)
        df_data=pd.read_excel('./Movie_info.xlsx',sheet_name='Alikhan', index_col=False)
        # df_data=pd.read_excel('/Users/ayn6k/Downloads/Movie_info.xlsx',sheet_name='Alikhan', index_col=False)
        frame_rates=df_data['Frame rate']#
        ends=df_data['anaphase onset']
        starts=df_data['metaphase']
        rescale=df_data['model scale factor']
        initial_angles=df_data['Initial angle']
        final_angle=df_data['Final angle']
        junc_spread=df_data['delta']
        junc_initial_angle=df_data['Junction angle initial']
        junc_final_angle=df_data['Junction angle final']
        spindle_length_df=df_data['Spindle length']
        spindle_width_df=df_data['Spindle width']
        total_time=(ends[c-1]-starts[c-1])*frame_rates[c-1]
        a, b = 1, 1
        spread = 2 * np.pi / 2
        r = 0.1 * spindle_length_df[c - 1] / 2
        w = 0.1 * spindle_width_df[c - 1] / 2
        AL,scale=1, 1
        sp_angle=np.deg2rad(initial_angles[c-1])*np.ones(50)
    elif cell_type == "FE":
        total_time=600
        a, b = 1, 1
        r=SL/2
        spread = 3 * np.pi / 2
        w = 0.8*r
        AL,scale=1, 1
        sp_angle=np.deg2rad(init_sp_angle)*np.ones(50)
    else:
        raise ValueError(f"Unknown cell type: {cell_type}")

    folder_name, new_dir_path = create_simulation_directory(test_folder_path, name, task_id )
    
    additional_params = {
        "job_id":job_id,
        "total_time": total_time,
        "a": a,
        "b": b,
        "r": r,
        "spread": spread,
        "w": w,
        "mean MT length": AL,
        "stdev MT length": scale,
        "sp_angle": sp_angle[0],
        "push": push,
        "length_MTs": config.length_MTs,
        "state_MTs": config.state_MTs,
        "extranote": "regular", #more FGs apically and basally gamma 1,1
    }
    # config.save_parameters_to_file(data_folder_path, name+"_parameters_"+str(task_id)+".txt", additional_params)    
    config.save_parameters_to_file(data_folder_path, name+"_parameters.txt", additional_params)   

    """
    list_params.append([1-[a,b,r-spindle semi-long axis length,w-spindle semiwidth], 
                        2-FGs,
                        3-A_MTs,
                        4-sp_angle,
                        5-folder_name, 
                        6-path to the run folder])
    """
    
    #Run simulation
    
    df_angle, df_count_push, df_count_pull, df_center,df_ypos, df_ratio, df_pull_t, df_push_t=plot_simulate([[a,b,r,w],motor_density,astral_number,sp_angle[int(task_id-1)],  folder_name, new_dir_path])
    
    #Save data
    data_dict = {
        "Angle": df_angle,
        "Center": df_center,
        "Y-pos": df_ypos,
        "N_pull": df_count_pull,
        "N_push": df_count_push,
        "Ratio": df_ratio,
        "Pull_t": df_pull_t,
        "Push_t": df_push_t
    }
    excel_file_name = f"{folder_name}_data.xlsx"
    excel_file_path = os.path.join(data_folder_path, excel_file_name)
    with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
        for sheet_name, data in data_dict.items():
            pd.DataFrame({"Run " + str(task_id): data}).to_excel(writer, sheet_name=sheet_name, index=False)
            
