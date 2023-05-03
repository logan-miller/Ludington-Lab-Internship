# -*- coding: utf-8 -*-
"""
Created on Wed May  3 00:48:56 2023

@author: bubba
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()

# Logistic growth model
def logistic(x, r, k, x_0):
    f = k / (1 + np.exp(-r * (x - x_0)))
    return f

# Linear model
def no_growth(x, m, b):
    y = (m * x) + b
    return y

# Estimate growth rate by finding maximum symmetric difference (slope)
def find_r(data):
    max_diff = 0
    for i in range(1, len(data) - 1):
        diff = (data[i + 1] - data[i - 1]) * 6
        if(diff > max_diff):
            max_diff = diff
    return max_diff

# Finds closest value in a list to a target value (closest OD to likely sigmoid midpoint)
def find_x_0(times, data, n):
    sig_mid = np.max(times)
    sig_mid_val = np.max(data)
    for i in range(len(times)):
        if(np.abs(data[i] - n) < sig_mid_val):
            sig_mid_val = np.abs(data[i] - n)
            sig_mid = times[i]
    if (sig_mid_val == np.max(data)):
        sig_mid = 0
    return sig_mid

# Reads in a single value from an Excel sheet
def find_read_count(filename, column):
    col = pd.read_excel(filename, usecols=column, header=None).values.tolist()
    found_start = False
    reads = 0
    skips = 0
    for val in col:
        if found_start:
            if val[0] != val[0]:
                break;
            reads += 1
        elif val[0] == 'A1':
            found_start = True
        else:
            skips += 1
    return reads, skips

# Normalizes a set of data using a control
def normalize(data, control):
    norm_factors = control.mean(axis=1).to_frame()
    #print(norm_factors)
    for i in range(1, len(data.columns)):
        norm_factors[i] = norm_factors.loc[:, 0]
    normed = np.subtract(data, np.asarray(norm_factors))
    normed[normed < 0] = 0
    return normed

def print_MIC50(file, antibiotic, strains, init_conc):
    
    # Establish timestamps from number of reads (listed in sheet)
    x_count, skip_count = find_read_count(file, 'D')
    x_vals = np.arange(1 / 12, (x_count + 1) / 12, 1 / 12)
    
    dilutions = [float(init_conc)]
    for i in range(1, 5):
        dilutions.append(dilutions[i - 1] / 2)
    dilutions.append(0)
    
    # For each of the two species
    for num in [0, 1]:
        displacement = num * 6
        
        # Establish the specific columns for data and control
        spec_cols = [1]
        for i in range(0, 84, 12):
            for j in range(6):
                spec_cols.append(i + j + 3 + displacement)       
        control_cols = [1]
        for i in range(87, 93):
            control_cols.append(i + displacement)
            
        # Initialize and normalize data
        species = pd.read_excel(file, usecols=spec_cols, index_col=0, header=0, skiprows=range(skip_count), nrows=x_count)
        control = pd.read_excel(file, usecols=control_cols, index_col=0, header=0, skiprows=range(skip_count), nrows=x_count)
        species = normalize(species, control)
        
        
        conc_r = {}
        conc_k = {}
        # For each dilution
        for conc in dilutions:
            
            # Find average for the dilution
            conc_cols = []
            for i in range(0, len(species.columns), 6):
                conc_cols.append(i + dilutions.index(conc))
            replicates = species.iloc[:, conc_cols]
            conc_data = replicates.mean(axis=1)
            
            # Guess logistic parameters
            k_guess = conc_data.max()
            r_guess = find_r(conc_data.values)
            x_0_guess = find_x_0(x_vals, conc_data.values, 0.1)
            
            # Implement curve fitting
            try:
                est_func, pcov = curve_fit(logistic, x_vals, conc_data.values, p0=[r_guess, k_guess, x_0_guess])
                conc_r[conc] = est_func[0]
                conc_k[conc] = est_func[1]
            # Assume the sample showed no significant growth
            except RuntimeError:
                est_func, pcov = curve_fit(no_growth, x_vals, conc_data.values, p0=[0, 0])
                conc_r[conc] = 0
                conc_k[conc] = k_guess
        mic_r = find_mic(conc_r)
        mic_k = find_mic(conc_k)
        #print(antibiotic + "+" + strains[num] + " GR MIC-50: " + str(mic_r))
        print(antibiotic + "+" + strains[num] + " CC MIC-50: " + str(mic_k))

def find_mic(concs):
    halved = concs[0] / 2;
    closest = max(concs.values())
    mic_50 = 0;
    for conc in concs:
        if abs(halved - concs[conc]) < closest:
            closest = halved - concs[conc]
            mic_50 = conc
    return mic_50

filename = input("Enter txt file with filename list: ")
with open(filename, 'r', encoding='utf-8-sig') as files:
    for line in files:
        # Creates list of each field in the line
        info = line.split()
        print_MIC50(info[0], info[1], [info[2], info[3]], info[4])