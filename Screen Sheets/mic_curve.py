# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 01:52:56 2023

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
def read_value_from_excel(filename, column, row):
    return pd.read_excel(filename, skiprows=row - 1, usecols=column, nrows=1, header=None, names=["Value"]).iloc[0]["Value"]

# Normalizes a set of data using a control
def normalize(data, control):
    norm_factors = control.mean(axis=1).to_frame()
    #print(norm_factors)
    for i in range(1, len(data.columns)):
        norm_factors[i] = norm_factors.loc[:, 0]
    normed = np.subtract(data, np.asarray(norm_factors))
    normed[normed < 0] = 0
    return normed

def mic_from(file):
    # Establish timestamps from number of reads (listed in sheet)
    read_string = read_value_from_excel(file, "B", 18)
    x_count = int(read_string[47:-6])
    x_vals = np.arange(1 / 12, (x_count + 1) / 12, 1 / 12)
    
    init_conc = 10;
    
    # Establish dilution values
    dilutions = [init_conc]
    for i in range(1, 5):
        dilutions.append(dilutions[i - 1] / 2)
    dilutions.append(0)
    strain_conc_rs = [];
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
        species = pd.read_excel(file, usecols=spec_cols, index_col=0, header=0, skiprows=range(28), nrows=x_count)
        control = pd.read_excel(file, usecols=control_cols, index_col=0, header=0, skiprows=range(28), nrows=x_count)
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
            conc_k[conc] = k_guess
            try:
                est_func, pcov = curve_fit(logistic, x_vals, conc_data.values, p0=[r_guess, k_guess, x_0_guess])
                conc_r[conc] = est_func[0]
                # Obtain fitted data
            # Assume the sample showed no significant growth
            except RuntimeError:
                conc_r[conc] = 0
        strain_conc_rs.append(conc_r)
    return strain_conc_rs

def generate_mic_curve(vals, ax):
    conc = [10.0, 5.0, 2.5, 1.25, 0.625, 0]
    for antibiotic in vals:
        ax.plot(conc, vals[antibiotic], label=antibiotic)

filename = input("Enter txt file with filename list: ")
strain_1 = input("Enter first strain: ")
strain_2 = input("Enter second strain: ")

fig_1, ax_1 = plt.subplots(1,1,figsize=(10,10), sharex=True, sharey=True) 
plt.xscale("log")
ax_1.set_xlabel("Antibiotic Concentration (ug/mL)")
ax_1.set_ylabel("Growth Rate")
ax_1.set(title=strain_1 + " Growth Rate Dependence on Antibiotic Concentration (ug/mL)")
fig_2, ax_2 = plt.subplots(1,1,figsize=(10,10), sharex=True, sharey=True) 
plt.xscale("log")
ax_2.set_xlabel("Antibiotic Concentration (ug/mL)")
ax_2.set_ylabel("Growth Rate")
ax_2.set(title=strain_2 + " Growth Rate Dependence on Antibiotic Concentration (ug/mL)")
first = {}
second = {}

with open(filename, 'r', encoding='utf-8-sig') as files:
    for line in files:
        # Creates list of each field in the line
        info = line.split()
        file = info[0]
        # Searches connection file for connections from this Node and adds
        # them to a list
        concs = mic_from(file)
        if (strain_1 == info[2] and strain_2 == info[3]):
            first[info[1]] = concs[0].values()
            second[info[1]] = concs[1].values()
        elif (strain_1 == info[3] and strain_2 == info[2]):
            first[info[1]] = concs[1].values()
            second[info[1]] = concs[0].values()
        else:
            print("Error: mismatched strain names")
generate_mic_curve(first, ax_1)
generate_mic_curve(second, ax_2)
ax_2.legend(loc='lower right')
ax_1.legend(loc='lower right')
if input("Save " + strain_1 + " MIC graph? y/n: ") == "y":
    fig_1.savefig(strain_1 + "MIC.png")
if input("Save " + strain_2 + " MIC graph? y/n: ") == "y":
    fig_2.savefig(strain_2 + "MIC.png")
        