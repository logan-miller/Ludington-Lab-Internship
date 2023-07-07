# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:36:10 2023

@author: bubba
"""


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
import re
sns.set_theme()

SPEC_COLORS = {"": "tab:blue",
               "Lp": "tab:orange",
               "Lb": "tab:green",
               "LpLb": "tab:red"}
TRIP_COLORS = {"AoLpLb": "tab:blue",
               "AmLpLb": "tab:orange",
               "AsLpLb": "tab:green"}

# Count number of reads in an experiment
def find_read_count(filename, column):
    # Pull single column from sheet
    col = pd.read_excel(filename, usecols=column, header=None).values.tolist()
    found_start = False
    reads = 0
    skips = 0
    for val in col: # For every row in column
        if found_start: # Count reads after A1 column label
            if val[0] != val[0]: break; # If NaN (end of reads)
            reads += 1
        elif val[0] == 'A1': found_start = True # found A1 column label
        else: skips += 1
    return reads, skips

def get_section(start, end, df): return df.iloc[:, range(start, end)]

def groups(file):
    x_count, skip_count = find_read_count(file, 'D')
    x_vals = np.arange(1 / 12, (x_count + 1) / 12, 1 / 12)
    specs = ["", "Ao", "Am", "As"]
    master_df = pd.read_excel(file, usecols=range(2, 99), index_col=0, 
                              header=0, skiprows=range(skip_count), 
                              nrows=x_count)
    sections = {}
    jump = 0
    for spec in specs:
        sections[spec] = get_section(jump, 6 + jump, master_df)

        sections[spec + "Lp"] = get_section(6 + jump, 12 + jump, master_df)
        sections[spec + "Lb"] = get_section(12 + jump, 18 + jump, master_df)
        sections[spec + "Lp" + "Lb"] = get_section(18 + jump, 24 + jump, 
                                                   master_df)
        jump += 24
    return sections, x_vals

def normalize(data, control):
    norm_factors = control.mean(axis=1).to_frame()
    for i in range(1, len(data.columns)): 
        # Make control array the same size as the data array
        norm_factors[i] = norm_factors.loc[:, 0]
    normed = np.subtract(data, np.asarray(norm_factors))
    normed[normed < 0] = 0 # Adjust neg values to 0
    return normed

# Logistic growth model
def logistic(x, r, k, x_0): return k / (1 + np.exp(-r * (x - x_0)))

# Linear model
def no_growth(x, m, b): return (m * x) + b

# Estimate growth rate by finding maximum symmetric difference (slope)
def find_r(data):
    max_diff = 0
    for i in range(1, len(data) - 1):
        diff = (data[i + 1] - data[i - 1]) * 6 # Symmetric difference
        if(diff > max_diff): max_diff = diff
    return max_diff

# Find closest value in a list to a target value n
# (closest OD to likely sigmoid midpoint)
def find_x_0(times, data, n):
    sig_mid = np.max(times)
    sig_mid_val = np.max(data)
    for i in range(len(times)):
        if(np.abs(data[i] - n) < sig_mid_val):
            sig_mid_val = np.abs(data[i] - n)
            sig_mid = times[i]
    if (sig_mid_val == np.max(data)): sig_mid = 0
    return sig_mid

def growth_curves(data, x_vals):
    curves = {}
    for section in data:
        run = data[section].mean(axis=1)
        k_guess = run.max()
        r_guess = find_r(run.values)
        x_0_guess = find_x_0(x_vals, run.values, 0.1)
        
        # Implement curve fitting
        try:
            est_func, pcov = curve_fit(logistic, x_vals, run.values, 
                                       p0=[r_guess, k_guess, x_0_guess])
            curves[section] = logistic(x_vals, *est_func)
        # Assume the sample showed no significant growth
        except RuntimeError:
            est_func, pcov = curve_fit(no_growth, x_vals, run.values, 
                                       p0=[0, 0])
            curves[section] = no_growth(x_vals, *est_func)
    return curves

def init_graph(name, x_vals, controls, lp, lb, lplb, null):
    fig, ax = plt.subplots(1,1,figsize=(15,15), sharex=True, 
                           sharey=True)
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("OD600")
    ax.set(title="Growth of " + name[0:2] + " Cocultures")
    ax.plot(x_vals, controls[0], label="Control Mean", c="tab:olive",
            linestyle='dashed')
    ax.plot(x_vals, controls[1], label="Control Fit", c="tab:olive")
    ax.plot(x_vals, lp[0], label="Lp Mean", c="tab:purple",
            linestyle='dashed')
    ax.plot(x_vals, lp[1], label="Lp Fit", c="tab:purple")
    ax.plot(x_vals, lb[0], label="Lb Mean", c="tab:pink", linestyle='dashed')
    ax.plot(x_vals, lb[1], label="Lb Fit", c="tab:pink")
    ax.plot(x_vals, lplb[0], label="LpLb Mean", c="tab:brown",
            linestyle='dashed')
    ax.plot(x_vals, lplb[1], label="LpLb Fit", c="tab:brown")
    ax.plot(x_vals, null[0], label="Null Model", c="tab:gray",
            linestyle='dashed')
    ax.plot(x_vals, null[1], label="Null Model Fit", c="tab:gray")
    return fig, ax

def null_model(triple, data, x_vals):
    indivs = re.findall("..", triple)
    pairs = itertools.combinations(indivs, 2)
    null = [0] * len(x_vals)
    for indiv in indivs:
        null = np.subtract(null, data[indiv].mean(axis=1))
    for pair in pairs:
        null = np.add(null, data["".join(pair)].mean(axis=1))
    k_guess = null.max()
    r_guess = find_r(null.values)
    x_0_guess = find_x_0(x_vals, null.values, 0.1)
    
    try:
        est_func, pcov = curve_fit(logistic, x_vals, null.values, 
                                   p0=[r_guess, k_guess, x_0_guess])
        est_curve = logistic(x_vals, *est_func)
    except RuntimeError:
        est_func, pcov = curve_fit(no_growth, x_vals, null.values, 
                                   p0=[0, 0])
        est_curve = no_growth(x_vals, *est_func)
    return null, est_curve, est_func

def graph_curves(data, curves, x_vals):
    plots = {}
    null_params = {}
    controls = (data[""].mean(axis=1), curves[""])
    lp = (data["Lp"].mean(axis=1), curves["Lp"])
    lb = (data["Lb"].mean(axis=1), curves["Lb"])
    lplb = (data["LpLb"].mean(axis=1), curves["LpLb"])
    for curve in curves:
        spec = curve[0:2]
        if curve != "" and curve[0] == 'A':
            if spec in plots:
                fig = plots[spec][0]
                ax = plots[spec][1]
            else:
                null = null_model(spec + "LpLb", data, x_vals)
                null_params[spec + "LpLb"] = null[2]
                fig, ax = init_graph(spec, x_vals, controls, 
                                          lp, lb, lplb, (null[0], null[1]))
                plots[spec] = (fig, ax)
            ax.plot(x_vals, curves[curve], label=curve + " Fit", 
                    c=SPEC_COLORS[curve[2:]])
            ax.plot(x_vals, data[curve].mean(axis=1), label=curve + " Mean", 
                    c=SPEC_COLORS[curve[2:]], linestyle='dashed')
            ax.legend(loc=2)
            ax.autoscale()
            ax.set_ylim(bottom=0)
    return null_params



def find_med(data, x_vals):
    param_dict = {}
    for section in data:
        param_list = []
        for i in range(len(data[section].columns)):
            run = data[section].iloc[:, i]
            
            k_guess = run.max()
            r_guess = find_r(run.values)
            x_0_guess = find_x_0(x_vals, run.values, 0.1)
            
            try:
                est_func, pcov = curve_fit(logistic, x_vals, run.values, 
                                           p0=[r_guess, k_guess, x_0_guess])
                param_list = est_func
            # Assume the sample showed no significant growth
            except RuntimeError:
                est_func, pcov = curve_fit(no_growth, x_vals, run.values, 
                                           p0=[0, 0])
                param_list = [0, k_guess, max(x_vals)]
        param_dict[section] = (np.median(param_list[0]), 
                               np.median(param_list[1]), 
                               np.median(param_list[2]))
    return param_dict
            
def extract(data, index):
    extracted = {}
    for entry in data: extracted[entry] = data[entry][index]
    return extracted

def bar_charts(title, unit, null, base):
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))
    data = list(base.keys())
    
    br1 = np.arange(len(data))
    br2 = [x + barWidth for x in br1]
    
    plt.bar(br2, list(null.values()), color='tab:gray', 
            width=barWidth, edgecolor='gray', label='Null Model')
    plt.bar(br1, list(base.values()), color='g', width=barWidth, 
            edgecolor='gray', label="Actual")
    plt.title(title + " of Cocultures", fontweight ='bold', fontsize = 20)
    plt.xlabel('Coculture', fontweight ='bold', fontsize = 15)
    plt.ylabel(title + unit, fontweight ='bold', fontsize = 15)
    plt.xticks([r + (0.5 * barWidth) for r in range(len(data))], data)
    plt.legend()
    plt.show()

def main():
    filename = input("Enter coculture sheet filename: ")
    sections, xs = groups(filename)
    normalized = {}
    for section in sections: 
        normalized[section] = normalize(sections[section], sections[""])
    curves = growth_curves(normalized, xs)
    null_params = graph_curves(normalized, curves, xs)
    meds = find_med(sections, xs)
    effs = {}
    params = {}
    for trip in null_params: 
        effs[trip] = (meds[trip][0] - null_params[trip][0], 
                      meds[trip][1] - null_params[trip][1])
        params[trip] = (meds[trip][0], meds[trip][1], meds[trip][2])
    titles = ["Growth Rate", "Carrying Capacity", "Lag Time"]
    units = ["", " (OD600)", " (hr)"]
    for i in range(3):
        focus = extract(params, i)
        null = extract(null_params, i)
        bar_charts(titles[i], units[i], null, focus)
        
if __name__ == "__main__": main()
    
    
