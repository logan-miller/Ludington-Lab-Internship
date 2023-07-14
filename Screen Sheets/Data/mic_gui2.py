# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
from tkinter import font
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import warnings
sns.set_theme()

# Maps ZTG strains to colors
SPEC_COLORS = {'L.plantarum': 'tab:blue',
               'L.brevis': 'tab:orange',
               'A.tropicalis': 'tab:green',
               'A.orientalis': 'tab:red',
               'A.malorum': 'tab:purple',
               'A.sicerae': 'tab:pink',
               'A.cerevisiae': 'tab:cyan'}

def fxn():
    warnings.warn('Covariance of the parameters could not be estimated', 
                  opt.OptimizeWarning)

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

# Normalize a set of data using a control
def normalize(data, control):
    norm_factors = control.mean(axis=1).to_frame()
    for i in range(1, len(data.columns)): 
        # Make control array the same size as the data array
        norm_factors[i] = norm_factors.loc[:, 0]
    normed = np.subtract(data, np.asarray(norm_factors))
    normed[normed < 0] = 0 # Adjust neg values to 0
    return normed

# Generate MIC plots from an Excel sheet
def mics(file, antibiotic, spec_1, spec_2, init_conc, plots, specs):
    
    strains = (spec_1, spec_2)
    # Find read count and locate data
    x_count, skip_count = find_read_count(file, 'D')
    # Establish timestamps from number of reads (listed in sheet)
    x_vals = np.arange(1 / 12, (x_count + 1) / 12, 1 / 12)
    
    # Initialize the concentrations used
    dilutions = [float(init_conc)]
    for i in range(1, 5):
        dilutions.append(dilutions[i - 1] / 2)
    dilutions.append(0)
    
    if antibiotic in plots: # If there are already plots for that antibiotic
        r_fig = plots[antibiotic][0]
        r_ax = plots[antibiotic][1]
        k_fig = plots[antibiotic][2]
        k_ax = plots[antibiotic][3]
    else: # Make new plots and add to dict
        r_fig, r_ax = plt.subplots(1,1,figsize=(15,15), sharex=True, 
                                   sharey=True)
        plt.xscale('log')
        r_ax.set_xlabel('[' + antibiotic + '] (ug/mL)')
        r_ax.set_ylabel('Growth Rate')
        r_ax.set(title='Growth Rate Dependence on ' + 
                 antibiotic + ' Concentration')
        
        k_fig, k_ax = plt.subplots(1,1,figsize=(15,15), sharex=True, 
                                   sharey=True)
        plt.xscale('log')
        k_ax.set_xlabel('[' + antibiotic + '] (ug/mL)')
        k_ax.set_ylabel('Carrying Capacity (OD600)')
        k_ax.set(title='Carrying Capacity Dependence on ' + 
                 antibiotic + ' Concentration')
        plots[antibiotic] = [r_fig, r_ax, k_fig, k_ax]
        specs[antibiotic] = []
    
    # For each of the two species
    for num in [0, 1]:
        
        # Establish the specific columns for data and control
        displacement = num * 6
        spec_cols = [1]
        for i in range(0, 84, 12):
            for j in range(6):
                spec_cols.append(i + j + 3 + displacement)       
        control_cols = [1]
        for i in range(87, 93):
            control_cols.append(i + displacement)
            
        # Initialize and normalize data
        species = pd.read_excel(file, usecols=spec_cols, index_col=0, header=0,
                                skiprows=range(skip_count), nrows=x_count)
        control = pd.read_excel(file, usecols=control_cols, index_col=0,
                                header=0, skiprows=range(skip_count),
                                nrows=x_count)
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
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fxn()
                    est_func, pcov = opt.curve_fit(logistic, x_vals, 
                                                   conc_data.values,
                                                   p0=[r_guess, k_guess,
                                                       x_0_guess])
                conc_r[conc] = est_func[0]
                if est_func[2] > (0.75 * x_vals.max()): conc_k[conc] = k_guess
                else: conc_k[conc] = est_func[1]
            # Assume the sample showed no significant growth
            except RuntimeError:
                est_func, pcov = opt.curve_fit(no_growth, x_vals, 
                                               conc_data.values, p0=[0, 0])
                conc_r[conc] = 0
                conc_k[conc] = k_guess
        # Plot r and k vs. [antibiotic]
        if strains[num] not in specs[antibiotic]: # If new species, add label
            specs[antibiotic].append(strains[num])
            r_ax.plot(conc_r.keys(), conc_r.values(), label=strains[num], 
                      c=SPEC_COLORS[strains[num]])
            k_ax.plot(conc_k.keys(), conc_k.values(), label=strains[num], 
                      c=SPEC_COLORS[strains[num]])
        else:
            r_ax.plot(conc_r.keys(), conc_r.values(), label='_', 
                      c=SPEC_COLORS[strains[num]])
            k_ax.plot(conc_k.keys(), conc_k.values(), label='_', 
                      c=SPEC_COLORS[strains[num]])
    return plots

def gcs(file, antibiotic, spec_1, spec_2, init_conc, plots):
    
    strains = (spec_1, spec_2)
    # Establish timestamps from number of reads (listed in sheet)
    x_count, skip_val = find_read_count(file, 'D')
    x_vals = np.arange(1 / 12, (x_count + 1) / 12, 1 / 12)
    
    # Establish dilution values
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
        species = pd.read_excel(file, usecols=spec_cols, index_col=0, header=0,
                                skiprows=range(skip_val), nrows=x_count)
        control = pd.read_excel(file, usecols=control_cols, index_col=0, 
                                header=0, skiprows=range(skip_val), 
                                nrows=x_count)
        species = normalize(species, control)
        name = strains[num]
        
        # Initialize log plot
        fig, spec_ax = plt.subplots(1,1,figsize=(15,15), sharex=True, 
                                    sharey=True)
        plt.yscale('log')
        
        color_count = 0
        color_keys = list(SPEC_COLORS.keys())
        
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
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fxn()
                    est_func, pcov = opt.curve_fit(logistic, x_vals, 
                                                   conc_data.values,
                                                   p0=[r_guess, k_guess,
                                                       x_0_guess])
                    est_curve = logistic(x_vals, *est_func) # Obtain fits
            except RuntimeError: # Assume the sample showed no real growth
                est_func, pcov = opt.curve_fit(no_growth, x_vals, 
                                               conc_data.values, p0=[0, 0])
                est_curve = no_growth(x_vals, *est_func)
            
            # Plot average and fit data onto log plot, coordinate colors
            spec_ax.plot(x_vals, conc_data.values, label='[' + 
                         antibiotic + '] = ' + str(conc) + ' ug/mL', 
                         c = SPEC_COLORS[color_keys[color_count]])
            spec_ax.plot(x_vals, est_curve, label=str(conc) + ' ug/mL fit', 
                         c = SPEC_COLORS[color_keys[color_count]],
                         linestyle='dashed')
            color_count += 1
            spec_ax.legend(loc='lower right')
            spec_ax.set_xlabel('Time (hr)')
            spec_ax.set_ylabel('OD600')
            spec_ax.set(title='Growth of ' + name + ' in ' + 
                        antibiotic + ' Concentrations')
            
        plots.append((file, strains[num], fig, spec_ax)) # Add to list
    return plots

# Check if the beginning of a word matches with a list of other words
def check_match(word, itin):
    if len(itin) == 0: return True
    for elem in itin:
        if re.search(elem, word): return True
    return False

def run(filename, sheet_name, filt, gc, mic, save_gc, save_mic, root):
    
    plots = {}
    gc_plots = []
    species = {}
    xls = pd.ExcelFile(filename)
    df = pd.read_excel(xls, sheet_name, usecols=list(range(5)))
    for index, row in df.iterrows():
        if check_match(row['Filename'], filt):
            if mic: plots = mics(*list(row), plots, species)
            if gc: gc_plots = gcs(*list(row), gc_plots)
    if mic:
        for plot in plots:
            plots[plot][1].legend(loc=4)
            plots[plot][3].legend(loc=4)
            plots[plot][1].autoscale()
            plots[plot][3].autoscale()
            plots[plot][1].set_ylim(bottom=0)
            plots[plot][3].set_ylim(bottom=0)
            if save_mic:
                plots[plot][0].savefig(plot + 'GR.png')
                plots[plot][2].savefig(plot + 'CC.png')
    if gc:
        for plot in gc_plots:
            plot[3].legend(loc=4)
            plot[3].autoscale()
            if save_gc: plot[2].savefig(plot[0][:-5] + '_' + 
                                        plot[1] + '_GC.png')
    root.destroy()

def toggle_state(check, val, check_val):
    if val.get(): check.config(state='normal')
    else: 
        check_val.set(0)
        check.config(state='disabled')
    

def main():
    root = tk.Tk()
    root.title('Antibiotic Screens')
    file_lbl = ttk.Label(root, text='Summary Sheet Filename:')
    file_ent = ttk.Entry(root, width=50)
    file_ent.insert(0, 'Screening_Summary.xlsx')
    sheet_lbl = ttk.Label(root, text='Sheet Name:')
    sheet_ent = ttk.Entry(root, width=50)
    sheet_ent.insert(0, 'ScreeningList')
    focus_lbl = ttk.Label(root, text='Filter:')
    focus_txt = tk.Text(root, height=1, width=50)
    focus_txt.configure(font=font.nametofont('TkDefaultFont'))
    gc_val = tk.IntVar(root, value=0)
    gc_sv_val = tk.IntVar(root, value=0)
    gc_sv_chk = ttk.Checkbutton(root, text='Save Growth Curve Graphs', 
                                variable=gc_sv_val)
    gc_switch = (lambda e=gc_sv_chk, v=gc_val, r=gc_sv_val:
                 toggle_state(e, v, r))
    gc_chk = ttk.Checkbutton(root, text='Generate Growth Curve Graphs', 
                             variable=gc_val, command=gc_switch)
    gc_sv_chk.config(state='disabled')
    mic_val = tk.IntVar(root, value=0)
    mic_sv_val = tk.IntVar(root, value=0)
    mic_sv_chk = ttk.Checkbutton(root, text='Save MIC Graphs', 
                                 variable=mic_sv_val)
    mic_switch = (lambda e=mic_sv_chk, v=mic_val, r=mic_sv_val: 
                  toggle_state(e, v, r))
    mic_chk = ttk.Checkbutton(root, text='Generate MIC Graphs', 
                              variable=mic_val, command=mic_switch)
    mic_sv_chk.config(state='disabled')
    go = (lambda e=file_ent, s=sheet_ent, t=focus_txt, g=gc_val, gs=gc_sv_val,
          m=mic_val, ms=mic_sv_val, r=root:
              run(e.get(), s.get(),t.get('1.0', tk.END).split(), g.get(), 
                  m.get(), gs.get(), ms.get(), r))
    run_btn = ttk.Button(root, text='Run', command=go)
    file_lbl.grid(row=0, column=0, sticky='e', padx = 5, pady = 5)
    file_ent.grid(row=0, column=1, padx = 5, pady = 5)
    sheet_lbl.grid(row=1, column=0, sticky='e', padx = 5, pady = 5)
    sheet_ent.grid(row=1, column=1, padx = 5, pady = 5)
    focus_lbl.grid(row=2, column=0, sticky='e', padx = 5, pady = 5)
    focus_txt.grid(row=2, column=1, padx = 5, pady = 5)
    gc_chk.grid(row=0, column=2, sticky='w', padx = 5, pady = 5)
    mic_chk.grid(row=2, column=2, sticky='w', padx = 5, pady = 5)
    gc_sv_chk.grid(row=1, column=2, sticky='w', padx = 5, pady = 5)
    mic_sv_chk.grid(row=3, column=2, sticky='w', padx = 5, pady = 5)
    run_btn.grid(row=4, column=3, sticky='w', padx = 5, pady = 5)
    
    root.mainloop()
    root.quit()
    return 0

if __name__ == '__main__': main()