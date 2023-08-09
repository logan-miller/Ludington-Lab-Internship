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

# Maps ZTG strains to possible filters
SPECS = {'L.plantarum': ['L.plantarum', 'Lp', 'ZTG301'],
         'L.brevis': ['L.brevis', 'Lb', 'ZTG304'],
         'A.tropicalis': ['A.tropicalis', 'At', 'ZTG310'],
         'A.orientalis': ['A.orientalis', 'Ao', 'ZTG331'],
         'A.malorum': ['A.malorum', 'Am', 'ZTG303'],
         'A.sicerae': ['A.sicerae', 'As', 'ZTG360'],
         'A.cerevisiae': ['A.cerevisiae', 'Ac', 'ZTG313']}

# Maps antibiotics to possible filters
ANTIBIOTICS = {'Chloramphenicol': ['Chloramphenicol', 'Cam', 'Chlor'],
               'Carbenicillin': ['Carbenicillin', 'Carb', 'Car'],
               'Erythromycin': ['Erythromycin', 'Erm'],
               'Gentamicin': ['Gentamicin', 'Gen'],
               'Kanamycin': ['Kanamycin', 'Kan'],
               'Rifampicin': ['Rifampicin', 'Rif'],
               'Tetracycline': ['Tetracycline', 'Tet']}

# Maps ZTG strains to colors
SPEC_COLORS = {'L.plantarum': 'tab:blue',
               'L.brevis': 'tab:orange',
               'A.tropicalis': 'tab:green',
               'A.orientalis': 'tab:red',
               'A.malorum': 'tab:purple',
               'A.sicerae': 'tab:pink',
               'A.cerevisiae': 'tab:cyan'}


# A class to represent a writable StringVar, used with Tkinter Labels
class WritableStringVar(tk.StringVar):
    # Write to the StringVar
    def write(self, added_text):
        new_text = self.get() + added_text
        self.set(new_text)
        
    # Clear the StringVar
    def clear(self):
        self.set("")

# A class to represent a single bacterial species in a single experiment
class Species:
    
    # Determine the antibiotic concentrations used in a run from the maximum
    def get_dilutions(self, concentration):
        concs = [float(concentration)]
        for i in range(1, 5): concs.append(concs[i - 1] / 2)
        concs.append(0)
        return concs
    
    # Count number of reads in an experiment
    def find_read_count(self, column):
        # Pull single column from sheet
        col = pd.read_excel(self.filename, usecols=column, 
                            header=None).values.tolist()
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
    
    # Initialization
    def __init__(self, filename, name, antibiotic, conc, pos):
        self.name = name # Bacterial species name
        self.antibiotic = antibiotic # Name of antibiotic
        # Dilutions of antibiotic used
        self.dilutions = self.get_dilutions(conc)
        # Filename of experimental result Excel sheet
        self.filename = filename
        # Number of reads and space until reads are present
        self.x_num, self.skip_num = self.find_read_count('D')
        # Experimental timestamps (in hr)
        self.xs = np.arange(1 / 12, (self.x_num + 1) / 12, 1 / 12)
        # Experimental position; 0 for left side of the plate, 1 for right
        self.pos = pos
        # Normalized growth curve data; keys are antibiotic concentrations and
        # values are a pair. The first is a list of raw OD600 measurements,
        # and the second is a list of fit measurements
        self.gc_data = {}
        # MIC data; keys are antibiotic concentrations and values the fit 
        # carrying capacity at the concentration.
        self.mic_data = {}
    
    # Return the species name
    def get_name(self): return self.name
    
    # Return the MIC data
    def get_mic(self): return self.mic_data
    
    # Add a data pair for a specific concentration
    def add_gc(self, conc, data, avg): self.gc_data[conc] = (data, avg)
    
    # Normalize a set of data using a control
    def normalize(self, data, control):
        norm_factors = control.mean(axis=1).to_frame()
        for i in range(1, len(data.columns)): 
            # Make control array the same size as the data array
            norm_factors[i] = norm_factors.loc[:, 0]
        normed = np.subtract(data, np.asarray(norm_factors))
        normed[normed < 0] = 0 # Adjust neg values to 0
        return normed
    
    # Read in and normalize the growth curve data, then obtain the average runs
    def process(self, mic, gc, gc_save):
        # Establish the specific columns for data and control
        displacement = (self.pos - 1) * 6
        spec_cols = [1]
        for i in range(0, 84, 12):
            for j in range(6): spec_cols.append(i + j + 3 + displacement)       
        control_cols = [1]
        for i in range(87, 93): control_cols.append(i + displacement)
            
        # Obtain and normalize data
        species = pd.read_excel(self.filename, usecols=spec_cols, index_col=0,
                                header=0, skiprows=range(self.skip_num),
                                nrows=self.x_num)
        control = pd.read_excel(self.filename, usecols=control_cols,
                                index_col=0, header=0,
                                skiprows=range(self.skip_num),
                                nrows=self.x_num)
        species = self.normalize(species, control)
        
        # Record average lines and fits
        self.get_lines(species, mic, gc, gc_save)
    
    # Used to ignore a curve fitting warning; thrown when there is a poor fit
    def fxn(self):
        warnings.warn('Covariance of the parameters could not be estimated', 
                      opt.OptimizeWarning)

    # Logistic growth model
    def logistic(self, x, r, k, x_0): return k / (1 + np.exp(-r * (x - x_0)))
    
    # Linear model
    def no_growth(self, x, m, b): return (m * x) + b
    
    # Estimate growth rate by finding maximum symmetric difference (slope)
    def find_r(self, data):
        max_diff = 0
        for i in range(1, len(data) - 1):
            diff = (data[i + 1] - data[i - 1]) * 6 # Symmetric difference
            if(diff > max_diff): max_diff = diff
        return max_diff

    # Find closest value in a list to a target value n
    # (closest OD to likely sigmoid midpoint)
    def find_x_0(self, data, n):
        sig_mid = np.max(self.xs)
        sig_mid_val = np.max(data)
        for i in range(self.x_num):
            if(np.abs(data[i] - n) < sig_mid_val):
                sig_mid_val = np.abs(data[i] - n)
                sig_mid = self.xs[i]
        if (sig_mid_val == np.max(data)): sig_mid = 0
        return sig_mid
    
    # Obtain average lines and their best fits
    def get_lines(self, data, mic, gc, gc_save):
        
        # MIC curve data; keys are antibiotic concentrations, values are the
        # average carrying capacity at that concentration
        mic_data = {}
        
        # For each dilution
        for conc in self.dilutions:
            
            # Find average run for the dilution
            conc_cols = []
            for i in range(0, len(data.columns), 6):
                conc_cols.append(i + self.dilutions.index(conc))
            replicates = data.iloc[:, conc_cols]
            conc_data = replicates.mean(axis=1)
            
            # Guess logistic parameters
            k_guess = conc_data.max()
            r_guess = self.find_r(conc_data.values)
            x_0_guess = self.find_x_0(conc_data.values, 0.1)
            
            # Implement curve fitting
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    self.fxn() # Used to ignore curve fitting failure warnings
                    est_func = opt.curve_fit(self.logistic, self.xs, 
                                             conc_data.values,
                                             p0=[r_guess, k_guess, 
                                                 x_0_guess])[0]
                
                # If the logistic curve fitting worked, but there was really no
                # growth, use the maximum growth value
                if est_func[2] > (0.75 * self.xs.max()):
                    mic_data[conc] = k_guess
                else: mic_data[conc] = est_func[1]
                
                # Add curve data if necessary
                if gc: 
                    self.add_gc(conc, conc_data, self.logistic(self.xs, 
                                                               *est_func))
    
            # Assume the sample showed no significant growth
            except RuntimeError:
                est_func = opt.curve_fit(self.no_growth, self.xs, 
                                         conc_data.values, p0=[0, 0])[0]
                mic_data[conc] = k_guess
                
                # Add curve data if necessary
                if gc: 
                    self.add_gc(conc, conc_data, self.no_growth(self.xs,
                                                                *est_func))
        
        if mic: self.mic_data = mic_data # Add MIC data to the Species
        if gc: self.graph_gcs(gc_save) # Graph the bacterial growth curves
    
    # Graph the growth curves of at all antibacterial concentrations on a
    # single line plot
    def graph_gcs(self, save):
        
        title = self.filename[:-5] # Crops filename of '.xlsx'
        
        # Initialize plot
        fig, ax = plt.subplots(1,1,figsize=(15,15), sharex=True, sharey=True)
        plt.yscale('log')
        
        # Used to cycle through colors when plotting
        color_count = 0
        colors = list(SPEC_COLORS.values())
        
        # For each dilution
        for conc in self.gc_data:
            # Plot raw data as solid line
            ax.plot(self.xs, self.gc_data[conc][0], label='[' + 
                    self.antibiotic + '] = ' + str(conc) + ' ug/mL',
                    c=colors[color_count])
            # Plot fitted data as dashed line
            ax.plot(self.xs, self.gc_data[conc][1], label=str(conc) + 
                    ' ug/mL fit', c=colors[color_count], linestyle='dashed')
            color_count += 1
        
        # More plot details
        ax.legend(loc='lower right')
        ax.set_xlabel('Time (hr)')
        ax.set_ylabel('OD600')
        ax.set(title='Growth of ' + self.name + ' in ' + self.antibiotic +
               ' Concentrations')
        if save: fig.savefig(title + '_' + self.name + '_GC.png')

# Check if the beginning of a word matches with a list of other words
def check_match(ab, spec, itin):
    
    if len(itin) == 0: return True # If there is no filter
    
    for elem in itin: # For each filter keyword
        for match in ANTIBIOTICS[ab]: # Check antibiotic matches
            if re.search(match, elem): return True
        for match in SPECS[spec]: # Check species name matches
            if re.search(match, elem): return True
    return False

# Determine if an x value is within the range of the MIC concentrations, and
# return its y value on the graph if so
def find_intermediate(x, data):
    concs = np.sort(list(data.keys()))
    for i in range(len(concs) - 1): # For every pair of concentrations
        # Check if x is between consecutive pairs
        if concs[i] <= x <= concs[i + 1]:
            slope = ((data[concs[i + 1]] - data[concs[i]]) / 
                     (concs[i + 1] - concs[i]))
            return (slope * x) + data[concs[i]]

# Determine an average MIC line for each species in a single antibiotic
def avg_mics(spec_list):
    
    # MIC data by species; keys are species names, values are a list of MIC runs
    spec_mics = {}
    
    # Organize single experimental MIC curves by species
    for spec in spec_list:
        name = spec.get_name()
        if name not in list(spec_mics.keys()): spec_mics[name] = []
        spec_mics[name].append(spec.get_mic())
    
    # Initial OD600 values; keys are species names, values are the average
    # untreated value across MIC runs for each species
    starts = {}
    
    # Antibiotic concentrations; keys are species names, values are the
    # set of concentrations used across all MIC runs for that species
    spec_concs = {}
    
    # Fill starts and spec_concs
    for spec in spec_mics:
        start_list = [] # Collects initial values
        spec_concs[spec] = []
        for mic_data in spec_mics[spec]:
            start = mic_data[0]
            start_list.append(start)
            for conc in mic_data:
                mic_data[conc] -= start
                spec_concs[spec].append(conc)
        spec_concs[spec] = np.sort([*set(spec_concs[spec])])
        starts[spec] = np.mean(start_list)
        
    # MIC curves; keys are species names, values are dicts containing the
    # average MIC data for that species and antibioitic
    avg_mic = {}
    
    # Fill avg_mic
    for spec in spec_mics: # For each species
        mic = {}
        for conc in spec_concs[spec]: # For each concentration used
            conc_vals = []
            # Append the value at each concentration
            for mic_data in spec_mics[spec]:
                if conc in list(mic_data.keys()):
                    conc_vals.append(mic_data[conc])
                elif conc < max(list(mic_data.keys())):
                    # If the concentration was used and is within the range of
                    # this experiment, but was not specifically tested,
                    # interpolate to find the concentration's value on the line
                    conc_vals.append(find_intermediate(conc, mic_data))
            mic[conc] = np.mean(conc_vals) + starts[spec] # Average the value
        avg_mic[spec] = mic
    
    return avg_mic

# Scale MIC data to be proportions of the maximum OD value
def scale(ab):
    scaled = {}
    for spec in ab:
        scaled[spec] = {}
        od_max = max(ab[spec].values()) # Find the max
        for conc in ab[spec]: scaled[spec][conc] = ab[spec][conc] / od_max
    return scaled

# Produce and save (if necessary) MIC graphs for an antibioitic
def graph_mic(antibiotic, spec_mics, save):
    
    # Initialization of the plot
    fig, ax = plt.subplots(1,1,figsize=(15,15), sharex=True, sharey=True)
    plt.xscale('symlog')
    ax.set_xlabel('[' + antibiotic + '] (ug/mL)')
    ax.set_ylabel('Proportion to Maximum Carrying Capacity')
    ax.set(title='Carrying Capacity Dependence on ' + antibiotic + 
           ' Concentration')
    
    # Plot each MIC graph
    for spec in spec_mics:
        ax.plot(spec_mics[spec].keys(), spec_mics[spec].values(), label=spec, 
                c=SPEC_COLORS[spec])
    
    # More plot details
    ax.legend(loc=4)
    ax.autoscale()
    ax.set_ylim(bottom=0)
    
    if save: fig.savefig(antibiotic + 'CC.png') # Save if necessary

# Determine if a concentration is on a MIC graph, based on a OD600 value, and
# return the concentration if so
def find_conc(y, data):
    
    concs = list(data.keys()) # Antibiotic concentrations
    for i in range(len(concs) - 1): # For each consecutive concentration pair
        # If the OD600 value is between any other consecutive OD600 values,
        # return the corresponding concentration
        if (data[concs[i]] > y > data[concs[i + 1]]
            or data[concs[i]] < y < data[concs[i + 1]]):
            slope = ((data[concs[i + 1]] - data[concs[i]]) / 
                     (concs[i + 1] - concs[i]))
            y_int = data[concs[i]] - (slope * concs[i])
            return (y - y_int) / slope
    
    return np.NaN # Otherwise, signify there is no MIC-50

# Determine which MIC graphs have a MIC-50 value, and store for display
def find_mics(data):
    mics = {}
    text = ""
    for ab in data:
        mics[ab] = {}
        for spec in data[ab]:
            goal = 0.5 * data[ab][spec][0] # Half of the untreated OD600 value
            mics[ab][spec] = find_conc(goal, data[ab][spec])
    
    # Record data in a displayable manner
    for ab in mics:
        text += ab + ":\n"
        for spec in mics[ab]:
            if mics[ab][spec] == mics[ab][spec]: # If not NaN
                text += ("\t" + spec + " - " + 
                         str(np.round(mics[ab][spec], decimals=3)) + 
                         " ug/mL\n")
            else: text += "\t" + spec + " - MIC-50 value not reached\n"
    
    return text

# Run the proper GUI commands (create GC/MIC graphs, find MIC-50 values)
def run(filename, sheet_name, filt, gc, mic, 
        save_gc, save_mic, fifty, txtvar, root):
    
    # Read in experimental summary data
    xls = pd.ExcelFile(filename)
    df = pd.read_excel(xls, sheet_name, usecols=list(range(7)))
    
    # MIC curves; keys are antibiotic names, values are the corresponding MIC 
    # curve data
    antibiotics = {}
    for index, row in df.iterrows(): # For each experimental plate
        for i in [1, 2]: # For each side of the experimental plate
            antibio = row['Antibiotic ' + str(i)] # Antibiotic name
            spec = row['Species ' + str(i)] # Species name
            # Summary details for this species run
            spec_details = [row['Filename'], spec, antibio, 
                            row['Initial Concentration ' + str(i)], i]
            if check_match(antibio, spec, filt):
                if antibio not in list(antibiotics.keys()): 
                    antibiotics[antibio] = []
                antibiotics[antibio].append(Species(*spec_details))
    
    # Average MIC curves; keys are antibiotics, values are dicts whose keys are
    # species names and values are the average MIC curve for that combination
    avg_lines = {}
    
    scaled = {}
    
    gen_lines = mic or fifty # Either one necessitates MIC lines to be created
    
    # Run all necessary functions
    for ab in antibiotics:
        for spec in antibiotics[ab]: spec.process(gen_lines, gc, save_gc)
        if gen_lines: 
            avg_lines[ab] = avg_mics(antibiotics[ab])
            scaled[ab] = scale(avg_lines[ab])
        if mic: graph_mic(ab, scaled[ab], save_mic)
        
    # Display MIC-50 data if necessary
    txtvar.clear()
    if fifty: print(find_mics(avg_lines), file=txtvar)

# Toggles the value and state of a Checkbutton (check) based on another 
# Checkbutton (val)
def toggle_state(check, val, check_val):
    # If val is checked, check is enabled
    if val.get(): check.config(state='normal')
    else: # If val is unchecked, check is unchecked and disabled
        check_val.set(0)
        check.config(state='disabled')
    
# Create and run a GUI to process experimental antibiotic screening data
def main():
    
    root = tk.Tk()
    root.title('Antibiotic Screens') # GUI banner title
    
    # Collect screening summary filename
    file_lbl = ttk.Label(root, text='Summary Sheet Filename:')
    file_ent = ttk.Entry(root, width=50)
    file_ent.insert(0, 'Antibiotic_Screen.xlsx')
    
    # Collect name of sheet to use in screening summary file
    sheet_lbl = ttk.Label(root, text='Sheet Name:')
    sheet_ent = ttk.Entry(root, width=50)
    sheet_ent.insert(0, 'ScreeningList')
    
    # Collect the filter to sort which antibiotics/species are to be used
    focus_lbl = ttk.Label(root, text='Filter:')
    focus_txt = tk.Text(root, height=1, width=50)
    focus_txt.configure(font=font.nametofont('TkDefaultFont'))
    
    # Determine whether growth curves should be generated/saved
    gc_val = tk.IntVar(root, value=0)
    gc_sv_val = tk.IntVar(root, value=0)
    gc_sv_chk = ttk.Checkbutton(root, text='Save Growth Curve Graphs', 
                                variable=gc_sv_val)
    gc_switch = (lambda e=gc_sv_chk, v=gc_val, r=gc_sv_val:
                 toggle_state(e, v, r))
    gc_chk = ttk.Checkbutton(root, text='Generate Growth Curve Graphs', 
                             variable=gc_val, command=gc_switch)
    gc_sv_chk.config(state='disabled')
    
    # Determine whether MIC curves should be generated/saved
    mic_val = tk.IntVar(root, value=0)
    mic_sv_val = tk.IntVar(root, value=0)
    mic_sv_chk = ttk.Checkbutton(root, text='Save MIC Graphs',
                                 variable=mic_sv_val)
    mic_switch = (lambda e=mic_sv_chk, v=mic_val, r=mic_sv_val: 
                  toggle_state(e, v, r))
    mic_chk = ttk.Checkbutton(root, text='Generate MIC Graphs', 
                              variable=mic_val, command=mic_switch)
    mic_sv_chk.config(state='disabled')
    
    # Determine whether MIC-50 data should be generated
    fifty_val = tk.IntVar(root, value=0)
    fifty_var = WritableStringVar(root)
    fifty_chk = ttk.Checkbutton(root, text='Display MIC-50 Values:', 
                              variable=fifty_val)
    fifty_lbl = tk.Label(root, anchor='w', justify='left',
                         textvariable=fifty_var)
    
    # Runs any selected functions
    go = (lambda e=file_ent, s=sheet_ent, t=focus_txt, g=gc_val, gs=gc_sv_val,
          m=mic_val, ms=mic_sv_val, f=fifty_val, fe=fifty_var, r=root:
              run(e.get(), s.get(),t.get('1.0', tk.END).split(), g.get(), 
                  m.get(), gs.get(), ms.get(), f.get(), fe, r))
    run_btn = ttk.Button(root, text='Run', command=go)
    
    # Exits the window
    quit_btn = ttk.Button(root, text='Quit', command=root.destroy)
    
    # Organization of elements within the window
    file_lbl.grid(row=0, column=0, sticky='e', padx=5, pady=5)
    file_ent.grid(row=0, column=1, padx=5, pady=5)
    sheet_lbl.grid(row=1, column=0, sticky='e', padx=5, pady=5)
    sheet_ent.grid(row=1, column=1, padx=5, pady=5)
    focus_lbl.grid(row=2, column=0, sticky='e', padx=5, pady=5)
    focus_txt.grid(row=2, column=1, padx=5, pady=5)
    gc_chk.grid(row=0, column=2, sticky='w', padx=5, pady=5)
    mic_chk.grid(row=2, column=2, sticky='w', padx=5, pady=5)
    gc_sv_chk.grid(row=1, column=2, sticky='w', padx=5, pady=5)
    mic_sv_chk.grid(row=3, column=2, sticky='nw', padx=5, pady=5)
    fifty_chk.grid(row=3, column=0, sticky='ne', padx=5, pady=5)
    fifty_lbl.grid(row=3, column=1, sticky='w', padx=5, pady=5)
    quit_btn.grid(row=4, column=0, sticky='w', padx=5, pady=5)
    run_btn.grid(row=4, column=3, sticky='e', padx=5, pady=5)
    
    root.mainloop() # Runs the window
    return 0

# Directs to main
if __name__ == '__main__': main()