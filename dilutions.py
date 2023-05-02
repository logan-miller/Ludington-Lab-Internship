import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme(palette='husl')

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

file = input("Enter filename: ")

# Establish timestamps from number of reads (listed in sheet)
read_string = read_value_from_excel(file, "B", 18)
x_count = int(read_string[47:-6])
x_vals = np.arange(1 / 12, (x_count + 1) / 12, 1 / 12)

antibiotic = input("Enter antibiotic name: ")
abbr = input("Enter antibiotic abbreviation: ")
init_conc = int(input("Enter initial " + antibiotic + " concentration (ug/mL): "))

# Establish dilution values
dilutions = [init_conc]
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
    species = pd.read_excel(file, usecols=spec_cols, index_col=0, header=0, skiprows=range(28), nrows=x_count)
    control = pd.read_excel(file, usecols=control_cols, index_col=0, header=0, skiprows=range(28), nrows=x_count)
    species = normalize(species, control)
    name = input("Enter species " + str(num + 1) + " name: ")
    
    # Initialize log plot
    fig, spec_ax = plt.subplots(1,1,figsize=(15,15), sharex=True, sharey=True)
    plt.yscale("log")
    
    conc_r = {}
    
    zeroes = []
    for i in range(x_count):
        zeroes.append(0)
    averages = pd.DataFrame(data=zeroes, index=x_vals)
    fits = pd.DataFrame(data=zeroes, index=x_vals)
    averages.columns = [init_conc]
    fits.columns = [init_conc]
    
    
    # For each dilution
    for conc in dilutions:
        
        # Find average for the dilution
        conc_cols = []
        for i in range(0, len(species.columns), 6):
            conc_cols.append(i + dilutions.index(conc))
        replicates = species.iloc[:, conc_cols]
        conc_data = replicates.mean(axis=1)
        average = conc_data.to_frame()
        average.columns = [conc]
        average.index=x_vals
        
        # Guess logistic parameters
        k_guess = conc_data.max()
        r_guess = find_r(conc_data.values)
        x_0_guess = find_x_0(x_vals, conc_data.values, 0.1)
        
        # Implement curve fitting
        try:
            est_func, pcov = curve_fit(logistic, x_vals, conc_data.values, p0=[r_guess, k_guess, x_0_guess])
            conc_r[conc] = est_func[0]
            # Obtain fitted data
            fits[conc] = logistic(x_vals, *est_func)
        # Assume the sample showed no significant growth
        except RuntimeError:
            est_func, pcov = curve_fit(no_growth, x_vals, conc_data.values, p0=[0, 0])
            conc_r[conc] = 0
            fits[conc] = no_growth(x_vals, *est_func)
        
        # Plot average and fit data onto log plot
        sns.lineplot(data=average, ax=spec_ax)
        #spec_ax.plot(x_vals, est_curve, label=str(conc) + "ug/mL fit")
        spec_ax.legend(loc='lower right')
        #pec_ax.set_xlabel("Time (hr)")
        spec_ax.set_ylabel("log(OD600)")
        spec_ax.set(title="Growth of " + name + " in " + antibiotic)
    #plt.yscale("log")
    #avg_plt = sns.lineplot(data=averages)
    #plt.clf()
    # Export log plot as a file
    #export = input("Save " + name + " + " + antibiotic + "? (y/n) ")
    #if (export == "y"):
        #plt.savefig(name + " " + abbr + ".png")
    
    # Plot r vs. [antibiotic]
    #fig, ratio_ax = plt.subplots(1,1,figsize=(15,15))
    #plt.xscale("log")
    #ratio_ax.plot(conc_r.keys(), conc_r.values())
    #ratio_ax.set_xlabel("[" + antibiotic + "] (ug/mL)")
    #ratio_ax.set_ylabel("Growth Rate")
    #ratio_ax.set(title="Growth rate of " + name + " dependence on [" + antibiotic + "] (ug/mL)")
    
    # Export r plot as a file
    #export = input("Save " + name + " growth rate vs " + antibiotic + " conc.? (y/n) ")
    #if (export == "y"):
        #plt.savefig(name + " GR " + abbr + ".png")