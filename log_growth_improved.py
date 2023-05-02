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

def read_value_from_excel(filename, column, row):
    return pd.read_excel(filename, skiprows=row - 1, usecols=column, nrows=1, header=None, names=["Value"]).iloc[0]["Value"]

def normalize(data, control):
    norm_factors = control.mean(axis=1).to_frame()
    #print(norm_factors)
    for i in range(1, len(data.columns)):
        norm_factors[i] = norm_factors.loc[:, 0]
    normed = np.subtract(data, np.asarray(norm_factors))
    normed[normed < 0] = 0
    return normed

file = input("Enter filename: ")
read_string = read_value_from_excel(file, "B", 18)
x_count = int(read_string[47:-6])
x_vals = np.arange(1 / 12, (x_count + 1) / 12, 1 / 12)

antibiotic = input("Enter antibiotic name: ")
abbr = input("Enter antibiotic abbreviation: ")
init_conc = int(input("Enter initial " + antibiotic + " concentration (ug/uL): "))

dilutions = [init_conc]
for i in range(1, 5):
    dilutions.append(dilutions[i - 1] / 2)
dilutions.append(0)

for num in [0, 1]:
    displacement = num * 6
    
    spec_cols = [1]
    for i in range(0, 84, 12):
        for j in range(6):
            spec_cols.append(i + j + 3 + displacement)
            
    control_cols = [1]
    for i in range(87, 93):
        control_cols.append(i + displacement)

    species = pd.read_excel(file, usecols=spec_cols, index_col=0, header=0, skiprows=range(28), nrows=x_count)
    control = pd.read_excel(file, usecols=control_cols, index_col=0, header=0, skiprows=range(28), nrows=x_count)
    species = normalize(species, control)
    name = input("Enter species " + str(num + 1) + " name: ")
    
    fig, spec_ax = plt.subplots(1,1,figsize=(15,15), sharex=True, sharey=True)
    plt.yscale("log")
    
    conc_r = {}
    for conc in dilutions:
        conc_cols = []
        for i in range(0, len(species.columns), 6):
            conc_cols.append(i + dilutions.index(conc))
        replicates = species.iloc[:, conc_cols]
        conc_data = replicates.mean(axis=1)
        
        k_guess = conc_data.max()
        r_guess = find_r(conc_data.values)
        x_0_guess = find_x_0(x_vals, conc_data.values, 0.1)
        
        # Implement curve fitting
        try:
            est_func, pcov = curve_fit(logistic, x_vals, conc_data.values, p0=[r_guess, k_guess, x_0_guess])
            conc_r[conc] = est_func[0]
            # Obtain fitted data
            est_curve = logistic(x_vals, *est_func)
        except RuntimeError:
            est_func, pcov = curve_fit(no_growth, x_vals, conc_data.values, p0=[0, 0])
            conc_r[conc] = 0
            est_curve = no_growth(x_vals, *est_func)
        
        # Plot replicate and fit data onto strain plot
        spec_ax.plot(x_vals, conc_data.values, label="[" + abbr + "] = " + str(conc) + " ug/uL")
        spec_ax.plot(x_vals, est_curve, label=str(conc) + "ug/uL fit")
        
        spec_ax.legend(loc='lower right')
        spec_ax.set_xlabel("Time (hr)")
        spec_ax.set_ylabel("log(OD600)")
        spec_ax.set(title="Growth of " + name + " in " + antibiotic)
        
    export = input("Save " + name + " + " + antibiotic + "? (y/n) ")
    if (export == "y"):
        plt.savefig(name + " " + abbr + ".png")
    
    fig, ratio_ax = plt.subplots(1,1,figsize=(15,15))
    plt.xscale("log")
    ratio_ax.plot(conc_r.keys(), conc_r.values())
    ratio_ax.set_xlabel("[" + antibiotic + "] (ug/uL)")
    ratio_ax.set_ylabel("Growth Rate")
    ratio_ax.set(title="Growth rate of " + name + " dependence on [" + antibiotic + "] (ug/uL)")
    
    export = input("Save " + name + " growth rate vs " + antibiotic + " conc.? (y/n) ")
    if (export == "y"):
        plt.savefig(name + " GR " + abbr + ".png")