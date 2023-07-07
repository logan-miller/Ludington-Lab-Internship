# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:32:52 2023

@author: bubba
"""

import matplotlib.pyplot as plt
import numpy as np

# intrinsic population growth rate of species
R1 = 2.0
R2 = 2.0
R3 = 2.0

# density-dependent self-limitation
D1 = 0.00000001
D2 = D1
D3 = D1

# saturation level of the hyperbolic functional response
A1 = 0.6
A2 = A1
A3 = A1

# half-saturation density of species
H1 = 0.3
H2 = H1
H3 = H1

# saturation level of resource supply function
B1 = 0.2
B2 = B1
B3 = 0

# half-saturation constant of resource supply function
E1 = 1.0
E2 = E1
E3 = E1


P1 = 1.0
P2 = P1
P3 = P1

C1 = 1
C2 = C1
C3 = C1

K = 10000

G3 = 0.1

def grow_diff(n1, n2, n3, num):
    if num == 1:
        f = (A1 * n2) / (H2 + n2)
        g = (B1 * n2) / (E1 + n1)
        change = n1 * (R1 + f - g - (D1 * n1))
        return n1 + change
    elif num == 2:
        f = (A2 * n1) / (H1 + n1)
        g = (B2 * n1) / (E2 + n2)
        h = (A3 * n1) / (H1 + n1)
        change = n2 * (R2 + f - g - (0.5 * h) - (D2 * n2))
        return n2 + change
    elif num == 3:
        f = (A3 * n1) / (H1 + n1)
        change = n3 * (R3 + (0.5 * f) - (D3 * n3))
        return n3 + change
    else:
        return 0

def grow_diff2(n1, n2, n3, num):
    if num == 1:
        n1 *= (R1 + ((P1 * n2) / (n3 + C1))) * (1 - ((n2 + n3 + n1) / K))
        return n1
    elif num == 2:
        n2 *= (R1 + (n1 / (n2 + n3 + C2))) * (1 - ((n2 + n3 + n1) / K))
        return n2
    elif num == 3:
        n3 *= (R1 + G3 + (n1 / (n2 + n3 + C2))) * (1 - ((n2 + n3 + n1) / K))
        return n3
    else:
        return 0

def main():
    steps = range(0, 1000)
    acet = 1
    lp = 2
    lb = 1
    acet_pops = []
    lp_pops = []
    lb_pops = []
    for time in steps:
        acet_pops.append(acet)
        lp_pops.append(lp)
        lb_pops.append(lb)
        new_acet = grow_diff2(acet, lp, lb, 1)
        if new_acet < 0: new_acet = 0;
        new_lp = grow_diff2(acet, lp, lb, 2)
        if new_lp < 0: new_lp = 0;
        new_lb = grow_diff2(acet, lp, lb, 3)
        if new_lb < 0: new_lb = 0;
        acet = new_acet
        lp = new_lp
        lb = new_lb
        
    acet_vec = np.array(acet_pops)
    lp_vec = np.array(lp_pops)
    lb_vec = np.array(lb_pops)
    vec_sum = acet_vec + lp_vec + lb_vec
    fig, ax = plt.subplots(1,1,figsize=(15,15), sharex=True, sharey=True)
    
    plt.plot(steps, acet_pops, label="Acet")
    plt.plot(steps, lp_pops, label="Lp")
    plt.plot(steps, lb_pops, label="Lb")
    plt.plot(steps, vec_sum, label="Sum")
    plt.legend()
        
if __name__ == "__main__":
    main()