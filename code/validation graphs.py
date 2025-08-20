"""
Compare target trajectory with simulated and inferred
Produce graphs and metrics
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from test_sto_parse import read_sto #sto to df
from textwrap import wrap

target_sto_path = r"C:\Users\renaa\Documents\SCONE\results\250723.144821.leg6dof9musc.FC2.MW.D2\0339_11677.804_11460.316.par.sto"
target_traj = read_sto(target_sto_path)

simulated_sto = r"C:\Users\renaa\Documents\SCONE\results\validation_results\leg6dof9musc_test_v2.sto"
simulated_traj = read_sto(simulated_sto)

inferred = pd.read_csv(r"C:\Rena\nair\activation_path2_-1.79769.csv")

"""
VALIDATION METRICS
"""
activation_cols = [col for col in target_traj.columns if 'activation' in col]
target_activations = target_traj[activation_cols]
inferred_activations = simulated_traj[activation_cols]
time = target_traj['time']

# Squared activations
target_sq = target_activations **2
inferred_sq = inferred_activations**2
# Integrate effort per muscle and take total
target_effort = np.sum(np.trapezoid(target_sq, dx=0.01))
inferred_effort = np.sum(np.trapezoid(inferred_sq, dx=0.01))
print(f"SQUARED SUM OF ACTIVATIONS: SCONE trajectory = {target_effort}  Inferred trajectory = {inferred_effort}")
# Peak activation
print(f"PEAK ACTIVATION: SCONE = {target_activations.max()}  Inferred = {inferred_activations.max()}")

"""
ERROR
"""
target_knee = target_traj["/jointset/knee_r/knee_angle_r/value"]
sim_knee = simulated_traj["/jointset/knee_r/knee_angle_r/value"]
error = target_knee - sim_knee
rmse = np.sqrt(np.mean(error**2))
# MAE
mae = np.mean(np.abs(error))
print(f"MAE: {mae}")

nrmse_range = rmse / (np.max(target_knee) - np.min(target_knee))

# Normalized RMSE by mean
nrmse_mean = rmse / np.mean(target_knee)

print("RMSE:", rmse)
print("NRMSE (range):", nrmse_range)
print("NRMSE (mean):", nrmse_mean)
"""
GRAPHING
"""
def formatting():
# formatting
    plt.xlabel("Time step")
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    #plt.legend(bbox_to_anchor=(0,-0.05), loc='lower left', borderaxespad=0)
    plt.legend(loc='best', borderaxespad=0)
    plt.tight_layout()

plt.figure(figsize=(24,16))
counter = 1
# go through full df to find relevant columns
for i in range(target_traj.shape[1]):
    col_name = target_traj.columns[i]
    if "activation" in col_name:
        plt.subplot(3,4,counter)
        plt.plot(target_traj.values[:,i], label=f"target")
        plt.plot(simulated_traj.values[:,i], label=f"simulated")
        plt.title(col_name)
        formatting()
        counter += 1
    elif col_name == "/jointset/knee_r/knee_angle_r/value" or col_name=="/jointset/ankle_r/ankle_angle_r/value":
        plt.subplot(3,4,counter)
        plt.plot(target_traj.values[:,i], label=f"target")
        plt.plot(simulated_traj.values[:,i], label=f"simulated")
        # also plot raw inferred dof
        plt.plot(inferred.loc[:, col_name], label = f"Inferred")
        plt.title(col_name)
        formatting()
        counter += 1
plt.savefig("pictures/validation_results_FULL")
plt.close()








