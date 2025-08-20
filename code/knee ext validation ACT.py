"""
Simulates trajectory in SCONE via inferred muscle activations
Custom simulation script
"""
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from sconetools import sconepy
import sys
import os
#from read_csv import merge_csvs
import pandas as pd

# add scone gym using a relative path
"""sconegym_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../NAIR_envs')) 
sys.path.append(sconegym_path)"""
sconegym_path = r"C:\Rena\nair\NAIR_envs"
today = datetime.now().strftime('%Y-%m-%d')  # Format: YYYY-MM-DD
print(sconegym_path)


# Scone step simulation definition
def scone_step(model, muscle_activations: np.array, step):
    # step in seconds
    # set_actuator_inputs takes np array
    model.set_actuator_inputs(muscle_activations)
    model.advance_simulation_to(step)
    return model.com_pos() #center of mass

# Sconepy model initialitation
model = sconepy.load_model("code\sim\Knee_extension_validation.scone")


# read csv
trajectory = pd.read_csv("activation_path2_-1.79769.csv")
print(f"TRAJECTORY LENGTH: {len(trajectory)}")
# Configuration  of time steps and simulation time
max_time = 2 # In seconds
timestep = 0.01
timesteps = int(max_time / timestep)

# Controller loop and main function
# Get col names
times = trajectory.columns[0]
expected_knee_ang = trajectory.columns[1]
expected_ankle_ang = trajectory.columns[2]

def run_simulation(model, store_data, random_seed, max_time = 2.0):
    model.reset()
    model.set_store_data(store_data)
    print(f"Actuators: {len(model.actuators())}")

    # set initial position and muscle activations
    actions = np.zeros(len(model.muscles()))
    for i in range(3, len(trajectory.columns)):
        actions[i-3] = trajectory.iloc[0][trajectory.columns[i]]
    model.init_muscle_activations(actions)
    # dofs should be defined in the model from zml?
    model.init_state_from_dofs() #equilibrate the muscles based on activations"""

    # Initialize lists for plotting
    time_list = []
    pos_list = []

    # Start simulation
    for t in np.arange(0, max_time, 0.01):
        step = int(t*100)
        actions = np.zeros(len(model.muscles()))
        print(step)
        # locate target activations
        for i in range(3, len(trajectory.columns)):
            actions[i-3] = trajectory.iloc[step][trajectory.columns[i]]
        print(f"actions: {actions}")

        #simulate
        model_com_pos = scone_step(model, actions, t)

        time_list.append(t)
        pos_list.append(model.dof_position_array().tolist())
        print(f"dof positions: {model.dof_position_array()}")

    # Write results (.sto file that can be opened in SCONE software)
    if store_data:
        dirname = "validation_results"
        filename = f"{model.name()}_test_v2"
        model.write_results(dirname, filename)
        print(f'Results written to {dirname}/{filename}; please use SCONE Studio to replay the .sto file.', flush=True)
        print(f"{pos_list}")
    
    position_df = pd.DataFrame(pos_list, columns = model.dofs())
    return position_df
    

duration = 0.0
random_seed = 1
model_time = 0.0
start_time = time.perf_counter()
while duration < 2.0:
    pos_list = run_simulation(model, True, random_seed, max_time=2.0)
    duration = time.perf_counter() - start_time
    random_seed += 1
    model_time += model.time()




