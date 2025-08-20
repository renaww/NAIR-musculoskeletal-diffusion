"""
Gets coordinates from SCONE optimizations via simulation
- Reads SCONE results and runs simulations
- Collects resulting .sto files into csv format
Modified from sconepy tutorial code
"""

import numpy as np
from scipy.stats import truncnorm
import time
from sconetools import sconepy
import os
import random
import test_sto_parse

file_paths = []
# Replace with SCONE results folder
results = "C:/Users/renaa/Documents/SCONE/results/knee ext DATA/clean"

# Function that loads a model and runs a simulation
def run_simulation(model, use_neural_delays, store_data, random_seed, max_time=3, filename="", dirname=""):
	# reset the model to the initial state
	model.reset()
	model.set_store_data(store_data)

	# initialize the rng
	rng = np.random.default_rng(random_seed)

	# Initialize muscle activations to random values
	muscle_activations = 0.1 + 0.4 * rng.random((len(model.muscles())))
	model.init_muscle_activations(muscle_activations)

	model.init_state_from_dofs()

 	# Start the simulation, with time t ranging from 0 to 5
	for t in np.arange(0, max_time, 0.01):

		# The model inputs are computed here
		# We use an example controller that mimics basic monosynaptic reflexes
		# by simply adding the muscle force, length and velocity of a muscle
		# and feeding it back as input to the same muscle
		if use_neural_delays:
			# Read sensor information WITH neural delays
			mus_in = model.delayed_muscle_force_array()
			mus_in += model.delayed_muscle_fiber_length_array() - 1.2
			mus_in += 0.1 * model.delayed_muscle_fiber_velocity_array()
		else:
			# Read sensor information WITHOUT neural delays
			mus_in = model.muscle_force_array()
			mus_in += model.muscle_fiber_length_array() - 1
			mus_in += 0.2 * model.muscle_fiber_velocity_array()

		# The muscle inputs (excitations) are set here
		if use_neural_delays:
			# Set muscle excitation WITH neural delays
			model.set_delayed_actuator_inputs(mus_in)
		else:
			# Set muscle excitation WITHOUT neural delays
			model.set_actuator_inputs(mus_in)

		# Advance the simulation to time t
		# Internally, this performs as many simulations steps as required
		# The internal step size is variable, and determined by the 'accuracy'
		# setting in the .scone file
		model.advance_simulation_to(t)
		
	# Write results to sto
	if store_data:
		model.write_results(dirname, filename)
		print(f"Results written to {dirname}/{filename}", flush=True)

# Find the par file corresponding to finished optimization
# Replace with list from fitnesses.txt
fitnesses = [10450.716, 10643.442, 10290.265, 10566.320, 6992.650, 7097.148, 11313.161, 11391.458, 8706.460, 8688.261, 7449.706, 7422.901, 8496.071, 11409.277, 11409.091, 6215.130, 6334.632, 8223.287, 8193.981, 8235.116, 8811.740, 8983.999, 8870.090, 8114.009, 8904.813, 8023.975, 9651.397, 9761.141, 10007.039, 8795.681, 8731.637, 8685.566, 9439.824, 9449.572, 9746.551, 8844.618, 8979.881, 9681.664, 11343.155, 11459.874, 11282.779, 8090.040, 8022.613, 8100.508, 10303.563, 10337.097, 10453.387, 10857.082, 10950.80]

for curr_fitness in fitnesses:
	# Search SCONE results directory
    for root, dirs, files in os.walk(results):
        for file in reversed(files):
            # print(f"Pathname: {os.path.basename(file)}  |   File: {file}")
			# No need to run simulation for .sto files that already existed (ie. simulation manually ran in SCONE GUI)
            if (f"{curr_fitness:.3f}") in file and file.endswith(".sto"):
                print(f"Found: {file}")
                file_path = f"{os.path.basename(root)}/{file}"
                print(f"PATH: {file_path}")
                file_paths.append(file_path)
                break
            elif (f"{curr_fitness:.3f}") in file and file.endswith(".par"):  # If .sto doesn't already exist
                file_path = f"{os.path.basename(root)}/{file}"
                print(file_path)
				# Run simulation, initializing with optimized parameter file
                model = sconepy.load_model("code\sim\Knee extension sim", f"{results}/{file_path}")
                random_seed = random.randrange(1, 50)
                duration = 0.0
                start_time = time.perf_counter()
				# Ends simulation after 2.5 secs
                while duration < 2.5:
                    run_simulation(model, False, True, random_seed, max_time=2.5, filename=file, dirname=os.path.basename(root))
                    duration = time.perf_counter() - start_time
                    random_seed += 1
                print(f"Simulation finished, found path {file_path}")
                file_paths.append(file_path)
                break
			
print(file_paths)
# Add sto to csv
for path in file_paths:
	test_sto_parse.convert_sto_to_csv(f"{results}/{path}", "full_dataset.csv", append=True)
	print(f"Added {path} to csv")				
				