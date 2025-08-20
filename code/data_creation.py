"""
Run knee extension optimizations on SCONE
Includes noise modulation on starting position for robustness
OUTPUT: parameter (.par) simulation files and list of all ending fitnesses

"""

import numpy as np
import time
from sconetools import sconepy
import os

# Set the SCONE log level to 3
sconepy.set_log_level(3)

# Show which SCONE version we are using
print("SCONE Version", sconepy.version())

knee_angles = []
# Generate random init angles - mean of -90, between [-120,0]
for i in range(0, 24):
	knee_angle = np.clip(np.random.normal(loc=np.deg2rad(-90), scale=np.deg2rad(15)), np.deg2rad(-120), 0)
	knee_angles = np.append(knee_angles, knee_angle)
ankle_angle = 0
print(knee_angles)

def run_sim():
    print(f"ANKLE: {ankle_angle}")
    # Load scenario 
    scenario = sconepy.load_scenario("code\sim\Knee extension data.scone");

    # Start multiple optimizations with different settings
    num_optimizers = 3
    opts = []
    for i in range(0, num_optimizers):
        # Change some settings based on i
        scenario.set("CmaOptimizer.random_seed", str(i+1))
        scenario.set("CmaOptimizer.SimulationObjective.FeedForwardController.symmetric", str(i%2))

        # Start the optimization as background task
        opts.append(scenario.start_optimization())

    # Wait for all optimizations to finish
    num_finished = 0
    while num_finished < num_optimizers:
        num_finished = 0
        time.sleep(1)
        # Iterate over all optimizers
        for i in range(0, num_optimizers):
            # Create a status string
            status = f"Optimization {i}: step={opts[i].current_step()} fitness={opts[i].fitness():.3f}";

            if opts[i].finished():
                # This optimization is finished, add to status string
                status += " FINISHED"
                num_finished += 1
            elif opts[i].current_step() >= 700:
                # Optional: terminal optimizations after 700 steps
                status += " TERMINATING"
                opts[i].terminate()

            # Print status 
            print(status, flush=True)

    # At this point, all optimizations have finished
    for i in range(0, num_optimizers):
        print(f"Finished optimization {i}: steps={opts[i].current_step()} fitness={opts[i].fitness():.3f}", flush=True)
        with open("code/fitnesses.txt", "a") as f:
            f.write(f"\n{opts[i].fitness():.3f}")

for angle in knee_angles:
	# Random ankle angles: mean of 0, between [-40, 20]
    ankle_angle = np.clip(np.random.normal(loc=0, scale=np.deg2rad(15)), np.deg2rad(-40), np.deg2rad(20))
    # write init file
    with open("code\sim\WriteableInit.zml", "w") as f:
        f.write(f"""
            values {{
                knee_angle_r = {angle}
                ankle_angle_r = {ankle_angle}
            }}
            velocities {{
                knee_angle_r = 0
                ankle_angle_r = 0
            }}""")
        
    # run optimization with scone file
    run_sim()



