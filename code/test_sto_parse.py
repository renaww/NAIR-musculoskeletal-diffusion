"""
Some utility functions to deal with .sto storage files (default output from SCONE)
"""
import pandas as pd
import os
from io import StringIO

# Used for testing
file_path = "C:/Users/renaa/Documents/SCONE/results/knee ext DATA/clean/250620.124259.leg6dof9musc.FC2.MW.D2.R1/0400_10821.429_10779.872.par.sto"

# IN: storage file
# OUT: Dataframe
def read_sto(sto_path):
    with open(sto_path, 'r') as file:
        lines = file.readlines()
    # Find start of data
    header_end_index = next(i for i, line in enumerate(lines) if 'endheader' in line.lower()) + 1
    data_str = ''.join(lines[header_end_index:])
    new_df = pd.read_csv(StringIO(data_str), sep='\s+')
    return new_df
# IN: storage file
# OUT: csv
def convert_sto_to_csv(sto_path, csv_path, append=False):
    new_df = read_sto(sto_path)

    if append and os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path, sep=',')
        # Optional: check if columns match
        if list(existing_df.columns) != list(new_df.columns):
            #print(list(existing_df.columns))
            #print(list(new_df.columns))
            print(f"existing length: {len(list(existing_df.columns))}   new length: {len(list(new_df.columns))}")
            print("Column mismatch between existing CSV and new .sto data.")

            # Drop all columns that have "penalty"
            cols_to_drop = ['knee_angle_r.position_penalty', 'knee_angle_r.acceleration_penalty', 'ankle_angle_r.velocity_penalty', 'ankle_angle_r.acceleration_penalty', 'Effort.penalty', 'Effort.bifemlh_r.penalty', 'Effort.bifemsh_r.penalty', 'Effort.glut_max2_r.penalty', 'Effort.psoas_r.penalty', 'Effort.rect_fem_r.penalty', 'Effort.vas_int_r.penalty', 'Effort.med_gas_r.penalty', 'Effort.soleus_r.penalty', 'Effort.tib_ant_r.penalty'] 
            if len(list(existing_df.columns)) > len(list(new_df.columns)):
                existing_df = existing_df.drop(cols_to_drop, axis=1, inplace=False)
            else: 
                new_df = new_df.drop(cols_to_drop, axis=1, inplace=False)

        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df.to_csv(csv_path, index=False)
        print(f"Added to {csv_path}")
    else:
        new_df.to_csv(csv_path, index=False)

