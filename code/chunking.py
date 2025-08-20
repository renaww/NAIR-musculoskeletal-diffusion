"""
Normalization functions, data chunking, and creation of PyTorch Dataset object
Based on: https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B?usp=sharing#scrollTo=X-XRB_g3vsgf

"""

import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Setting a global seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
set_seed(42)

# Replace with custom dataset
data_path = "full_dataset.csv"
existing_df = pd.read_csv(data_path, sep=',')

# normalize data - single array
def get_data_stats(data):
    stats = {
        'min': data.min(axis=0),
        'max': data.max(axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    print(f"Normalized data, mean = {ndata.mean(axis=0)}")
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# Obtain indices of segments with length = the prediction horizon
def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            # buffer: includes padding
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

# Obtain chunked data based on indices
def sample_sequence(train_data, sequence_length,buffer_start_idx, buffer_end_idx,sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        input_arr = input_arr.to_numpy()
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=np.float64)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            expected_shape = sample_end_idx - sample_start_idx
            actual_shape = sample.shape[0]
            if expected_shape != actual_shape:
                raise ValueError(f"Mismatch: expected shape {expected_shape}, got {actual_shape}")

            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

""" SET UP TRAINING AND MODEL ENVIRONMENT, parameters referenced in the unet"""
# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 4

target_cols = []
# OLD
"""for col_name in list(existing_df.columns):
    if any(s in col_name for s in ("patella_r.com_pos_","patella_r.ori_","talus_r.ori_","talus_r.com_pos_")):
        target_cols.append(col_name) 
    elif col_name == "/jointset/knee_r/knee_angle_r/value" or col_name=="/jointset/ankle_r/ankle_angle_r/value":
        target_cols.append(col_name)"""

# Updated: muscle activations + dof position
for col_name in list(existing_df.columns):
    if "activation" in col_name or col_name == "/jointset/knee_r/knee_angle_r/value" or col_name=="/jointset/ankle_r/ankle_angle_r/value":
        target_cols.append(col_name) 
print(target_cols)
dim = len(target_cols)

class JointDataset(Dataset):
    def __init__(self, raw_file, pred_horizon, obs_horizon, action_horizon):
        # Set titles of columns to pay attention to 
        # (N, action/obs dimension)
        obs = raw_file[target_cols]
        act = raw_file[target_cols]
        train_data = {'action': act, 'obs': obs}
        # Get indices of end of each episode
        time = raw_file["time"].values
        # Find when episode starts, remove first value and append length to get 1 past indices for the ends
        episode_starts = np.where(time < 0.006)[0]
        episode_ends = np.append(episode_starts[1:], len(raw_file))

        # Start and end of state-action sequence
        indices = create_sample_indices(episode_ends, pred_horizon, pad_before=obs_horizon-1, pad_after=action_horizon-1)

        # Normalize data via computing of stats. Normalized to [-1, 1]
        stats = dict()
        normalized_train_data = dict()
        for data_title, data in train_data.items():
            print(data_title)
            stats[data_title] = get_data_stats(data)
            normalized_train_data[data_title] = normalize_data(data, stats[data_title])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        # all possible segments of the dataset
        return len(self.indices)

    def __getitem__(self, idx):
        # Get full array of start/ends
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        # Get normalizd data
        nsample = sample_sequence(self.normalized_train_data, self.pred_horizon, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        # Discard unused observations?
        return nsample

dataset = JointDataset(existing_df, pred_horizon, obs_horizon, action_horizon)
stats = dataset.stats # Save stats

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size = 128,
    num_workers = 4,
    shuffle = True,
    pin_memory = True,
    persistent_workers = True
)
