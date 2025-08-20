# https://huggingface.co/blog/annotated-diffusion
# https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B?usp=sharing#scrollTo=X-XRB_g3vsgf

"""
Code to perform inference based on pretrained states
Includes a copy of the architecture that is the same as Unet_1d.py
"""
if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch import einsum
    import torchvision
    from torchvision import datasets, transforms
    import torchvision.utils as vutils

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    from einops import rearrange, reduce
    from einops.layers.torch import Rearrange
    import copy
    import math
    from inspect import isfunction
    from functools import partial
    from typing import Union
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.training_utils import EMAModel
    from diffusers.optimization import get_scheduler
    from tqdm.auto import tqdm
    import pandas as pd
    import collections
    import copy


    import torch_directml
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    from test_sto_parse import read_sto
    from chunking import pred_horizon, obs_horizon, action_horizon, dim, target_cols, stats, normalize_data, unnormalize_data

    # Setting a global seed for reproducibility
    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
    set_seed(42)

    # PARAMETERS - CHANGE IF DATA IS CHANGED
    obs_dim = dim
    action_dim = dim

    # HELPER FUNCTIONS
    def exists(x):
        return x is not None

    def default(val, d):
        if exists(val):
            return val
        return d() if isfunction(d) else d

    def num_to_groups(num, divisor):
        groups = num // divisor
        remainder = num % divisor
        arr = [divisor] * groups
        if remainder > 0:
            arr.append(remainder)
        return arr

    """POSITION EMBEDDING: encodes time step (noise level).
        batch size: # of timesteps | one dimension: one sin/cos pair
    """
    class SinusoidalPositionEmbeddings(nn.Module):
        def __init__(self, dim):
            #subclass of module class
            super().__init__()
            self.dim = dim
        
        # get embeddings, time is a tensor with # of timestpes as rows
        def forward(self, time):
            device = time.device  #ensure same device as time
            half_dim = self.dim//2  #half for sin, half for cos
            exponent = (math.log(10000) / (half_dim - 1))  #scaling factor of frequencies
            # arange: evenly spaced values within specific range
            frequencies = torch.exp(torch.arange(half_dim, device = device) * exponent) #range of frequencies in log scale
            args = time[:, None] * frequencies[None, :] #args[i][j] = timesteps[i] * freqs[j]
            embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) #dim=-1 means sticking together along last dimension, horiontally
            return embeddings

    """UP/DOWNSAMPLING: Change temporal resolution. Ex-upsampling captures increasingly abstract features
        Done via convolution (slide across features to create feature map)
        torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    """
    class Downsample1d(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

        def forward(self, x):
            return self.conv(x)

    class Upsample1d(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

        def forward(self, x):
            return self.conv(x)

    """FEATURE EXTRACTION LAYER: Conv1d --> GroupNorm --> Mish
    """
    class Conv1dBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, n_groups=8):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding = kernel_size//2),
                nn.GroupNorm(n_groups, out_channels),
                nn.Mish(),
            )
        
        def forward(self, x):
            return self.block(x)

    """RESNET: Multiple blocks + conditioning + skip connections
        FiLM modulation uses code from https://arxiv.org/abs/1709.07871 
    """
    class FiLMParam(nn.Module):
        # Conditioning input -> FiLM Parameters
        def __init__(self, dim, num_features):
            """ dim: input dimension; 
            num_features: number of channels per spacial location, how many scale/shift values to produce"""
            super().__init__()
            self.gamma = nn.Linear(dim, num_features)
            self.beta = nn.Linear(dim, num_features)

        def forward(self, cond_input):
            gamma = self.gamma(cond_input) # shape: [batch size x num features]
            beta = self.beta(cond_input)
            return gamma, beta
        
    class FiLMLayer(nn.Module):
        # Actual application of FiLM to feature map
        def __init__(self, num_features):
            super(FiLMLayer, self).__init__()
            self.num_features = num_features

        def forward(self, features, gamma, beta):
            # Features: feature map with size [batch_size, num_features = out_channels, L]
            gamma = gamma.unsqueeze(2) #Shape: [B, C, 1]
            beta = beta.unsqueeze(2)
            # Apply transformation
            return gamma * features + beta

    class ConditionalResnetBlock(nn.Module):
        def __init__(self, dim, dim_out, cond_dim, kernel_size=3, time_emb_dim=None, groups=8): #time_emb_dim for conditioning
            super().__init__()
            self.block1 = Conv1dBlock(dim, dim_out, kernel_size, groups)
            self.block2 = Conv1dBlock(dim_out, dim_out, kernel_size, groups)

            # num_features = dim_out
            self.film_params = FiLMParam(cond_dim, dim_out)
            self.film = FiLMLayer(dim_out)

            # check dimension compatibility
            self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        
        def forward(self, x, cond):
            '''
                x : [ batch_size x in_channels x horizon ]
                cond : [ batch_size x cond_dim] | applied to x with FiLM conditioning

                returns:
                out : [ batch_size x out_channels x horizon ]
            '''
            #print("x shape: ", x.shape)
            out = self.block1(x)
            #print("after block 1: ", out.shape)
            beta, gamma = self.film_params(cond)
            out = self.film(out, gamma, beta)
            #print("after film: ", out.shape)
            # second block - extra refinement
            out = self.block2(out)
            #print("after block2: ", out.shape)
            # skip connection F(x)+x -> allows stability, can reuse features from earlier layers
            out = self.res_conv(x) + out
            #print("after skip connection: ", out.shape)
            return out
            
    """ ATTENTION MODULE? Implemented in 2D"""

    """ THE ACTUAL UNET"""
    class ConditionalUnet1D(nn.Module):
        def __init__(self, input_dim, global_cond_dim, dsed=256, down_dims=[128, 256, 512, 1024], kernel_size=3, n_groups=8):
            """
            input_dim: Dim of actions.
            global_cond_dim: Dim of global conditioning applied with FiLM in addition to diffusion step embedding. 
            diffusion_step_embed_dim (dsed): Size of positional encoding for diffusion iteration k
            down_dims: Channel size for each UNet level.
            The length of this array determines numebr of levels.
            """
            super().__init__()
            # Determine dimensions
            all_dims = [input_dim] + list(down_dims)
            start_dim = down_dims[0]
            # Time embeddings
            time_emb = dsed * 4   # higher dimension
            self.diffusion_step_encoder = nn.Sequential(
                SinusoidalPositionEmbeddings(dsed),
                nn.Linear(dsed, time_emb),
                nn.Mish(),
                nn.Linear(time_emb, dsed)
            )
            cond_dim = dsed + global_cond_dim
            # Pair up dimension with the next (to connect layers)
            in_out = list(zip(all_dims[:-1], all_dims[1:]))
            # layers
            self.downs = nn.ModuleList([])
            self.ups = nn.ModuleList([])
            num_resolutions = len(in_out)
            # Turnaround point (bottleneck)
            mid_dim = all_dims[-1]
            self.mids = nn.ModuleList([
                ConditionalResnetBlock(mid_dim, mid_dim, cond_dim),   #kernel size = 3, groups = 8
                ConditionalResnetBlock(mid_dim, mid_dim, cond_dim)
            ])
            # Downsampling
            for index, (dim_in, dim_out) in enumerate(in_out):
                is_last = index >=(num_resolutions - 1)  # check if index is last one
                self.downs.append(nn.ModuleList([
                    ConditionalResnetBlock(dim_in, dim_out, cond_dim),
                    ConditionalResnetBlock(dim_out, dim_out, cond_dim),
                    Downsample1d(dim_out) if not is_last else nn.Identity()
                ])) 
            # Upsampling
            for index, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):  # First item has no feature map to upsample from
                is_last = index >= num_resolutions -1
                self.ups.append(nn.ModuleList([
                    ConditionalResnetBlock(dim_out*2, dim_in, cond_dim),  #dim_out*2 because skip connections concatenated
                    ConditionalResnetBlock(dim_in, dim_in, cond_dim),
                    Upsample1d(dim_in) if not is_last else nn.Identity()
                ]))
            # Go back to original input dimension
            self.final_conv = nn.Sequential(
                Conv1dBlock(start_dim, start_dim, kernel_size = 3),
                nn.Conv1d(start_dim, input_dim, 1)
            )
            # test
            print("number of parameters: {:e}".format(
                sum(p.numel() for p in self.parameters()))
            )
        
        def forward(self, sample:torch.Tensor, timestep:Union[torch.Tensor, float, int], global_cond=None):
            """
            x: (B,T,input_dim)
            timestep: (B,) or int, diffusion step
            global_cond: (B,global_cond_dim)
            output: (B,T,input_dim)
            """
            # (B,T,C) -> (B,C,T)
            sample = sample.moveaxis(-1, -2)
            # Time
            timesteps = timestep
            if not torch.is_tensor(timestep):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
            elif torch.is_tensor(timestep) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)
            # expand timesteps to batch dimension (repetition)
            timesteps = timesteps.expand(sample.shape[0])
            global_feature = self.diffusion_step_encoder(timesteps)

            if global_cond is not None:
                global_feature = torch.cat([global_feature, global_cond], axis=-1)  # include timesteps
            
            x = sample  # initial convolution
            h = []
            # perform actual sampling
            for index, (block1, block2, downsample) in enumerate(self.downs):
                #print("DOWNSAMPLING")
                x = block1(x, global_feature)
                x = block2(x, global_feature)
                h.append(x)
                x = downsample(x)
            for mid_module in self.mids:
                #print("MID")
                x = mid_module(x, global_feature)
            for index, (block1, block2, upsample) in enumerate(self.ups):
                #print("UPSAMPLING")
                # h.pop() retrieves feature map
                skip = h.pop()
                #print(f"x shape: {x.shape}  |  skip shape: {skip.shape}")
                x = torch.cat((x, skip), dim = 1)
                x = block1(x, global_feature)
                x = block2(x, global_feature)
                x = upsample(x)
            x = self.final_conv(x)
            # Rearrange back to original size
            x = x.moveaxis(-1, -2)
            return x
        
    """ NEED DEVICE TRANSFER?
    # device transfer
    device = torch.device('cuda')
    _ = noise_pred_net.to(device)
    """

    # Create network object
    noise_pred_net = ConditionalUnet1D(input_dim = action_dim, global_cond_dim=obs_dim * obs_horizon)

    # From diffuser package - imporve performance and speed
    num_diffusion_iters = 105
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type = 'epsilon' # predicts noise instead of action
    )
    _ = noise_pred_net.to(device)

    """SET UP MODEL"""
    num_epochs = 150
    losses = []
    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=noise_pred_net.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # LOAD PRETRAINED
    ema_states = torch.load("activation_model_states.pt", weights_only=True)
    #print("ema_states: ", ema_states)
    ema.load_state_dict(ema_states)
    print("Weights loaded")

    ema_noise_pred_net = copy.deepcopy(noise_pred_net)
    # replace parameters of ema_noise_pred_net with those of ema
    ema.copy_to(ema_noise_pred_net.parameters())

    """
    INFERENCE
    """
    max_steps = 200
    # get first observation  |  Format: np array of positions--columns in chunking.py
    # order of columns: knee_r, ankle_r, patella (pos + ori), talus (pos + ori)
    # change to fit custom init positions
    sto_location = r"C:\Users\renaa\Documents\SCONE\results\250723.144821.leg6dof9musc.FC2.MW.D2\0339_11677.804_11460.316.par.sto"
    all_positions = read_sto(sto_location)
    obs = all_positions[target_cols].iloc[0]
    print(f"starting pos: {obs}")
    starting_knee = obs['/jointset/knee_r/knee_angle_r/value']

    # keep queue of last obs_horizon=2 steps of observation
    obs_deque = collections.deque([obs]*obs_horizon, maxlen=obs_horizon)
    print("Obs deque: ", obs_deque)

    # save rewards/log position?
    rewards = list()
    # Generate path
    done = False
    step_idx = 0
    final_path = list()
    #final_path.append(obs)
    with tqdm(total=max_steps, desc="Eval") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(obs_deque)
            print(obs_seq)
            # normalize observation
            #print("STATS: ", stats['obs'])
            nobs = []
            for i in range(len(obs_seq)):
                nobs.append(normalize_data(obs_seq[i], stats=stats['obs']))
            nobs = np.array(nobs)
            #print("NOBS", nobs)
            
            # device transfer
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            # infer action
            with torch.no_grad(): # reduce memory consumption
                # (B, obs_horizon*obs_dim)
                obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)
                # initialize gaussian noise
                noisy_action = torch.randn((B, pred_horizon, action_dim), device=device)
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for t in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_noise_pred_net(sample=noisy_action, timestep=t, global_cond=obs_cond)
                    # remove noise
                    noisy_action = noise_scheduler.step(model_output=noise_pred, timestep=t, sample=noisy_action).prev_sample
                # get inner values
                naction = noisy_action.detach().to('cpu').numpy()
                print("Predicted std:", naction.std(), "Mean:", naction.mean())
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                print("NACTION: ", naction)
                #action_pred = unnormalize_data(naction, stats=stats['action'])

                # take action horizon number of actions
                start = obs_horizon-1
                end = start + action_horizon
                action = naction[start:end,:]  #start:end rows, all cols, action has shape (action_horizon, action_dim)
                action_pred=[]
                for step in action:
                    action_pred.append(unnormalize_data(step, stats=stats['action']))
                print('ACTION PRED: ', action_pred)
            

                # execute action_horizon number of actions
                for obs in action_pred:
                    obs_deque.append(obs)
                    final_path.append(obs)
                    step_idx += 1
                    pbar.update(1)
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break
    # Output trajectory: dof positions and all activations
    final_path_df = pd.DataFrame(final_path)
    final_path_df.to_csv(f"activation_path2_{starting_knee}.csv")

    # Plot each action dim over time
    plt.figure(figsize=(20, 14))
    for i in range(final_path_df.shape[1]):
        plt.plot(final_path_df.values[:, i], label=final_path_df.columns[i])
    plt.title(f"Predicted Action Trajectory")
    plt.xlabel("Time step")
    plt.ylabel("Action")
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(f"pictures/new_inference_startingangle={starting_knee}.png")
    plt.close()

    


    
    





