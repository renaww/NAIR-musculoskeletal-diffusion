# Based on:
# https://huggingface.co/blog/annotated-diffusion
# https://colab.research.google.com/drive/1gxdkgRVfM55zihY9TFLja97cSVZOZq2B?usp=sharing#scrollTo=X-XRB_g3vsgf

"""
1D UNET Architecture and training
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
    import copy

    # Make directory to save training checkpoints
    os.makedirs('checkpoints', exist_ok=True)


    # Change to GPU if on Nvidia 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    from test_sto_parse import read_sto
    from chunking import dataloader, obs_horizon, dim, target_cols

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
            
    """ ATTENTION MODULE (WIP) Implemented in 2D"""

    """ THE ACTUAL UNET"""
    class ConditionalUnet1D(nn.Module):
        def __init__(self, input_dim, global_cond_dim, dsed=256, down_dims=[128, 256, 512, 1024], kernel_size=3, n_groups=8):
            """
            input_dim: Dim of actions.
            global_cond_dim: Dim of global conditioning applied with FiLM in addition to diffusion step embedding
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

    """TRAINING"""
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
    
    # Cosine LR scheduler
    lr_scheduler = get_scheduler(name='cosine', optimizer=optimizer, num_warmup_steps=500, num_training_steps=len(dataloader)*num_epochs)

    with tqdm(range(num_epochs), desc = "Epoch") as tglobal:
        for epoch_ind in tglobal:
            epoch_loss = list()
            # loop through by batch
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for batch in tepoch:
                    # device transfer
                    curr_obs = batch['obs'].to(device)
                    curr_action = batch['action'].to(device)
                    batch_size = curr_obs.shape[0]

                    # use observations for FiLM conditioning, change size
                    obs_cond = curr_obs[:,:obs_horizon,:] #(B, obs_horizon, obs_dim)
                    obs_cond = obs_cond.flatten(start_dim = 1) #(B, obs_horizon*obs_dim)
                    # sample noise
                    noise = torch.randn(curr_action.shape, device=device)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device="cpu").long()
                    # work with diffusers ddpm scheduler
                    #timesteps = timesteps.to("cpu")

                    # forward diffusion: add noise at each step
                    print("curr_action:", curr_action.shape, curr_action.device)
                    print("noise:", noise.shape, noise.device)
                    print("timesteps:", timesteps.shape, timesteps.device, timesteps.dtype)

                    noisy_actions = noise_scheduler.add_noise(curr_action, noise, timesteps)
                    # predict the noise residual (apply unet)
                    noisy_actions = noisy_actions.float()
                    obs_cond = obs_cond.float()
                    noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    # optimize
                    loss.backward()  # previous loss values
                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(noise_pred_net.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    # step lr scheduler
                    lr_scheduler.step()

                    # update Exponential Moving Average of the model weights
                    ema.step(noise_pred_net.parameters())
                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            mean_loss = np.mean(epoch_loss)
            tglobal.set_postfix(loss=mean_loss)
            losses.append(mean_loss)

            """DEBUG: visualize rollout"""
            if epoch_ind % 3 == 0:
                with torch.no_grad():
                    B = 1
                    sample_obs = curr_obs[0:1, :obs_horizon, :]  # shape (1, H, D)
                    sample_obs_cond = sample_obs.flatten(start_dim=1).float()  # float32

                    noisy_action = torch.randn((1, curr_action.shape[1], curr_action.shape[2]), device=device).float()

                    # Create noisy starting point
                    noisy_action = torch.randn((B, curr_action.shape[1], curr_action.shape[2]), device=device)
                    noise_scheduler.set_timesteps(num_diffusion_iters)
                    for t in noise_scheduler.timesteps:
                        noise_pred = noise_pred_net(noisy_action, timestep=t, global_cond=sample_obs_cond)
                        noisy_action = noise_scheduler.step(model_output=noise_pred, timestep=t, sample=noisy_action).prev_sample

                    # (B, T, D)
                    pred_action = noisy_action[0].detach().cpu().numpy()

                    # Plot each action dim over time
                    plt.figure(figsize=(10, 4))
                    for i in range(pred_action.shape[1]):
                        plt.plot(pred_action[:, i], label=f"Dim {i}")
                    plt.title(f"Predicted Action Trajectory – Epoch {epoch_ind}")
                    plt.xlabel("Time step")
                    plt.ylabel("Action")
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f"pictures/act_training_rollout/debug_rollout_epoch_{epoch_ind}.png")
                    plt.close()
                
                # save checkpoint in case crash
                checkpoint = {
                    'epoch': epoch_ind,
                    'model_state_dict': noise_pred_net.state_dict(),
                    'ema_state_dict': ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': mean_loss
                }

                torch.save(checkpoint, f'checkpoints/model_epoch_{epoch_ind}.pt')


    print(f"LOSSES: {losses}")
    # Weights of the EMA model
    # is used for inference
    ema_noise_pred_net = copy.deepcopy(noise_pred_net)
    ema.copy_to(ema_noise_pred_net.parameters())

    # SAVE the state
    torch.save(ema.state_dict(), "activation_model_states.pt")

    
    





