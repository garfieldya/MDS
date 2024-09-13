#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
from gym.vector import SyncVectorEnv, AsyncVectorEnv
import wandb
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import FrameStack
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical
import random
from collections import deque
from torchvision import transforms as T
from torch import nn
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import math
import os
import itertools
import time
import cv2

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
) #required gym 0.21.0 which should be compatible with our current mario

from stable_baselines3 import PPO


# In[3]:


class Shared_CNN(nn.Module):
    def __init__ (self, input_channel, output):
        super(Shared_CNN, self).__init__()
    #CNN
        self.conv1 = nn.Conv2d(input_channel, out_channels=32, kernel_size=8, stride=4)  #becasue i change the stride from 2 to 4 so the sstate sppace rediuce from 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # from 16384 to 3136
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
#         self.layer1= nn.Linear(16384, 512)
        self.layer1= nn.Linear(3136, 512)
        self.actor = nn.Linear(512, output)
        self.critic = nn.Linear(512, 1)
        
        self.improve_init()
        
    def improve_init(self):
#         for module in self.modules():
#             if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#                 nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
#                 # nn.init.xavier_uniform_(module.weight)
#                 # nn.init.kaiming_uniform_(module.weight)
#                 nn.init.constant_(module.bias, 0)
        
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                module.bias.data.zero_()
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity="relu")
        self.layer1.bias.data.zero_()
        nn.init.xavier_uniform_(self.actor.weight)
        self.actor.bias.data.zero_()
        nn.init.xavier_uniform_(self.critic.weight)
        self.critic.bias.data.zero_()
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.layer1(x))
        actor_output = self.actor(x)
        critic_output = self.critic(x)
      
        return actor_output, critic_output


# In[ ]:





# In[8]:


class normalisation(gym.ObservationWrapper):
    def __init__(self, env):
        super(normalisation, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        return obs.__array__()/ 255.0
    
    

def get_env(custom_reward=False):
    def wrap():
        env = gym_super_mario_bros.make('SuperMarioBros-2-3-v0') #, render_mode='rgb_array', apply_api_compatibility=True
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = MaxAndSkipEnv(env, 4)#maxSkip(env)
        env = gym.wrappers.ResizeObservation(env,84)
        env = gym.wrappers.GrayScaleObservation(env)
        env = FrameStack(env, 4)
        env = normalisation(env)
        
        if custom_reward:
            env = flagReward(env)
            
          # get the episode return to make graph 
        return gym.wrappers.RecordEpisodeStatistics(env)
    return wrap


# In[7]:


learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
envs = gym.vector.SyncVectorEnv([get_env(custom_reward=False) for _ in range(4)])
agent = Shared_CNN(envs.single_observation_space.shape[0],envs.single_action_space.n).to(device)
optimiser = torch.optim.Adam(agent.parameters(),lr= learning_rate, eps=1e-5)
print(envs.single_observation_space.shape[0], envs.single_action_space.n)


# In[16]:


checkpoint_dir = "PPO_2_3_all/"
def save_checkpoint(agent, optimiser, global_steps, checkpoint_dir):
   
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint005_{global_steps}.pth")
    torch.save({
        'global_steps': global_steps,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimiser.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at step {global_steps} to {checkpoint_path}")


# In[10]:


wandb.init(project="ppo_project-2-3-2-v0", entity="chaya-ff7-durham",mode="offline")


# In[17]:




# Define the environment and parameters
num_steps = 512
num_env = 4
minibatch_size = 128
batch_size = num_steps * num_env
total_timestep = 2000000
gamma = 0.9
gea_rate = 0.95
epoch_update = 10
clip = 0.05
entro_rate = 0.01
learning_rate = 1e-4
loss_function = nn.SmoothL1Loss()

schedule_lr = False
global_steps = 0
print(f"clip rate: {clip}")
# Allocate tensors for storing data
states = torch.zeros((num_steps, num_env) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((num_steps, num_env) + envs.single_action_space.shape).to(device)
log_probs = torch.zeros((num_steps, num_env)).to(device)
old_values = torch.zeros((num_steps, num_env)).to(device)
rewards = torch.zeros((num_steps, num_env)).to(device)
dones = torch.zeros((num_steps, num_env)).to(device)

# Initialize environment and agent state
next_state = torch.tensor(envs.reset()).to(device)
next_done = torch.zeros(num_env).to(device)

# Define action selection function
def get_action(state):
    with torch.no_grad():
        logit, old_value = agent(state)
        distribution = Categorical(F.softmax(logit, dim=1))
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
    return action, log_prob, old_value

# Training loop
while global_steps <= total_timestep:
    if schedule_lr:
        discount = 1.0 - global_steps * (1.0 / total_timestep)
        update_lr = discount * learning_rate
        optimiser.param_groups[0]["lr"] = update_lr
        print(f"global_step: {global_steps}")
    
    for i in range(num_steps):
        global_steps += num_env
        if global_steps % 1000000 == 0:
            print("time to save the checkpoint")
            save_checkpoint(agent, optimiser, global_steps, checkpoint_dir)


        states[i] = next_state
        dones[i] = next_done
        
        action, logit, old_value = get_action(next_state)
        
        old_values[i] = old_value.flatten().detach()
        actions[i] = action
        log_probs[i] = logit.detach()
        
        next_state, reward, done, info = envs.step(action.cpu().numpy())
        
        rewards[i] = torch.tensor(reward).to(device).view(-1)
        next_done = torch.tensor(done).to(device)
        next_state = torch.tensor(next_state).to(device)
        
        for env_index, item in enumerate(info):
            if "episode" in item.keys():
                wandb.log({
                    f"reward_env_{env_index}": item["episode"]["r"],  # Directly log the total reward
                    "global_steps": global_steps
                })
        
        
        
        
    with torch.no_grad():
        _, last_value = agent(next_state)
        advantages = torch.zeros_like(rewards).to(device)
        last_gae = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_value = last_value.reshape(1, -1)
                done_terminal = 1.0 - next_done.float()
            else:
                next_value = old_values[t + 1]
                done_terminal = 1.0 - dones[t + 1].float()
            
            delta = rewards[t] + gamma * next_value * done_terminal - old_values[t]
            advantages[t] = last_gae = delta + gamma * gea_rate * done_terminal * last_gae
        
        returns = advantages + old_values

    b_states = states.reshape((-1,) + envs.single_observation_space.shape)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_log_probs = log_probs.reshape(-1).detach()
    b_old_values = old_values.reshape(-1).detach()
    b_advantages = advantages.reshape(-1).detach()
    b_returns = returns.reshape(-1).detach()
    
    index = np.arange(batch_size)
    
    for epoch in range(epoch_update):
        np.random.shuffle(index)
        for k in range(0, batch_size, minibatch_size):
            batch = index[k:k + minibatch_size]
            
            new_logit, new_value = agent(b_states[batch])
            new_dist = Categorical(F.softmax(new_logit, dim=1))
            new_log_prob = new_dist.log_prob(b_actions.long()[batch])
            entropy_loss = new_dist.entropy().mean()
            
            change_ratio = (new_log_prob - b_log_probs[batch]).exp()
            normalized_adv = (b_advantages[batch] - b_advantages[batch].mean()) / (b_advantages[batch].std() + 1e-8)
            
            unclipped_loss = change_ratio * -normalized_adv
            clipped_loss = torch.clamp(change_ratio, 1 - clip, 1 + clip) * -normalized_adv
            actor_loss = torch.max(unclipped_loss, clipped_loss).mean()
            critic_loss = 0.5 * loss_function(new_value.view(-1), b_returns[batch])
            loss = actor_loss + critic_loss - entro_rate * entropy_loss
            
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimiser.step()
            
            for name, param in agent.named_parameters():
                if param.requires_grad:
                    abs_weights = torch.abs(param)  # Compute absolute values of weights
                    average_abs_weight = torch.mean(abs_weights).item()  # Calculate the average
                    wandb.log({f"Average Absolute Weight/{name}": average_abs_weight,
                               "global_steps": global_steps})  # Log to wandb

            wandb.log({
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "learning_rate": optimiser.param_groups[0]["lr"],
            "global_steps": global_steps
        })
    
            
wandb.finish()


# In[ ]:





# In[24]:


# envs = gym.vector.SyncVectorEnv([get_env(custom_reward=False) for _ in range(4)])  
# envs.reset()
        
# actions = envs.action_space.sample()
# next_state, reward, done, info = envs.step(actions)

# _, _, value = get_action(torch.tensor(next_state, dtype=torch.float32).to(device))      


# In[29]:


# value.shape, value.reshape(1, -1).shape, value.view(-1).shape


# In[69]:


# def wrap():
#         env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0') #, render_mode='rgb_array', apply_api_compatibility=True
#         env = JoypadSpace(env, SIMPLE_MOVEMENT)
#         env = MaxAndSkipEnv(env, 4)#maxSkip(env)
#         env = gym.wrappers.ResizeObservation(env,84)
#         env = gym.wrappers.GrayScaleObservation(env)
#         env = FrameStack(env, 4)
#         env = normalisation(env)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
        
# #         if custom_reward:
# #             env = flagReward(env)
            
#           # get the episode return to make graph 
#         return env
# env = wrap()   
# reward_list = []
# epi =[]
# for episode in range(2):
#     state = env.reset()
#     reward_epi = 0
#     done = False

#     while not done:
#         action = env.action_space.sample()
#         state, reward, done, info = env.step(action)
            
#         reward_epi += reward
#         if done:
#             print(f"Episode {episode + 1} Reward: {reward_epi}")
#             epi.append(info["episode"]["r"])
#             reward_list.append(reward_epi)
#             break

# # Close the environment to save the monitor data
# env.close()  


# In[39]:


# x = torch.zeros((num_step, num_env)+envs.single_observation_space.shape)
# x.shape, x.reshape((-1,)+envs.single_observation_space.shape).shape


# In[45]:


# x = ["good", "bad", "normal", "idk"]
# for env_idx, item in enumerate(x):
#     print(env_idx, item)

