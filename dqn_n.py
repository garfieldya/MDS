#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
from collections import namedtuple, deque
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


# In[2]:


class normalisation(gym.ObservationWrapper):
    def __init__(self, env):
        super(normalisation, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        return obs.__array__()/ 255.0
    
    

def wrap(custom_reward=False):
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


# In[5]:


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))

class ReplayMemory:

    def __init__(self, max_memory):
        self.memory = deque(maxlen=max_memory)

    def push(self, state, action, next_state, reward, done):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)

    # Ensure action is of shape [1, 1] before storing
        if action.dim() == 1:
            action = action.unsqueeze(1)
    
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        self.memory.append(Transition(state, action, next_state, reward, done))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# In[7]:


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)


        self.layer1 = nn.Linear(3136, 512)
        self.layer2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)  # Flatten starting from dimension 1 after the batch
        #linear layers
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

        


# In[6]:


def chosen_action(state):
    global step_done
    sample = random.random()
    eps_threshold = epsi_end + (epsi_start - epsi_end) * math.exp(-1 * (step_done / decay_rate))

    step_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            action = dqn_policy(state).argmax(dim=1, keepdim=True)
    else:
        action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    return action


# In[8]:


def optimiser_fn():
    if len(replay_buf) < step_required:
        return
    if count% update_frequent==0:
        target_policy.load_state_dict(dqn_policy.state_dict())
    batch = replay_buf.sample(batch_size)
    batch_memory = Transition(*zip(*batch))

    state_cat = torch.cat(batch_memory.state).to(device)
    action_cat = torch.cat(batch_memory.action).to(device)
    next_state_cat = torch.cat(batch_memory.next_state).to(device)
    reward_cat = torch.cat(batch_memory.reward).to(device)
    done_cat = torch.cat(batch_memory.done).to(device)

    if action_cat.dim() == 1:
        action_cat = action_cat.unsqueeze(1)

    value_state_action = dqn_policy(state_cat).gather(1, action_cat)

    with torch.no_grad():
        value_next_state = target_policy(next_state_cat).max(1)[0]
        expected_value = reward_cat + (gamma * value_next_state * (1 - done_cat.float()))

    loss = loss_fn(value_state_action, expected_value.unsqueeze(1))  # TD error
    loss_total.append(loss)

    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(dqn_policy.parameters(), 0.5)
    optimiser.step()


# In[9]:


def get_plot(episode, graph_directory, show_result=False):
    plt.figure()
    accu_reward =torch.tensor(episode_rewards, dtype=torch.float)
    durations_t = torch.tensor(episode_done, dtype=torch.float)
    episodes = torch.arange(len(episode_done), dtype=torch.float)

    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')

    plt.xlabel('Episode')
    plt.ylabel('Duration/Reward')

    # Plot durations
    plt.plot(episodes.numpy(), accu_reward.numpy(), label='Rewards')
    plt.plot(episodes.numpy(), durations_t.numpy(), label='Duration')

    # Compute and plot a moving average of durations if there are enough data points
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        reward_means = accu_reward.unfold(0, 100, 1).mean(1).view(-1)
        reward_means = torch.cat((torch.zeros(99), reward_means))

        plt.plot(episodes.numpy(), means.numpy(), label='Moving Average (100 episodes)')
        plt.plot(episodes.numpy(), reward_means.numpy(), label='Reward Moving Average (100 episodes)')

    plt.legend()
    plt.grid(True)
    # plt.pause(0.001)  # pause a bit so that plots are updated
    if episode % 50 == 0:
        plot_name = f"plot_{episode+1}.png"
        plot_name_path = os.path.join(graph_directory, plot_name)
        plt.savefig(plot_name_path, format="png", dpi=300)

    
    plt.close()


# In[10]:


def model_save(model, optimiser, episode, model_directory):
    file_name = f"epoch_{episode + 1}.pth"
    path = os.path.join(model_directory, file_name)
    check_point = {'episode': episode,
                    'dqn': model.state_dict(),
                    'optimiser': optimiser.state_dict()}
    torch.save(check_point, path)
    print(f"Model saved to {path}")


# In[19]:


import wandb
import torch
import itertools

# Initialize wandb with project details
wandb.init(project="dqn_project-2-3-v0", entity="chaya-ff7-durham",mode="offline")

best_score = 3490
def main():
    count = 0
    max_step = 2000000  # Example, ensure you set max_step appropriately
    step_done = 0  # Example variable, ensure it's initialized correctly
    
    for episode in range(20000):
        if count > max_step:
            break
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        accumulated_reward_epi = 0
        
        print(f"Episode : {episode}")
        for i in itertools.count():
            action = chosen_action(state)
            next_state, reward, done, info = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
            replay_buf.push(state, action, next_state, reward, done)
            count += 1
            accumulated_reward_epi += reward
            optimiser_fn()
            state = next_state
            
            # Log the current step progress every 1000 steps
            if i % 1000 == 0:
                print(f"Episode running at step: {i}")
              
            if done:
                print(f"Count: {count}")
                episode_done.append(i + 1)
                episode_rewards.append(accumulated_reward_epi)
                print(f"Total accumulated reward: {accumulated_reward_epi}")
                print(f"Step_done: {step_done}")

                # Log episode metrics to wandb when the episode is done
                wandb.log({
                    "episode": episode,
                    "accumulated_reward_epi": accumulated_reward_epi,
                    "episode_length": i + 1,
                    "step_done": step_done,
                    "total_steps": count
                })

               
                global best_score
                if accumulated_reward_epi > best_score:
                    best_score = accumulated_reward_epi
                    model_save(dqn_policy, optimiser, episode, model_directory)  # Save the model
                    print(f"New best score achieved: {best_score}, model saved.")
                
                if (episode + 1) % 10000 == 0:
                    model_save(dqn_policy, optimiser, episode, model_directory)

                break

        env.reset()

    env.close()

# Call main


# In[ ]:





# In[16]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = wrap(custom_reward=False)
input_shape = env.observation_space.shape
print(input_shape)
n_action = env.action_space.n
dqn_policy = DQN(input_shape, n_action).to(device)
target_policy = DQN(input_shape, n_action).to(device)
target_policy.load_state_dict(dqn_policy.state_dict())

batch_size = 256
step_required = 10000
tau = 0.005
count = 0
gamma = 0.95
accumulated_reward_epi =0
episode_done = []
episode_rewards = []
loss_total = []
learning_rate = 1e-4
replay_buf = ReplayMemory(50000)
step_done =0
epsi_start = 0.9
epsi_end = 0.05
update_frequent=10000
decay_rate = 50000
max_step = 2000000
loss_fn = nn.SmoothL1Loss()
optimiser = torch.optim.Adam(dqn_policy.parameters(), lr=learning_rate, amsgrad=True)
model_directory= "dqn23_model/"
if not os.path.exists(model_directory):
    print("little shit does not exist please creat it smart qeuun")
    os.makedirs("dqn23_model/")

graph_directory = "dqn23_graph/"
if not os.path.exists(graph_directory):
    print("graph directory does not exist")
    os.makedirs("dqn23_graph/")
print(device)


# In[18]:


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print('near Complete')
    print(f"time spent: {end-start}")
    wandb.finish()
    if not isinstance(loss_total, torch.Tensor):
        loss_total = torch.tensor(loss_total)
    
    file_name = "loss_data.pth"
    path = os.path.join(model_directory, file_name)
    torch.save(loss_total, path)
    if not isinstance(episode_rewards, torch.Tensor):
        reward_save = torch.tensor(episode_rewards)
        file_name = "reward_data.pth"
        path = os.path.join(model_directory, file_name)
        torch.save(reward_save, path)
    else:
        file_name = "reward_data.pth"
        path = os.path.join(model_directory, file_name)
        torch.save(episode_reward, path)
    
    if not isinstance(episode_done, torch.Tensor):
        episode_data = torch.tensor(episode_done)
        file_name = "episode_data.pth"
        path = os.path.join(model_directory, file_name)
        torch.save(episode_data, path)
    else:
        file_name = "episode_data.pth"
        path = os.path.join(model_directory, file_name)
        torch.save(episode_done, path)

print('Complete')  # it should start to ezploit at 2mil step
      # Mark the end of the run

