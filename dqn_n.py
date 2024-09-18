"""
I have added the commented code version and edit out personal wanbd key out. Therefore, to log the data please enter wanbd key
and edit the entity to your username in this line ------> wandb.init(project="dqn_project-2-3-v0", entity="---user name-----------",mode="offline")
for example entity="somethingusername"
if u have further information, want a cleaner code with class or found error in the code please email f.natchaya@gmail.com
"""


import gym   #rendering the environment 0.21.0 for compatibitlity with stablebaseline
from gym.vector import SyncVectorEnv, AsyncVectorEnv # for parallel environment 
import wandb #saving information and plot the graph
import gym_super_mario_bros 
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT  #simple movement, right only, and the complex movement setting
from gym.wrappers import FrameStack   
import matplotlib.pyplot as plt  #plot 
import numpy as np
import torch
from collections import namedtuple, deque   # for experience replay
from torch.distributions import Categorical  
import random
from torchvision import transforms as T
from torch import nn
from torch.nn.parallel import DataParallel  #sharing gpu if u have more then 1 gpu
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import functional as F
import math
import os
import itertools  #time count
import time
import cv2   # greyscale and frame modification
import wandb


from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
) #required gym 0.21.0 which should be compatible with our current mario



class normalisation(gym.ObservationWrapper):    #normalising the pixel from 0-255 to 0-1
    def __init__(self, env):
        super(normalisation, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=self.observation_space.shape, dtype=np.float32   # change the observation space when call
        )

    def observation(self, obs):
        return obs.__array__()/ 255.0  # when using framestack it will produce lazyframe output which causing a lot of problem in the network, so we need to get the frames in the numpy array form
    
    
    

def wrap(custom_reward=False):
    env = gym_super_mario_bros.make('SuperMarioBros-2-3-v0') # you change change the world [1-8], stage [1-4] and version of just Ram one dimension with v3 
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = MaxAndSkipEnv(env, 4) #process 4 frames and pick the one with most pixel to aviod the flicker in the old atari game
    env = gym.wrappers.ResizeObservation(env,84) # resize it to 84x84
    env = gym.wrappers.GrayScaleObservation(env) # change from 3 channels to 1 greyscale
    env = FrameStack(env, 4)  #stacked 4 frames up when feeding to neural network
    env = normalisation(env)  #normalise the image pixel to 0-1, with value approaching 1 means white

    if custom_reward:
        env = flagReward(env)  # adding the incentive to get the flag with +15 and penalise it when it does not with -15
        
    return gym.wrappers.RecordEpisodeStatistics(env)  #get the episode stat whether the duration and reward


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))  #transition containing all the state and others

class ReplayMemory:  

    def __init__(self, max_memory):
        self.memory = deque(maxlen=max_memory) #set the maximum memory it can save

    def push(self, state, action, next_state, reward, done):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)

   
        if action.dim() == 1:
            action = action.unsqueeze(1) #make sure the action has the shape [1, 1]
    
        reward = torch.tensor([reward], dtype=torch.float32) 
        done = torch.tensor([done], dtype=torch.float32)    

        self.memory.append(Transition(state, action, next_state, reward, done))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  #sample the batch of memory

    def __len__(self):
        return len(self.memory) #check the len of the memory to see if it has saved enough before it can generate the batch of memory


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

def get_plot(episode, graph_directory, show_result=False):
    if show_result:
        plt.figure()
        plt.title('Result')
        plt.xlabel('Episode')
        plt.ylabel('Duration/Reward')
        
        accu_reward =torch.tensor(episode_rewards, dtype=torch.float)
        durations_t = torch.tensor(episode_done, dtype=torch.float)
        episodes = torch.arange(len(episode_done), dtype=torch.float)
        plt.plot(episodes.numpy(), accu_reward.numpy(), label='Rewards')
        plt.plot(episodes.numpy(), durations_t.numpy(), label='Duration') #plot episode length as mario learn the length should decrease


        if len(durations_t) >= 100: # plot a moving average of durations if there are enough data 
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            reward_means = accu_reward.unfold(0, 100, 1).mean(1).view(-1)
            reward_means = torch.cat((torch.zeros(99), reward_means))

            plt.plot(episodes.numpy(), means.numpy(), label='Moving Average (100 episodes)')
            plt.plot(episodes.numpy(), reward_means.numpy(), label='Reward Moving Average (100 episodes)')

        plt.legend()
        plt.grid(True)
        if episode % 50 == 0:
            plot_name = f"plot_{episode+1}.png"
            plot_name_path = os.path.join(graph_directory, plot_name)
            plt.savefig(plot_name_path, format="png", dpi=300)


        plt.close()


def model_save(model, optimiser, episode, model_directory):
    file_name = f"epoch_{episode + 1}.pth"
    path = os.path.join(model_directory, file_name)
    check_point = {'episode': episode,
                    'dqn': model.state_dict(),
                    'optimiser': optimiser.state_dict()}
    torch.save(check_point, path)
    print(f"Model saved to {path}")


#init wandb to transfer the data to plot the graph
wandb.init(project="dqn_project-2-3-v0", entity="---user name-----------",mode="offline") #this can be set online or offline depends on the internet connection/ offline can be sncy later

best_score = 3490  #score citeria for saving the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device employed: {device}")

#init the environment
env = wrap(custom_reward=False)
input_shape = env.observation_space.shape
n_action = env.action_space.n
print(f"input shape: {input_shape}, number of action: {n_action}")

#init the online, target network and replay buffer
dqn_policy = DQN(input_shape, n_action).to(device)
target_policy = DQN(input_shape, n_action).to(device)
target_policy.load_state_dict(dqn_policy.state_dict())
replay_buf = ReplayMemory(50000)

#Hyperparameters
batch_size = 256   #number of samples in each batch
step_required = 10000  #minimum number of step taken by the agent before undating the network
tau = 0.005        #this can be used to penalise the update of target network if u want/ but it wont be use in the current example
gamma = 0.95       #discount factor 
epsi_start = 0.9   # initial epsilon value
epsi_end = 0.05    #epsilon value at the end
decay_rate = 50000  #decay rate
update_frequent=10000  # when to update the target network
learning_rate = 1e-4   #this can be change to 1e-3 or 1e-5 or even learning rate schedule

accumulated_reward_epi =0  #manually accumulated reward, the same as gymstat
episode_done = []          #list to keep track of episode duration # only needed if u generate plot by matplotlib
episode_rewards = []       #list of rewards in each episode
loss_total = []            

#Setting optimiser and loss function
loss_fn = nn.SmoothL1Loss()
optimiser = torch.optim.Adam(dqn_policy.parameters(), lr=learning_rate, amsgrad=True)

#setting the directory to save the checkpoints and graph from matplot
model_directory= "dqn23_model/"
if not os.path.exists(model_directory):
    print("little shit does not exist please creat it smart qeuun")
    os.makedirs("dqn23_model/")

graph_directory = "dqn23_graph/"
if not os.path.exists(graph_directory):
    print("graph directory does not exist")
    os.makedirs("dqn23_graph/")

def main():
    count = 0
    max_step = 2000000  # maximum step the run can take 
    step_done = 0  # init step 
    
    for episode in range(20000):
        if count > max_step: # if the step taken by Mario is over 2mil, ti will stop the game
            break
        state = env.reset()  #reset env to starting point, the state is in numpy array
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) #change the np.array to torch and add batch index from (4, 84, 84) to (1, 4, 84, 84)
        accumulated_reward_epi = 0 #init reward
        
        print(f"Episode : {episode}")
        for i in itertools.count(): #start counting time
            action = chosen_action(state) #select action from epsilon greedy approach
            next_state, reward, done, info = env.step(action.item()) #take a step from the action chosen randomly or by network
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device) #change to torch as network can only processs torch
            replay_buf.push(state, action, next_state, reward, done) #remember the transition
            count += 1                       #count this interaction as one
            accumulated_reward_epi += reward #add reward from each step given by the environment
            optimiser_fn()                   #calculate the loss and update the network
            state = next_state              
            

            if done: #if the episode is done
                print(f"Count: {count}") 
                episode_done.append(i + 1)  #add the epsiode duration into the list
                episode_rewards.append(accumulated_reward_epi) #add the total episode reward in the list
                print(f"Total accumulated reward: {accumulated_reward_epi}")
                print(f"Step_done: {step_done}")

                wandb.log({  #log data for graph to wanbd
                    "episode": episode,
                    "accumulated_reward_epi": accumulated_reward_epi,
                    "episode_length": i + 1,
                    "step_done": step_done,
                    "total_steps": count
                })

               
                global best_score
                if accumulated_reward_epi > best_score: #if the episode reward received is greater than the best score
                    best_score = accumulated_reward_epi #set the new best score
                    model_save(dqn_policy, optimiser, episode, model_directory)  # Save the model with currently highest reward
                    print(f"New best score achieved: {best_score}, model saved.")
                
                if (episode + 1) % 10000 == 0:  # this can be use if u want to save model every 100, 1000, 10000 episode
                    model_save(dqn_policy, optimiser, episode, model_directory)

                break 

        env.reset()

    env.close() #close the environment




if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print('near Complete')
    print(f"time spent: {end-start}")
    wandb.finish() #close the wanbd
    
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

print('Complete') 
     

