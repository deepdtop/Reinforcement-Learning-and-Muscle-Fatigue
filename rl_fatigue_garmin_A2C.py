#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Advantage Actor-Critic (A2C) applied to FatigueEnv with Continuous Actions

Environment & State:
- State vector: concatenation of current biomechanical parameters (Cadence, GCT, HR), 
  previous biomechanical parameters (pre_efficiency_specific), and current fatigue level.
- Action space: continuous vector in [-1, 1]^3 (one per biomechanical param)
- Actions are scaled and applied to update biomechanical parameters, 
  with fatigue adjustments based on increase/decrease/no change.

Key components:
- Actor network outputs mean actions (μ(s)) bounded by tanh to [-1,1].
- Action distribution: Normal(μ(s), σ^2) with fixed std σ=0.1.
- Critic network outputs state value V(s).

Action Sampling:
  a_t ~ π_θ(a_t|s_t) = Normal(μ(s_t), σ^2)
  log_prob = log π_θ(a_t|s_t)

Reward & Transition:
  r_t = environment reward from state transition after applying action a_t
  s_{t+1} = next state from env.step(a_t)

Advantage Estimation with Generalized Advantage Estimation (GAE):

For a trajectory of length T, given discount factor γ and GAE parameter λ:

  δ_t = r_t + γ V(s_{t+1}) * (1 - done_t) - V(s_t)

  A_t = δ_t + γ λ (1 - done_t) A_{t+1}, for t = T-1,...,0
  with A_T = 0

Returns (target for critic):

  R_t = A_t + V(s_t)

Loss Functions:

- Actor loss (policy gradient):

  L_actor = - E_{t} [log π_θ(a_t|s_t) * A_t]

- Critic loss (value function regression):

  L_critic = E_{t} [(V(s_t) - R_t)^2]

Optimization:

- Use separate Adam optimizers for actor and critic.
- Minimize L_actor and L_critic with respect to their parameters.

Training Loop Summary:

1. Reset environment and scaler-transform initial state.
2. For each step t in episode:
   - Sample action a_t from actor policy.
   - Execute a_t in env → obtain s_{t+1}, r_t, done.
   - Store (log_prob, V(s_t), r_t, done).
   - Update state = s_{t+1}.
3. After episode end, get bootstrap value V(s_{T+1}) from critic.
4. Compute advantages A_t and returns R_t via GAE.
5. Update actor and critic networks using collected batch.

Environment specifics:

- Biomechanical parameters constrained within physical plausible bounds.
- Fatigue level updated based on direction and magnitude of changes in biomechanical params.
- Reward encourages improving efficiency proxy: 
   
   efficiency = 10*cadence/200 - 2*GCT/400 - 8*HR/200 - 2*fatigue/100

- This reward guides agent to balance performance vs fatigue.

Data Normalization:

- States normalized using sklearn StandardScaler for stable training.

Visualization:

- Plot continuous actions per parameter over time.
- Plot fatigue and reward evolution.
- KDE plots of action distributions for analysis.

@author: Dan Terracina, PhD
"""
import argparse
import os
import pickle
import matplotlib.pyplot as plt
from fitparse import FitFile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import itertools
import datetime
import warnings
import glob
warnings.filterwarnings('ignore')
import seaborn as sns
#import gym
#from gym import spaces
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F  # for activation functions, losses, etc.
import torch.optim as optim

scale_eval = np.array([2.0, 2.0, 0.1, 0.3, 3])
def extract_fit_metrics():
    """
    Loads pre-processed biomechanical data (Cadence, GCT, HR) from a pickle file.
    Returns the relevant portion as a NumPy array for modeling.
    """
    downloads_path = os.path.expanduser("~/Downloads")
    full_path = '/Users/danterracina/Downloads/19043835544_ACTIVITY.fit'
    filename = 'data_all.pkl'
    with open(filename, 'rb') as f:  # 'rb' means read binary
        data = pickle.load(f)
    columns_to_get = ['Cadence', 'GCT', 'vertical_ratio', 'enhanced_speed', 'power']
    data_df = data[columns_to_get].values[8:,:]
    plt.figure()
    data.plot()
    return data_df, columns_to_get

def get_scaler(env, n_episodes=10):
    """
    Collects a sequence of states by taking random *continuous* actions in the environment.
    Fits a StandardScaler to normalize state input features.
    """
    states = []
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, done, action, info = env.step(action)
            states.append(state)
    if done:
        state = env.reset()
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class FatigueEnv:
    """
    Custom environment to simulate fatigue progression based on biomechanical parameters and actions.
    """
    def __init__(self, data, initial_fatigue=50):
        self.fatigue_history = data
        self.n_step, self.n_biomeca_params = self.fatigue_history.shape
        self.initial_fatigue = initial_fatigue
        self.cur_step = None
        self.max_penalize = None
        self.current_biomeca_levels = None
        self.pre_efficency_specific = None
        self.current_fatigue = None
        # actions can be increase speed, decrease speed, keep speed
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_biomeca_params,), dtype=np.float32)
        self.action_scale = scale_eval #['Cadence', 'GCT', 'vertical_ratio', 'enhanced_speed', 'power']
        self.state_dim = self.n_biomeca_params * 2 + 1 #biomecanical parameters + fatigue levels
        self.param_bounds = {
            0: (120, 220),  # Cadence
            1: (150, 400),  # GCT
            2: (6, 15),  # Vertical ratio
            3: (8, 20.0),  # Speed
            4: (300, 600.0),  # Power
        }

        self.reset()

    def reset(self):
        """
        Resets to the start of the episode.
        """
        self.cur_step = 0
        self.current_biomeca_levels = self.fatigue_history[self.cur_step]
        self.pre_efficency_specific = np.array([self.fatigue_history[self.cur_step][0],
                                                self.fatigue_history[self.cur_step][1],
                                                self.fatigue_history[self.cur_step][2],
                                                self.fatigue_history[self.cur_step][3],
                                                self.fatigue_history[self.cur_step][4]])
        self.current_fatigue = self.initial_fatigue
        return self._get_obs()

    def step(self, action):
        """
        Applies an action, updates biomechanical states and fatigue, returns new state and reward.
        """
        #assert self.action_space.contains(action)
        prev_val = self._get_val() # get value of current fatigue: efficiency = cadence / (heart_rate * contact_time) or -raw["heart_rate"] * 0.5 + raw["cadence"]* 1.0 -raw["contact_time"] * 0.3
        self.cur_step += 1
        self.current_biomeca_levels = self.fatigue_history[self.cur_step]
        self._move(action) #perform action (trade in RL trading)
        cur_val = self._get_val()
        reward = cur_val - prev_val
        done = self.cur_step == self.n_step - 1
        info = {'cur_val' : cur_val}
        return self._get_obs(), reward, done, action, info

    def _get_obs(self):
        """
        Returns the current state vector.
        """
        obs = np.empty(self.state_dim)
        obs[:self.n_biomeca_params] = self.current_biomeca_levels
        obs[self.n_biomeca_params:2*self.n_biomeca_params] = self.pre_efficency_specific
        obs[-1] = self.current_fatigue
        return obs

    def _get_val(self):
        """
        Efficiency proxy (reward): higher cadence, lower HR and GCT is better.
        """
        # for reward calculation #['Cadence', 'GCT', 'vertical_ratio', 'enhanced_speed', 'power']
        efficiency = self.pre_efficency_specific[0]/200 - self.pre_efficency_specific[1]/400 - self.pre_efficency_specific[2]/12 + self.pre_efficency_specific[3]/20\
                        + self.pre_efficency_specific[4]/450  \
                        - 0.8*self.current_fatigue/100 - 5*np.sqrt((self.current_biomeca_levels[2]/12 - self.pre_efficency_specific[2]/12)**2 +
                        (self.current_biomeca_levels[1]/400 - self.pre_efficency_specific[1]/400)**2 +
                        (self.current_biomeca_levels[0]/200 - self.pre_efficency_specific[0]/200)**2 +
                        (self.current_biomeca_levels[3]/20 - self.pre_efficency_specific[3]/20)**2 +
                        (self.current_biomeca_levels[4] / 450 - self.pre_efficency_specific[4] / 450) ** 2)# 'Cadence', 'GCT','HR'
        danger_penalty = 0
        for i in range(self.n_biomeca_params):
            min_val, max_val = self.param_bounds[i]
            val_ = self.pre_efficency_specific[i]
            if val_ <= min_val+0.1*min_val or val_ >= max_val+0.1*max_val:
                danger_penalty += 1  # or use smooth penalty like: danger_penalty += np.exp(abs(val - max_val))

        danger_penalty_fatigue = 0
        danger_penalty_inneficiency = 0
        if self.current_fatigue >= 90:
            danger_penalty_fatigue += np.abs(self.current_fatigue - 90)
        elif self.current_fatigue <= 30:
            danger_penalty_inneficiency += 0.5  # or use smooth penalty like: danger_penalty += np.exp(abs(val - max_val))

        return efficiency - danger_penalty - danger_penalty_fatigue - danger_penalty_inneficiency

    def _move(self, action):
        """
        Updates biomechanical levels and fatigue based on the selected action.
        Action is decoded into biomechanical effects.
        """
        if isinstance(action, tuple):
            action = action[0]
        delta = action

        for i in range(self.n_biomeca_params):
            old_value = self.pre_efficency_specific[i]
            new_value = old_value + delta[i]

            # Enforce physiological limits
            min_val, max_val = self.param_bounds[i]
            new_value = np.clip(new_value, min_val, max_val)

            # Fatigue update logic
            if new_value > old_value:
                self.pre_efficency_specific[i] = new_value
                self.current_fatigue += 0.02
            elif new_value < old_value:
                if self.current_fatigue > 10:
                    self.pre_efficency_specific[i] = new_value
                    self.current_fatigue -= 0.01
            else:
                if self.current_fatigue > 1:
                    self.current_fatigue -= 0.01

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable log std

    def forward(self, x):
        x = self.base(x)
        mean = self.mean_head(x)
        std = torch.exp(self.log_std)
        return mean, std  # unbounded mean, std

class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class A2CAgent:
    def __init__(self, state_dim, action_dim, scale_eval, actor_lr=1e-4, critic_lr=1e-3):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.action_scale = scale_eval

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_mean, std_action = self.actor(state)
        dist = torch.distributions.Normal(action_mean, std_action)
        raw_action = dist.rsample()
        squashed_action = torch.tanh(raw_action)
        scaled_action = squashed_action * torch.FloatTensor(self.action_scale)
        log_prob = dist.log_prob(raw_action) - torch.log(1 - squashed_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1).squeeze()
        value = self.critic(state)
        return scaled_action.squeeze().detach().numpy(), log_prob, value.squeeze()

    def compute_advantage(self, rewards, values, next_value, dones, gamma=0.99, lam=0.95):
        # Ensure rewards and dones are lists
        if not isinstance(rewards, list):
            rewards = [rewards]
        if not isinstance(dones, list):
            dones = [dones]
        # Convert everything to tensors
        rewards = [torch.tensor(r, dtype=torch.float32) for r in rewards]
        dones = [torch.tensor(d, dtype=torch.float32) for d in dones]
        values_tensor = torch.tensor([v.item() if isinstance(v, torch.Tensor) else float(v) for v in values], dtype=torch.float32)
        # Append next_value to the end for V(s_{t+1})
        if isinstance(next_value, (np.ndarray, float)):
            next_value = torch.tensor(next_value, dtype=torch.float32)
        values_tensor = torch.cat([values_tensor, next_value.view(1)])
        # GAE advantage estimation
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values_tensor[t + 1] * (1 - dones[t]) - values_tensor[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.stack(advantages)
        returns = advantages + values_tensor[:-1]  # R_t = A_t + V(s_t)
        return returns.detach(), advantages.detach()

    def train(self, log_probs, values, rewards, next_value, dones, scaler):
        returns, advantages = self.compute_advantage(rewards, values, next_value, dones)
        values_tensor = torch.cat([v.unsqueeze(0) for v in values])  # Shape: [T]
        returns = returns.view(-1)
        critic_loss = nn.MSELoss()(values_tensor, returns)
        log_probs_tensor = torch.stack(log_probs).view(-1)
        actor_loss = -(log_probs_tensor * advantages.detach().view(-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def save(self, path_prefix='a2c_model'):
        torch.save(self.actor.state_dict(), f'{path_prefix}_actor.pth')
        torch.save(self.critic.state_dict(), f'{path_prefix}_critic.pth')
        print(f"Models saved as {path_prefix}_actor.pth and {path_prefix}_critic.pth")

    def load(self, path_prefix='a2c_model'):
        self.actor.load_state_dict(torch.load(f'{path_prefix}_actor.pth'))
        self.critic.load_state_dict(torch.load(f'{path_prefix}_critic.pth'))
        self.actor.eval()
        self.critic.eval()
        print(f"Models loaded from {path_prefix}_actor.pth and {path_prefix}_critic.pth")


def play_one_episode(agent, env, is_train):
    state = env.reset()
    state = scaler.transform([state])
    done = False
    action_list = []
    fatigue_list = [env.current_fatigue]
    state_evol = []
    reward_val = 0
    # Storage for training
    log_probs = []
    values = []
    rewards = []
    dones = []

    while not done:
        # Get action, log_prob, value estimate
        action, log_prob, value = agent.act(state)
        # Apply action to environment
        next_state, reward, done, action_perf, info = env.step(action)
        next_state = scaler.transform([next_state])

        if is_train == 'train':
            # Store experience
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)

        state = next_state
        action_list.append(action_perf)
        fatigue_list.append(env.current_fatigue)
        state_evol.append(scaler.inverse_transform(next_state)[0, :])
        reward_val += reward

    # Final value estimate for bootstrapping
    next_value = agent.critic(torch.FloatTensor(next_state)).detach()

    if is_train == 'train':
        agent.train(log_probs, values, rewards, next_value, dones, scaler)

    return info['cur_val'], action_list, fatigue_list, reward_val, pd.DataFrame(state_evol).values


if __name__ == '__main__':
    models_folder =  'linear_rl_fatigue_model'
    rewards_folder = 'linear_rl_fatigue_rewards'
    num_episodes = 200
    batch_size = 64
    initial_fatigue = 5
    type_of = 'train'
    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)
    data, columns_data = extract_fit_metrics()
    n_timesteps, n_biomechanical = data.shape
    n_train = n_timesteps//2
    train_data = data[:n_train]
    test_data = data[n_train:]
    for col in range(test_data.shape[1]):
        for row in range(1, test_data.shape[0]):
            if np.isnan(test_data[row, col]):
                test_data[row, col] = test_data[row - 1, col]
    env = FatigueEnv(train_data, initial_fatigue)
    state_size = env.state_dim
    action_size = (env.action_space.shape[0])
    print(f'action size {action_size}')
    agent = A2CAgent(state_size, action_size, scale_eval)
    scaler = get_scaler(env)
    #print(train_data[:20])
    efficiency_evol = []
    all_actions = []
    all_fatigues = []
    all_reward = []
    if type_of == 'test':
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        env = FatigueEnv(test_data, initial_fatigue)
        agent.epsilon = 0.01
        agent.load(f'{models_folder}/linear.npz')

    # play episode
    for e in range(num_episodes):
        t0 = datetime.datetime.now()
        val, actions, fatigue, reward_l, state_evol = play_one_episode(agent, env, type_of)
        dt = datetime.datetime.now() - t0
        print(f"episode: {e+1}/{num_episodes}, episode and value: {val:.2f}, duration: {dt} ")
        efficiency_evol.append(val)
        all_actions.append(actions)
        all_fatigues.append(fatigue)
        all_reward.append(reward_l)
        if (e == num_episodes % num_episodes or e == int(num_episodes / 4) or e == int(
                2 * num_episodes / 4) or e == int(3 * num_episodes / 4) or e == num_episodes - 1):
            # Assuming all_actions[-1] shape = (timesteps, 3) continuous values for [cadence, GTC, HR]
            actions_n_ = np.array(all_actions[-1])
            scale = scale_eval
            actions_np = scale * actions_n_
            # Use columns_data first 3 columns as parameter names
            action_param_names = columns_data
            df_actions = pd.DataFrame(actions_np, columns=action_param_names)

            fig = plt.figure(figsize=(20, 10))

            # Top left: Actions taken over time (continuous values per parameter)
            ax1 = fig.add_subplot(2, 4, 1)
            for col in df_actions.columns:
                ax1.plot(df_actions[col], marker='o', label=col)
            ax1.set_title(f'Actions Taken Over Time (Continuous). Episode {e}')
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Action Value')
            ax1.legend()

            # Top right: Fatigue evolution
            ax2 = fig.add_subplot(2, 4, 2)
            ax2.plot(all_fatigues[-1], color='red')
            ax2.set_title('Fatigue Evolution Over Time')
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Fatigue Level')

            # Reward evolution
            ax4 = fig.add_subplot(2, 4, 3)
            ax4.plot(all_reward, color='green')
            ax4.set_title('Reward Evolution Over Time (total reward per episode)')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Reward Level')

            # State evolution
            ax5 = fig.add_subplot(2, 4, 4)#['Cadence', 'GCT', 'vertical_ratio', 'enhanced_speed', 'power']
            ax5.plot(state_evol[:, 5], color='orange', linestyle=':', label='Cadence pred')
            ax5.plot(state_evol[:, 0], color='orange', label='Cadence true')
            ax5.plot(state_evol[:, 6], color='green', linestyle=':', label='GT pred')
            ax5.plot(state_evol[:, 1], color='green', label='GT true')
            ax5.plot(state_evol[:, 7], color='blue', linestyle=':', label='vertical ratio')
            ax5.plot(state_evol[:, 2], color='blue', label='vertical ratio')
            ax5.plot(state_evol[:, 8], color='red', linestyle=':', label='enhanced speed')
            ax5.plot(state_evol[:, 3], color='red', label='enhanced speed')
            ax5.plot(state_evol[:, 9], color='black', linestyle=':', label='power')
            ax5.plot(state_evol[:, 4], color='black', label='power')
            ax5.set_title('State Evolution Over Time')
            ax5.set_xlabel('Timestep')
            ax5.set_ylabel('State Level')
            ax5.legend()

            # Bottom (full width): KDE distribution of continuous actions taken
            ax3 = fig.add_subplot(2, 1, 2)
            for col in df_actions.columns:
                ax3.hist(df_actions[col], bins=30, alpha=0.5, label=col, density=True)
            ax3.set_title('Distribution of Continuous Actions Taken (Last Episode)')
            ax3.set_xlabel('Action Value')
            ax3.set_ylabel('Counts (capped to 10 for visualization)')
            ax3.legend()
            plt.tight_layout()
            ax3.set_ylim(0, 10)

            plt.show()

    if type_of == 'train':
        agent.save(f'{models_folder}/linear.npz')
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

    np.save(f'{rewards_folder}/{type_of}.npy', efficiency_evol)
    with open(f'{rewards_folder}/all_actions.pkl', 'wb') as f:
        pickle.dump(all_actions, f)
    with open(f'{rewards_folder}/all_fatigues.pkl', 'wb') as f:
        pickle.dump(all_fatigues, f)


