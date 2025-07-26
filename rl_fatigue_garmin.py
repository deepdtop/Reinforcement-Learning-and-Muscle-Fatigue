#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:47:15 2025

This project trains a DQN agent to learn how to adjust biomechanical parameters (cadence, HR, GCT) during a run to minimize fatigue and maximize running efficiency.
The key components are:
1) Data Extraction – Load biomechanical data from a file.
2) Environment (FatigueEnv) – Simulate running session + fatigue response to actions.
3) Agent (DQN) – Learn Q-values (state → action) using a linear model. --> discrete action space
4) Training Loop – Run many episodes to learn optimal behavior.
5) Evaluation / Visualization – Plot fatigue and actions for inspection.

----------------------

how it works:

Deep Q-Learning Training Loop Explanation
-----------------------------------------

This script implements a basic Deep Q-Learning (DQN) algorithm using a simple linear model
as the Q-function approximator and SGD with momentum for parameter updates.

Main Steps in the Algorithm
---------------------------

1. Reset Environment:
   - At the beginning of each episode, the environment is reset to an initial state `s`.

2. Action Selection (Epsilon-Greedy):
   - With probability ε (epsilon), the agent selects a random action → encourages exploration.
   - With probability 1 - ε, the agent selects the action `a` that maximizes the predicted Q-value:
       Q(s, a) = model.predict(s)[a]
   - This balances exploration and exploitation.

3. Execute Action and Observe Outcome:
   - The chosen action `a` is executed in the environment.
   - The environment returns:
       - next_state `s'`
       - reward `r`
       - done flag (whether the episode has ended)

4. Compute Bellman Target:
   - The agent uses the next state `s'` to predict Q-values for all next actions: Q(s', a')
   - The Bellman target for the chosen action is:
       target = r + γ * max_a' Q(s', a')
   - If the episode is done, then:
       target = r

5. Train the Q-function Approximator:
   - Predict the current Q-values: `target_full = model.predict(s)`
   - Only update the Q-value of the action that was actually taken:
       target_full[0, a] = target
   - Use stochastic gradient descent (SGD) with momentum to minimize the MSE between:
       - predicted Q-values: model.predict(s)
       - target values: target_full
   - This trains the model to better approximate the expected future reward for each action.

6. Loop or End:
   - If the episode is not done, the agent continues from the new state `s'`.
   - If done, the episode ends and the agent resets the environment for the next run.

What the Agent Learns:
----------------------
- The model learns to approximate the Q-function Q(s, a):
    - The expected cumulative discounted reward of taking action `a` in state `s`
      and following the learned policy thereafter.
- The agent's policy improves over time as it favors actions with higher learned Q-values.

Model Details:
--------------
- The Q-function is approximated using a linear model: Q(s, a) = s @ W + b
- Training is done using gradient descent on the Mean Squared Error (MSE) between:
    - Predicted Q-values and Bellman targets
- Only the Q-value of the action that was actually taken is updated during training.
- Training uses a momentum term to accelerate convergence and smooth updates.

Epsilon Decay (Optional):
-------------------------
- In practice, epsilon is often decayed over time:
    - Start with ε = 1.0 (more exploration)
    - Gradually reduce ε to a smaller value (e.g., 0.1) as learning progresses


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
warnings.filterwarnings('ignore')
import gym
from gym import spaces


def extract_fit_metrics():
    """
    Loads pre-processed biomechanical data (Cadence, GCT, HR) from a pickle file.
    Returns the relevant portion as a NumPy array for modeling.
    """
    downloads_path = os.path.expanduser("~/Downloads")
    full_path = '/Users/danterracina/Downloads/19043835544_ACTIVITY.fit'
    filename = 'data.pkl'
    with open(filename, 'rb') as f:  # 'rb' means read binary
        data = pickle.load(f)
    columns_to_get = ['Cadence', 'GCT','HR']
    data_df = data[columns_to_get].values[8:,:]
    plt.figure()
    data.plot()
    return data_df, columns_to_get

def get_scaler(env):
    """
    Collects a sequence of states by taking random actions in the environment.
    Fits a StandardScaler to normalize state input features.
    """
    states = []
    for _ in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, action, info = env.step(action)
        states.append(state)
        if done:
            break
    scaler = StandardScaler()
    scaler.fit(states)  # maybe we can add multiple episodes to fit to state.
    return scaler

def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class LinearModel:
    """
    A simple linear model for Q-learning with SGD and momentum.
    Q(s, a) = state.dot(W) + b
    """
    def __init__(self, input_dim, n_action):
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)
        # momentum terms
        self.vW = 0
        self.vb = 0
        self.losses = []

    def predict(self, X):
        """
        Forward pass: computes Q-values.
        """
        # make sure X is N x D
        assert (len(X.shape) == 2)
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        """
        Performs one SGD update with momentum to minimize MSE loss.
        """
        # make sure X is N x D
        assert (len(X.shape) == 2)
        # the loss values are 2-D. normally we would divide by N only but now we divide by N x K
        num_values = np.prod(Y.shape)
        # do one step of gradient descent, we multiply by 2 to get the exact gradient
        Yhat = self.predict(X)
        # print(f'VEEEEC {Y-Yhat}')
        # print('yhat-y {}'.format(Yhat - Y))
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values
        # print('gW {}'.format(gW))
        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb
        # print('vW {}'.format(self.vW))
        # update params
        self.W += self.vW
        self.b += self.vb
        mse = np.mean((Yhat - Y) ** 2)
        self.losses.append(mse)

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)

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
        self.action_list_ = list(map(list, itertools.product([0,1,2], repeat=self.n_biomeca_params))) #but some actions are not compatible: i.e decrease in HR while increasing any of the cadence or step legnth or reducing step length and reducing cadence or viceversa
        self.excluded_actions = [
            [2, 2, 0],
            [2, 0, 1],
            [2, 0, 0],
            [0, 2, 1],
            [0, 2, 2],
            [0, 0, 0],
            [0, 0, 2],
            [0, 0, 1],
            [1, 0, 0],
            [1, 2, 2],
            [2, 2, 2],
            [2, 2, 1],
            [2, 1, 0],
            [0, 1, 2]
        ]
        self.action_list = [a for a in self.action_list_ if a not in self.excluded_actions]
        self.action_space = np.arange(len(self.action_list))
        print('self.action_list  {} and action space {}'.format(self.action_list, self.action_space))
        self.state_dim = self.n_biomeca_params * 2 + 1 #biomecanical parameters + fatigue levels
        self.reset()

    def reset(self):
        """
        Resets to the start of the episode.
        """
        self.cur_step = 0
        self.current_biomeca_levels = self.fatigue_history[self.cur_step]
        self.pre_efficency_specific = np.array([self.fatigue_history[self.cur_step][0], self.fatigue_history[self.cur_step][1], self.fatigue_history[self.cur_step][2]])
        self.current_fatigue = self.initial_fatigue
        return self._get_obs()

    def step(self, action):
        """
        Applies an action, updates biomechanical states and fatigue, returns new state and reward.
        """
        #print(f'action {action}')
        assert action in self.action_space
        prev_val = self._get_val() # get value of current fatigue: efficiency = cadence / (heart_rate * contact_time) or -raw["heart_rate"] * 0.5 + raw["cadence"]* 1.0 -raw["contact_time"] * 0.3
        self.cur_step += 1
        #print(f'CURRENT STEP IS {self.cur_step}')
        self.current_biomeca_levels = self.fatigue_history[self.cur_step]
        #print(f'current_biomeca_levels BEFORE MOVE {self.current_biomeca_levels}')
        self._move(action) #perform action (trade in RL trading)
        cur_val = self._get_val()
        #print(f'cur_val reward is {cur_val} and {prev_val}')
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
        #print(f'obs: {obs}')
        return obs

    def _get_val(self):
        """
        Efficiency proxy (reward): higher cadence, lower HR and GCT is better.
        """
        # for reward calculation
        #print(f'current_biomeca_levels in get val: {self.current_biomeca_levels}')
        efficiency = 10*self.pre_efficency_specific[0]/200 - 2*self.pre_efficency_specific[1]/400 - 8*self.pre_efficency_specific[2]/200 \
                        - 2*self.current_fatigue/100   #\
                        #- np.sqrt((self.current_biomeca_levels[2]/200 - self.pre_efficency_specific[2]/200)**2 +
                        #          (self.current_biomeca_levels[1]/400 - self.pre_efficency_specific[1]/400)**2 +
                        #          (self.current_biomeca_levels[0]/200 - self.pre_efficency_specific[0]/200)**2)# 'Cadence', 'GCT','HR'
        return efficiency

    def _move(self, action):
        """
        Updates biomechanical levels and fatigue based on the selected action.
        Action is decoded into biomechanical effects.
        """
        action_vec = self.action_list[action]
        reduce_biomecanical_parameter = []
        increase_biomecanical_parameter = []
        keep_biomecanical_parameter = []
        self.max_penalize = None
        for i, a in enumerate(action_vec):
            if a == 0: # 0 : reduce biomeca parameter; 1: keep biomeca, 2 = increase biomeca parameter
                reduce_biomecanical_parameter.append(i)
            elif a == 1:
                keep_biomecanical_parameter.append(i)
            elif a == 2:
                increase_biomecanical_parameter.append(i)
        if reduce_biomecanical_parameter:
            can_decrease_biomecanical_efficiency = True
            for i in reduce_biomecanical_parameter: # 0: Cadence 1: GCT 2: HR
                if self.current_fatigue > 10 and can_decrease_biomecanical_efficiency and self.pre_efficency_specific[0] >49 and self.pre_efficency_specific[0] <200 and self.pre_efficency_specific[1] >199 and self.pre_efficency_specific[1] <400  and self.pre_efficency_specific[2] >69 and self.pre_efficency_specific[2] < 200:
                    if i == 0:
                        self.pre_efficency_specific[i] = self.pre_efficency_specific[i] - 0.5
                    elif i == 1:
                        self.pre_efficency_specific[i] = self.pre_efficency_specific[i] + 0.5
                    elif i == 2:
                        self.pre_efficency_specific[i] = self.pre_efficency_specific[i] - 0.5
                    self.current_fatigue = self.current_fatigue - 0.2
                else:
                    can_increase_biomecanical_efficiency = True

        if increase_biomecanical_parameter:
            can_increase_biomecanical_efficiency = True
            for i in increase_biomecanical_parameter:
                if self.current_fatigue < 99 and can_increase_biomecanical_efficiency and self.pre_efficency_specific[0] >50 and self.pre_efficency_specific[0] <201 and self.pre_efficency_specific[1] >200 and self.pre_efficency_specific[1] <401 and self.pre_efficency_specific[2] >70 and self.pre_efficency_specific[2] < 201:
                    if i == 0:
                        self.pre_efficency_specific[i] = self.pre_efficency_specific[i] + 0.5
                    elif i == 1:
                        self.pre_efficency_specific[i] = self.pre_efficency_specific[i] - 0.5
                    elif i == 2:
                        self.pre_efficency_specific[i] = self.pre_efficency_specific[i] + 0.85
                    self.current_fatigue = self.current_fatigue + 0.3 #increasing power and speed, increases faster fatigue than it reduces, therefore reducing it its parameter is smaller
                else:
                    can_increase_biomecanical_efficiency = True # if too tired, don't push, but rather reduce
                    can_decrease_biomecanical_efficiency = True

        if keep_biomecanical_parameter:
            can_keep_biomecanical_efficiency = True
            for i in keep_biomecanical_parameter:
                if self.current_fatigue > 1 and can_keep_biomecanical_efficiency and self.pre_efficency_specific[0] >49 and self.pre_efficency_specific[0] <201 and self.pre_efficency_specific[1] >199 and self.pre_efficency_specific[1] <401 and self.pre_efficency_specific[2] >69 and self.pre_efficency_specific[2] < 201:
                    self.current_fatigue = self.current_fatigue - 0.1
                    self.pre_efficency_specific[i] = self.pre_efficency_specific[i] # we could actually individualize how much we are reducing each. currently reducing 2 Universal units of each
                else:
                    can_keep_biomecanical_efficiency = True

        #print(f'action_vec is {action_vec} and biomeca levels caluclated after action -> {self.current_biomeca_levels} preeficiency {self.pre_efficency_specific}')

class DQNAgent(object):
    """
    Deep Q-Learning Agent using a LinearModel as function approximator.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        print(f'state action {self.state_size} and {self.action_size}')
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = LinearModel(state_size, action_size)

    def act(self, state):
        """
        Chooses action using epsilon-greedy policy.
        """
        randomality = np.random.rand()
        if randomality <= self.epsilon:
            action_taken = np.random.choice(self.action_size)
            #print(f' aaaaa {self.action_size}!!')
            return action_taken
        act_values = self.model.predict(state)
        # print(f'not random {randomality, self.epsilon}')
        return np.argmax(act_values[0])  # returns action

    def train(self, state, action, reward, next_state, done,scaler_):
        """
        Trains the model using the Bellman equation.
        """
        if done:
            target = reward
        else:
            #print(f'self.model.predict(next_state) {self.model.predict(next_state)}')
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1) #DQN : Q(s,a)←r+ gamma* max_a′ Q(s ′,a′) where a' is ANY (AND HERE IS THE KEY) ACTION FUTURE.
            #target = reward + self.gamma * (self.model.predict(next_state)[0][action]) #SARSA
        # print("target is {}".format(target))
        # print("np.amax is {}".format(np.amax(self.model.predict(next_state), axis=1)))
        target_full = self.model.predict(state)
        target_full[0, action] = target
        #print(f'target_full {target_full} and target: {target} action {action}')
        #target_full_t =[target_full[0][0],target_full[0][1],target_full[0][2],0]
        #target_full__ = scaler_.inverse_transform(np.array(target_full_t).reshape(1, -1))
        #print("target full is {}".format(target_full__))
        # Run one training step
        self.model.sgd(state, target_full)
        if self.epsilon > self.epsilon_min and done:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def play_one_episode(agent, env, is_train):
    """
    Plays one full episode, collecting actions and fatigue history.
    Optionally trains the agent during the episode.
    """
    state = env.reset()
    #print(f'examp state {state}')
    state = scaler.transform([state])
    done = False
    action_list = []
    fatigue_list = [env.current_fatigue]
    state_evol = []
    reward_val = 0
    while not done:
        action = agent.act(state)
        next_state, reward, done, action_perf, info = env.step(action)
        next_state = scaler.transform([next_state])
        if is_train == 'train':
            #print(f'state is {scaler.inverse_transform(state)} and next state is {scaler.inverse_transform(next_state)}')
            agent.train(state, action, reward, next_state, done, scaler)

        state = next_state
        action_list.append(action_perf)
        fatigue_list.append(env.current_fatigue)
        state_evol.append(scaler.inverse_transform(next_state)[0,:])
        reward_val += reward
        try:
            if len(set(fatigue_list[-16:])) == 1:
                #print(f' fatigue list {fatigue_list[-16:]} actipns: {action_list[-16:]} state : {state_evol[-16:]}')
                debugger_on = True
        except:
            continue
    return info['cur_val'], action_list, fatigue_list, reward_val, pd.DataFrame(state_evol).values

if __name__ == '__main__':
    models_folder =  'linear_rl_fatigue_model'
    rewards_folder = 'linear_rl_fatigue_rewards'
    num_episodes = 200
    batch_size = 32
    initial_fatigue = 5
    type_of = 'train'
    maybe_make_dir(models_folder)
    maybe_make_dir(rewards_folder)
    data, columns_data = extract_fit_metrics()
    n_timesteps, n_biomechanical = data.shape
    n_train = n_timesteps//2
    train_data = data[:n_train]
    test_data = data[n_train:]
    env = FatigueEnv(train_data, initial_fatigue)
    state_size = env.state_dim
    action_size = len(env.action_space)
    print(f'action size {action_size}')
    agent = DQNAgent(state_size, action_size)
    scaler = get_scaler(env)
    print(train_data[:20])
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
        if (e == num_episodes%num_episodes or e == int(num_episodes/4) or e == int(2*num_episodes/4) or e == int(3*num_episodes/4) or  e == num_episodes-1) :
            action_map = {0: "RED", 1: "KEEP", 2: "INC"}
            # Generate all possible action triplets
            action_tuples_ = list(itertools.product([0, 1, 2], repeat=3))
            excluded_actions_ = [
                (2, 2, 0),
                (2, 0, 1),
                (2, 0, 0),
                (0, 2, 1),
                (0, 2, 2),
                (0, 0, 0),
                (0, 0, 2),
                (0, 0, 1),
                (1, 0, 0),
                (1, 2, 2),
                (2, 2, 2),
                (2, 2, 1),
                (2, 1, 0),
                (0, 1, 2)
            ]
            action_tuples = [a for a in action_tuples_ if a not in excluded_actions_]
            print(action_tuples)
            action_labels = ['-'.join(f"{action_map[a]} {columns_data[idx]}" for idx, a in enumerate(t)) for t in action_tuples]
            action_labels_bis = ['\n'.join(f"{action_map[a]} {columns_data[idx]}" for idx, a in enumerate(t)) for t in action_tuples]

            x = list(range(len(action_labels)))
            x_bis = list(range(len(action_labels_bis)))
            fig = plt.figure(figsize=(20, 10))

            # Top left: Actions taken over time
            ax1 = fig.add_subplot(2, 4, 1)  # 2 rows, 2 cols, pos 1
            ax1.plot(all_actions[-1], marker='o')
            ax1.set_yticks(x)
            ax1.set_yticklabels(action_labels, rotation=0, fontsize=8)
            ax1.set_title(f'Actions Taken Over Time. Episode {e}')
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Action (0=decrease, 1=keep, 2=increase)')

            # Top right: Fatigue evolution
            ax2 = fig.add_subplot(2, 4, 2)  # 2 rows, 2 cols, pos 2
            ax2.plot(all_fatigues[-1], color='red')
            ax2.set_title('Fatigue Evolution Over Time')
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Fatigue Level')

            ax4 = fig.add_subplot(2, 4, 3)  # 2 rows, 2 cols, pos 2
            ax4.plot(all_reward, color='green')
            ax4.set_title('Reward Evolution Over Time (total reward per episode)')
            ax4.set_xlabel('Timestep')
            ax4.set_ylabel('Reward Level')

            ax5 = fig.add_subplot(2, 4, 4)  # 2 rows, 2 cols, pos 2
            ax5.plot(state_evol[:,3], color='orange',linestyle=':', label='Cadence pred')
            ax5.plot(state_evol[:,0], color='orange',  label='Cadence true')
            ax5.plot(state_evol[:,4], color='green',  linestyle=':', label='GT pred')
            ax5.plot(state_evol[:,1], color='green', label='GT true')
            ax5.plot(state_evol[:,5], color='blue',linestyle=':',  label='HR pred')
            ax5.plot(state_evol[:,2], color='blue',  label='HR true')
            ax5.set_title('State Evolution Over Time (total reward per episode)')
            ax5.set_xlabel('Timestep')
            ax5.set_ylabel('State Level')
            ax5.legend()
            # Bottom (full width): Histogram of actions taken
            ax3 = fig.add_subplot(2, 1, 2)  # 2 rows, 1 col, pos 2 (full width bottom)
            ax3.hist(all_actions[-1], bins=len(action_tuples), rwidth=0.8)
            ax3.set_xticks(x_bis)
            ax3.set_xticklabels(action_labels_bis, rotation=70, fontsize=8)
            ax3.set_title('Actions Taken Distribution')
            ax3.set_xlabel('Action Combination')
            ax3.set_ylabel('Frequency')

            plt.tight_layout()
            plt.show()

    if type_of == 'train':
        agent.save(f'{models_folder}/linear.npz')
        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        plt.plot(agent.model.losses)
        plt.show()
    np.save(f'{rewards_folder}/{type_of}.npy', efficiency_evol)
    with open(f'{rewards_folder}/all_actions.pkl', 'wb') as f:
        pickle.dump(all_actions, f)
    with open(f'{rewards_folder}/all_fatigues.pkl', 'wb') as f:
        pickle.dump(all_fatigues, f)


