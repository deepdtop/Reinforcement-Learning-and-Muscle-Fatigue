# Reinforcement-Learning-and-Muscle-Fatigue
Reinforcement learning (classic and Deep RL algorithms) for muscle fatigue actions
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
