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
