import retro
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from Discretizer2 import Discretizer

# ----------------------------
# Define Mortal Kombat button combos
# ----------------------------
combos = [
    ["UP"], ["DOWN"], ["LEFT"], ["RIGHT"],
    ["A"], ["B"], ["C"], ["D"],
    ["A","C"], ["B","D"], ["UP","A"], ["DOWN","B"]
]

# ----------------------------
# Environment factory for training
# ----------------------------
def make_env():
    env = retro.make('MortalKombatII-Genesis', players=2, use_restricted_actions=retro.Actions.ALL)
    env = Discretizer(env, combos)
    return env

# ----------------------------
# Train PPO agent
# ----------------------------
ppo_env = DummyVecEnv([make_env])
ppo_agent = PPO('CnnPolicy', ppo_env, device='cpu', verbose=1, learning_rate=0.01)
ppo_agent.learn(total_timesteps=5000)
ppo_env.close()

# ----------------------------
# Train DQN agent
# ----------------------------
dqn_env = DummyVecEnv([make_env])
dqn_agent = DQN('CnnPolicy', dqn_env, device='cpu', verbose=1, learning_rate=0.01)
dqn_agent.learn(total_timesteps=5000)
dqn_env.close()

# ----------------------------
# Competition environment (2 players)
# ----------------------------
comp_env = retro.make('MortalKombatII-Genesis', players=2, use_restricted_actions=retro.Actions.ALL)
comp_env = Discretizer(comp_env, combos)
obs = comp_env.reset()

# ----------------------------
# Play loop: PPO vs DQN
# ----------------------------
while True:
    # Predict discrete action indices
    action1_idx, _ = ppo_agent.predict(obs)
    action2_idx, _ = dqn_agent.predict(obs)

    # Convert indices to MultiBinary actions
    action1_array = comp_env._decode_discrete_action[action1_idx]
    action2_array = comp_env._decode_discrete_action[action2_idx]

    # Merge actions for 2-player step
    combined_action = np.concatenate([action1_array, action2_array])

    # Step the environment
    obs, reward, done, info = comp_env.step(combined_action)
    comp_env.render()

    if done:
        obs = comp_env.reset()

comp_env.close()
