import gymnasium as retro
import gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from RetroEnv import RetroEnv


# ---- Wrapper: Discrete combos for DQN ----
class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, combos):
        super().__init__(env)
        self.combos = combos
        self.action_space = gym.spaces.Discrete(len(combos))

    def action(self, action_idx):
        return self.combos[action_idx]


# Define limited combos for DQN agent
combos = [
    np.zeros(12, dtype=int),
    np.array([1,0,0,0,0,0,0,0,0,0,0,0]),
    np.array([0,1,0,0,0,0,0,0,0,0,0,0]),
    np.array([0,0,1,0,0,0,0,0,0,0,0,0]),
    np.array([0,0,0,1,0,0,0,0,0,0,0,0]),
    np.array([1,0,1,0,0,0,0,0,0,0,0,0]),
]


# ---- Train PPO ----
def train_ppo():
    env = RetroEnv('MortalKombatII-Genesis', players=2, use_restricted_actions=retro.Actions.FILTERED)
    env = DummyVecEnv([lambda: env])
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=10000)
    env.close()
    return model


# ---- Train DQN ----
def train_dqn():
    env = RetroEnv('MortalKombatII-Genesis', players=1, use_restricted_actions=retro.Actions.FILTERED)
    env = DiscreteActionWrapper(env, combos)
    env = DummyVecEnv([lambda: env])
    model = DQN("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=10000)
    env.close()
    return model


# ---- Main ----
ppo_agent = train_ppo()
dqn_agent = train_dqn()

# Competition env (only one emulator instance here!)
env = RetroEnv('MortalKombatII-Genesis', players=2, use_restricted_actions=retro.Actions.FILTERED)
obs = env.reset()

while True:
    action1, _ = ppo_agent.predict(obs)   # PPO controls Player 1
    action2_idx, _ = dqn_agent.predict(obs)  # DQN controls Player 2
    action2 = combos[action2_idx]

    combined_action = np.concatenate([action1, action2])

    obs, reward, done, info = env.step(combined_action)
    env.render()

    if done:
        obs = env.reset()

env.close()
