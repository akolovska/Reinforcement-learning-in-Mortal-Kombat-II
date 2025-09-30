import retro
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG, PPO, DQN
from RetroEnv import RetroEnv


env = RetroEnv('MortalKombatII-Genesis', players=2, use_restricted_actions=retro.Actions.FILTERED)
env = DummyVecEnv([lambda: env])

model1 = PPO('MlpPolicy', env, device='cpu', verbose=1)
model1.learn(10000)
model2 = PPO('MlpPolicy', env, device='cpu', verbose=1)
model2.learn(10000)

obs = env.reset()

while True:
    action1, _ = model1.predict(obs)
    action2, _ = model2.predict(obs)
    action = np.concatenate([action1, action2])
    obs, reward, done, info = env.step(action)
    env.render()
    if done.any():
        obs = env.reset()

env.close()
