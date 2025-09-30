import retro
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym.wrappers import GrayScaleObservation, ResizeObservation

from Discretizer import Discretizer
from RetroEnv import RetroEnv

combos = [
    ["UP"], ["DOWN"], ["LEFT"], ["RIGHT"],
    ["A"], ["B"], ["C"], ["D"],
    ["A","C"], ["B","D"], ["UP","A"], ["DOWN","B"]
]

# Function to create a fresh wrapped env
# def make_env():
#     env = retro.make('MortalKombatII-Genesis', players=2, use_restricted_actions=retro.Actions.ALL)
#     env = Discretizer(env, combos)   # now action_space becomes Discrete
#     return env
#
# # Wrap in DummyVecEnv
# env = DummyVecEnv([make_env])
#
# # Train PPO
# model1 = PPO('MlpPolicy', env, device='cpu', verbose=1, learning_rate=0.01)
# model1.learn(5000)
#
# # Train DQN (separate environment)
# vec_env2 = DummyVecEnv([make_env])
# model2 = DQN('MlpPolicy', vec_env2, device='cpu', verbose=1, learning_rate=0.01)
# model2.learn(5000)


env = RetroEnv('MortalKombatII-Genesis', players=2, use_restricted_actions=retro.Actions.ALL)
# env = retro.make('MortalKombatII-Genesis', players=2, use_restricted_actions=retro.Actions.ALL)
print("Action space:", env.action_space)
env = Discretizer(env, combos)
env = DummyVecEnv([lambda: env])

model1 = PPO('MlpPolicy', env, device='cpu', verbose=1, learning_rate=0.01)
model1.learn(5000)
model2 = DQN('MlpPolicy', env, device='cpu', verbose=1, learning_rate=0.01)
model2.learn(5000)

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
