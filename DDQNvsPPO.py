import gym.utils.seeding as seeding

from agents.DDQN import DDQNAgent
from agents.PPO import PPOAgent
from env.MK2env import MK2TwoAgentEnv

if not hasattr(seeding, 'hash_seed'):
    import hashlib

    def hash_seed(seed):
        h = hashlib.sha512(str(seed).encode('utf-8')).digest()
        return int.from_bytes(h[:4], 'big')

    seeding.hash_seed = hash_seed

import numpy as np
import torch
from collections import namedtuple
import cv2

GAME = "MortalKombatII-Genesis"

P1_HP_ADDR = 0x00B622
P2_HP_ADDR = 0x00B712

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

USE_PER = True  # set False for standard DDQN


def show_frame(frame, delay=1):
    """
    Show an RGB frame using OpenCV.
    frame: (H, W, 3) in RGB.
    """
    if frame is None:
        return
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("MK2 AI Battle", img)
    cv2.waitKey(delay)

def preprocess_obs(obs):
    """
    obs: (H, W, 3), uint8
    -> (N,) float32 flattened grayscale
    """
    gray = obs.mean(axis=2) / 255.0  # (H, W)
    return gray.astype(np.float32).ravel()  # (H*W,)

def run_evaluation_match(env, ddqn, ppo, render=True):
    """
    Runs a BEST OF 3 evaluation match.
    Returns winner: 'DDQN', 'PPO', or 'Tie'
    """
    ddqn_wins = 0
    ppo_wins = 0

    for round_idx in range(3):
        obs_ddqn, obs_ppo = env.reset()
        done = False

        print(f"\n=== Evaluation Round {round_idx + 1} ===")

        while not done:
            a_ddqn = ddqn.act(obs_ddqn, greedy=False)
            a_ppo, _, _ = ppo.act(obs_ppo, greedy=False)

            next_obs_ddqn, next_obs_ppo, r_ddqn, r_ppo, done, info = env.step(a_ddqn, a_ppo)
            obs_ddqn, obs_ppo = next_obs_ddqn, next_obs_ppo

        p1_hp, p2_hp = env._read_hp()
        print(f"End HP -> DDQN (P1): {p1_hp}, PPO (P2): {p2_hp}")
        if p1_hp > p2_hp:
            ddqn_wins += 1
            print("DDQN wins round!")
        elif p2_hp > p1_hp:
            ppo_wins += 1
            print("PPO wins round!")
        else:
            print("Tie round")

        if ddqn_wins == 2 or ppo_wins == 2:
            break

    if ddqn_wins > ppo_wins:
        return "DDQN"
    elif ppo_wins > ddqn_wins:
        return "PPO"
    else:
        return "Tie"


def main():
    env = MK2TwoAgentEnv(render=False)

    obs_ddqn, obs_ppo = env.reset()
    obs_dim = obs_ddqn.shape[0]
    n_actions = env.n_actions

    ddqn = DDQNAgent(obs_dim, n_actions, use_per=USE_PER)
    ppo = PPOAgent(obs_dim, n_actions)

    num_episodes = 1000
    max_steps = 2000

    print(f"Training DDQN (PER={USE_PER}) vs PPO for {num_episodes} episodes...")

    for ep in range(num_episodes):
        obs_ddqn, obs_ppo = env.reset()
        done = False
        ep_r_ddqn = 0.0
        ep_r_ppo = 0.0
        steps = 0

        while not done and steps < max_steps:
            steps += 1

            a_ddqn = ddqn.act(obs_ddqn, greedy=False)
            a_ppo, logp_ppo, v_ppo = ppo.act(obs_ppo, greedy=False)

            next_obs_ddqn, next_obs_ppo, r_ddqn, r_ppo, done, info = env.step(a_ddqn, a_ppo)

            ep_r_ddqn += r_ddqn
            ep_r_ppo += r_ppo

            ddqn.store(
                obs_ddqn,
                a_ddqn,
                r_ddqn,
                next_obs_ddqn,
                float(done),
            )

            ppo.store((obs_ppo, a_ppo, logp_ppo, r_ppo, v_ppo, float(done)))

            obs_ddqn, obs_ppo = next_obs_ddqn, next_obs_ppo

            ddqn.train_step()

        ppo.finish_trajectory_and_update()

        print(f"Episode {ep} | DDQN reward: {ep_r_ddqn:.1f} | PPO reward: {ep_r_ppo:.1f} | steps: {steps}")

    env.close()

    print("\n==============================")
    print("Training finished. Starting evaluation match...")
    print("==============================")

    eval_env = MK2TwoAgentEnv(render=True)
    winner = run_evaluation_match(eval_env, ddqn, ppo, render=True)
    print("\n=== FINAL RESULT ===")
    print("Winner of Best-of-3:", winner)

    eval_env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
