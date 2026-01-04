import gym.utils.seeding as seeding
from agents.DQN import DQNAgent
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
import cv2
import os
import json
from datetime import datetime

GAME = "MortalKombatII-Genesis"

P1_HP_ADDR = 0x00B622
P2_HP_ADDR = 0x00B712

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint settings
CHECKPOINT_DIR = "checkpoints"
SAVE_CHECKPOINT_EVERY = 100  # Save every N episodes


def run_evaluation_match(env, dqn, ppo, render=True):
    """
    Runs a BEST OF 3 evaluation match.
    Returns winner: 'DQN', 'PPO', or 'Tie'
    """
    dqn_wins = 0
    ppo_wins = 0

    for round_idx in range(3):
        obs_dqn, obs_ppo = env.reset()
        done = False

        print(f"\n=== Evaluation Round {round_idx + 1} ===")

        while not done:
            # Use greedy=True for evaluation (no exploration)
            a_dqn = dqn.act(obs_dqn, greedy=True)
            a_ppo, _, _ = ppo.act(obs_ppo, greedy=True)

            next_obs_dqn, next_obs_ppo, r_dqn, r_ppo, done, info = env.step(a_dqn, a_ppo)
            obs_dqn, obs_ppo = next_obs_dqn, next_obs_ppo

        p1_hp, p2_hp = env._read_hp()
        print(f"End HP -> DQN (P1): {p1_hp}, PPO (P2): {p2_hp}")
        if p1_hp > p2_hp:
            dqn_wins += 1
            print("DQN wins round!")
        elif p2_hp > p1_hp:
            ppo_wins += 1
            print("PPO wins round!")
        else:
            print("Tie round")

        if dqn_wins == 2 or ppo_wins == 2:
            break

    if dqn_wins > ppo_wins:
        return "DQN"
    elif ppo_wins > dqn_wins:
        return "PPO"
    else:
        return "Tie"


def save_training_stats(stats, filepath):
    """Save training statistics to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Training stats saved to {filepath}")


def main():
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(CHECKPOINT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Training run directory: {run_dir}")
    print(f"Using device: {DEVICE}")

    env = MK2TwoAgentEnv(render=False)

    obs_dqn, obs_ppo = env.reset()
    obs_dim = obs_dqn.shape[0]
    n_actions = env.n_actions

    dqn = DQNAgent(obs_dim, n_actions)
    ppo = PPOAgent(obs_dim, n_actions)

    num_episodes = 1000
    max_steps = 2000

    # Training statistics
    training_stats = {
        'episodes': [],
        'dqn_rewards': [],
        'ppo_rewards': [],
        'steps': [],
        'timestamp': timestamp,
    }

    print("\n" + "=" * 70)
    print(f"Training DQN vs PPO for {num_episodes} episodes...")
    print(f"Checkpoints will be saved every {SAVE_CHECKPOINT_EVERY} episodes")
    print("=" * 70)

    for ep in range(num_episodes):
        obs_dqn, obs_ppo = env.reset()
        done = False
        ep_r_dqn = 0.0
        ep_r_ppo = 0.0
        steps = 0

        while not done and steps < max_steps:
            steps += 1

            # Use greedy=False for training (exploration enabled)
            a_dqn = dqn.act(obs_dqn, greedy=False)
            a_ppo, logp_ppo, v_ppo = ppo.act(obs_ppo, greedy=False)

            next_obs_dqn, next_obs_ppo, r_dqn, r_ppo, done, info = env.step(a_dqn, a_ppo)

            ep_r_dqn += r_dqn
            ep_r_ppo += r_ppo

            dqn.store(obs_dqn, a_dqn, r_dqn, next_obs_dqn, float(done))
            ppo.store((obs_ppo, a_ppo, logp_ppo, r_ppo, v_ppo, float(done)))

            obs_dqn, obs_ppo = next_obs_dqn, next_obs_ppo
            dqn.train_step()

        ppo.finish_trajectory_and_update()

        # Record statistics
        training_stats['episodes'].append(ep)
        training_stats['dqn_rewards'].append(float(ep_r_dqn))
        training_stats['ppo_rewards'].append(float(ep_r_ppo))
        training_stats['steps'].append(steps)

        print(f"Episode {ep:4d} | DQN: {ep_r_dqn:6.1f} | PPO: {ep_r_ppo:6.1f} | Steps: {steps:4d}")

        # Save checkpoints periodically
        if (ep + 1) % SAVE_CHECKPOINT_EVERY == 0:
            dqn_path = os.path.join(run_dir, f"dqn_ep{ep + 1}.pt")
            ppo_path = os.path.join(run_dir, f"ppo_ep{ep + 1}.pt")
            dqn.save_checkpoint(dqn_path)
            ppo.save_checkpoint(ppo_path)

            # Save training stats
            stats_path = os.path.join(run_dir, "training_stats.json")
            save_training_stats(training_stats, stats_path)

    # Save final checkpoints
    print("\n" + "=" * 70)
    print("Training finished! Saving final checkpoints...")
    dqn.save_checkpoint(os.path.join(run_dir, "dqn_final.pt"))
    ppo.save_checkpoint(os.path.join(run_dir, "ppo_final.pt"))
    save_training_stats(training_stats, os.path.join(run_dir, "training_stats.json"))

    env.close()

    print("\n" + "=" * 70)
    print("Starting 10 evaluation matches...")
    print("=" * 70)

    eval_env = MK2TwoAgentEnv(render=True)

    dqn_total = 0
    ppo_total = 0
    ties = 0

    num_matches = 10
    eval_results = []

    for i in range(num_matches):
        print(f"\n######## Evaluation Match {i + 1}/{num_matches} ########")
        winner = run_evaluation_match(eval_env, dqn, ppo, render=True)

        if winner == "DQN":
            dqn_total += 1
        elif winner == "PPO":
            ppo_total += 1
        else:
            ties += 1

        eval_results.append({'match': i + 1, 'winner': winner})
        print(f"Match {i + 1} winner: {winner}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS ACROSS 10 MATCHES")
    print("=" * 70)
    print(f"DQN wins: {dqn_total}")
    print(f"PPO wins: {ppo_total}")
    print(f"Ties    : {ties}")
    print("=" * 70)

    # Save evaluation results
    eval_stats = {
        'dqn_wins': dqn_total,
        'ppo_wins': ppo_total,
        'ties': ties,
        'matches': eval_results,
    }
    eval_path = os.path.join(run_dir, "evaluation_results.json")
    save_training_stats(eval_stats, eval_path)

    eval_env.close()
    cv2.destroyAllWindows()

    print(f"\nAll results saved to: {run_dir}")


if __name__ == "__main__":
    main()