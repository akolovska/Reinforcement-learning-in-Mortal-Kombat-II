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
            a_ddqn = ddqn.act(obs_ddqn, greedy=True)
            a_ppo, _, _ = ppo.act(obs_ppo, greedy=True)

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

    obs_ddqn, obs_ppo = env.reset()
    obs_dim = obs_ddqn.shape[0]
    n_actions = env.n_actions

    ddqn = DDQNAgent(obs_dim, n_actions)
    ppo = PPOAgent(obs_dim, n_actions)

    num_episodes = 1000
    max_steps = 2000

    # Training statistics
    training_stats = {
        'episodes': [],
        'ddqn_rewards': [],
        'ppo_rewards': [],
        'steps': [],
        'timestamp': timestamp,
    }

    print(f"\nTraining DDQN vs PPO for {num_episodes} episodes...")
    print(f"Checkpoints will be saved every {SAVE_CHECKPOINT_EVERY} episodes")
    print("=" * 70)

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

        # Record statistics
        training_stats['episodes'].append(ep)
        training_stats['ddqn_rewards'].append(float(ep_r_ddqn))
        training_stats['ppo_rewards'].append(float(ep_r_ppo))
        training_stats['steps'].append(steps)

        print(f"Episode {ep:4d} | DDQN: {ep_r_ddqn:6.1f} | PPO: {ep_r_ppo:6.1f} | Steps: {steps:4d}")

        # Save checkpoints periodically
        if (ep + 1) % SAVE_CHECKPOINT_EVERY == 0:
            ddqn_path = os.path.join(run_dir, f"ddqn_ep{ep + 1}.pt")
            ppo_path = os.path.join(run_dir, f"ppo_ep{ep + 1}.pt")
            ddqn.save_checkpoint(ddqn_path)
            ppo.save_checkpoint(ppo_path)

            # Save training stats
            stats_path = os.path.join(run_dir, "training_stats.json")
            save_training_stats(training_stats, stats_path)

    # Save final checkpoints
    print("\n" + "=" * 70)
    print("Training finished! Saving final checkpoints...")
    ddqn.save_checkpoint(os.path.join(run_dir, "ddqn_final.pt"))
    ppo.save_checkpoint(os.path.join(run_dir, "ppo_final.pt"))
    save_training_stats(training_stats, os.path.join(run_dir, "training_stats.json"))

    env.close()

    print("\n" + "=" * 70)
    print("Starting 10 evaluation matches...")
    print("=" * 70)

    eval_env = MK2TwoAgentEnv(render=True)

    ddqn_total = 0
    ppo_total = 0
    ties = 0

    num_matches = 10
    eval_results = []

    for i in range(num_matches):
        print(f"\n######## Evaluation Match {i + 1}/{num_matches} ########")
        winner = run_evaluation_match(eval_env, ddqn, ppo, render=True)

        if winner == "DDQN":
            ddqn_total += 1
        elif winner == "PPO":
            ppo_total += 1
        else:
            ties += 1

        eval_results.append({'match': i + 1, 'winner': winner})
        print(f"Match {i + 1} winner: {winner}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS ACROSS 10 MATCHES")
    print("=" * 70)
    print(f"DDQN wins: {ddqn_total}")
    print(f"PPO wins : {ppo_total}")
    print(f"Ties     : {ties}")
    print("=" * 70)

    # Save evaluation results
    eval_stats = {
        'ddqn_wins': ddqn_total,
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