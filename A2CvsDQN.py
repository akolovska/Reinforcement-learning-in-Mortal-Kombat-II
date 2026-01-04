import gym.utils.seeding as seeding
import os
import json
from datetime import datetime
import cv2
import torch

from agents.DQN import DQNAgent
from agents.A2C import A2CAgent
from env.MK2env import MK2TwoAgentEnv

# --- Gym Retro seeding fix ---
if not hasattr(seeding, "hash_seed"):
    import hashlib

    def hash_seed(seed):
        h = hashlib.sha512(str(seed).encode("utf-8")).digest()
        return int.from_bytes(h[:4], "big")

    seeding.hash_seed = hash_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint settings (same idea as your PPOvsDQN)
CHECKPOINT_DIR = "checkpoints"
SAVE_CHECKPOINT_EVERY = 100


def save_json(obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved: {filepath}")


# ===============================
# Evaluation (Best-of-3)
# ===============================
def run_eval_match(env, dqn, a2c):
    dqn_wins, a2c_wins = 0, 0

    for round_idx in range(3):
        obs_dqn, obs_a2c = env.reset()
        done = False

        print(f"\n=== Evaluation Round {round_idx + 1} ===")

        while not done:
            # deterministic evaluation
            a_dqn = dqn.act(obs_dqn, greedy=True)
            a_a2c, _, _, _ = a2c.act(obs_a2c, greedy=True)

            next_obs_dqn, next_obs_a2c, _, _, done, _ = env.step(a_dqn, a_a2c)
            obs_dqn, obs_a2c = next_obs_dqn, next_obs_a2c

        p1_hp, p2_hp = env._read_hp()
        print(f"End HP -> DQN (P1): {p1_hp}, A2C (P2): {p2_hp}")

        if p1_hp > p2_hp:
            dqn_wins += 1
            print("DQN wins round!")
        elif p2_hp > p1_hp:
            a2c_wins += 1
            print("A2C wins round!")
        else:
            print("Tie round")

        if dqn_wins == 2 or a2c_wins == 2:
            break

    if dqn_wins > a2c_wins:
        return "DQN"
    elif a2c_wins > dqn_wins:
        return "A2C"
    return "Tie"


# ===============================
# Main
# ===============================
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(CHECKPOINT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Training run directory: {run_dir}")
    print(f"Using device: {DEVICE}")

    env = MK2TwoAgentEnv(render=False)

    obs_dqn, obs_a2c = env.reset()
    obs_dim = obs_dqn.shape[0]
    n_actions = env.n_actions

    dqn = DQNAgent(obs_dim, n_actions)
    a2c = A2CAgent(obs_dim, n_actions)

    num_episodes = 1000
    max_steps = 2000

    training_stats = {
        "episodes": [],
        "dqn_rewards": [],
        "a2c_rewards": [],
        "steps": [],
        "timestamp": timestamp,
    }

    print("\n" + "=" * 70)
    print(f"Training A2C vs DQN for {num_episodes} episodes.")
    print(f"Checkpoints will be saved every {SAVE_CHECKPOINT_EVERY} episodes")
    print("=" * 70)

    for ep in range(num_episodes):
        obs_dqn, obs_a2c = env.reset()
        done = False
        steps = 0
        ep_r_dqn = 0.0
        ep_r_a2c = 0.0

        while not done and steps < max_steps:
            steps += 1

            # training = exploration/sampling
            a_dqn = dqn.act(obs_dqn, greedy=False)
            a_a2c, logp, value, entropy = a2c.act(obs_a2c, greedy=False)

            next_obs_dqn, next_obs_a2c, r_dqn, r_a2c, done, _ = env.step(a_dqn, a_a2c)

            ep_r_dqn += r_dqn
            ep_r_a2c += r_a2c

            # --- DQN update ---
            dqn.store(obs_dqn, a_dqn, r_dqn, next_obs_dqn, float(done))
            dqn.train_step()

            # --- A2C online update ---
            with torch.no_grad():
                next_obs_t = torch.from_numpy(next_obs_a2c).float().unsqueeze(0).to(DEVICE) / 255.0
                _, next_value = a2c.net(next_obs_t)
                next_value = next_value.squeeze(0)

            a2c.train_step(
                logp=logp,
                value=value,
                entropy=entropy,
                reward=float(r_a2c),
                next_value=next_value,
                done=float(done),
            )

            obs_dqn, obs_a2c = next_obs_dqn, next_obs_a2c

        training_stats["episodes"].append(ep)
        training_stats["dqn_rewards"].append(float(ep_r_dqn))
        training_stats["a2c_rewards"].append(float(ep_r_a2c))
        training_stats["steps"].append(int(steps))

        print(f"Episode {ep:4d} | DQN: {ep_r_dqn:6.1f} | A2C: {ep_r_a2c:6.1f} | Steps: {steps:4d}")

        # checkpoints
        if (ep + 1) % SAVE_CHECKPOINT_EVERY == 0:
            dqn.save_checkpoint(os.path.join(run_dir, f"dqn_ep{ep+1}.pt"))
            a2c.save_checkpoint(os.path.join(run_dir, f"a2c_ep{ep+1}.pt"))
            save_json(training_stats, os.path.join(run_dir, "training_stats.json"))

    # final save
    dqn.save_checkpoint(os.path.join(run_dir, "dqn_final.pt"))
    a2c.save_checkpoint(os.path.join(run_dir, "a2c_final.pt"))
    save_json(training_stats, os.path.join(run_dir, "training_stats.json"))
    env.close()

    # ===============================
    # 10 evaluation matches
    # ===============================
    print("\n" + "=" * 70)
    print("Starting 10 evaluation matches.")
    print("=" * 70)

    eval_env = MK2TwoAgentEnv(render=True)

    dqn_total, a2c_total, ties = 0, 0, 0
    num_matches = 10
    eval_results = []

    for i in range(num_matches):
        print(f"\n######## Evaluation Match {i + 1}/{num_matches} ########")
        winner = run_eval_match(eval_env, dqn, a2c)
        eval_results.append({"match": i + 1, "winner": winner})

        if winner == "DQN":
            dqn_total += 1
        elif winner == "A2C":
            a2c_total += 1
        else:
            ties += 1

        print(f"Match {i + 1} winner: {winner}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS ACROSS 10 MATCHES")
    print("=" * 70)
    print(f"DQN wins: {dqn_total}")
    print(f"A2C wins: {a2c_total}")
    print(f"Ties    : {ties}")
    print("=" * 70)

    save_json(
        {
            "dqn_wins": dqn_total,
            "a2c_wins": a2c_total,
            "ties": ties,
            "matches": eval_results,
        },
        os.path.join(run_dir, "evaluation_results.json"),
    )

    eval_env.close()
    cv2.destroyAllWindows()
    print(f"\nAll results saved to: {run_dir}")


if __name__ == "__main__":
    main()
