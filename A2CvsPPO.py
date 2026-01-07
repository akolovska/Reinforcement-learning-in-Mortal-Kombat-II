import gym.utils.seeding as seeding
import os
import json
from datetime import datetime
import cv2
import torch

from agents.PPO import PPOAgent
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

# ===============================
# Checkpoint + logging settings
# ===============================
CHECKPOINT_DIR = "checkpoints"
SAVE_CHECKPOINT_EVERY = 100  # episodes
NUM_EVAL_MATCHES = 10        # matches at the end
MAX_EVAL_ROUNDS = 3          # best-of-3


def save_json(obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved: {filepath}")


def save_agent_checkpoint(agent, filepath, fallback_key="agent"):
    """
    Uses agent.save_checkpoint(path) if it exists.
    Otherwise falls back to saving .net/.optimizer state_dict if present.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if hasattr(agent, "save_checkpoint") and callable(getattr(agent, "save_checkpoint")):
        agent.save_checkpoint(filepath)
        return

    payload = {}
    if hasattr(agent, "net"):
        payload["net_state_dict"] = agent.net.state_dict()
    if hasattr(agent, "optimizer"):
        payload["optimizer_state_dict"] = agent.optimizer.state_dict()

    if not payload:
        raise RuntimeError(f"{fallback_key} has no save_checkpoint() and no (net/optimizer) to save.")

    torch.save(payload, filepath)
    print(f"{fallback_key} checkpoint saved to {filepath} (fallback)")


# ===============================
# Evaluation (Best-of-3)
# ===============================
def run_eval_match(env, ppo, a2c):
    """
    One match = Best-of-3 rounds.
    Returns winner: "PPO" | "A2C" | "Tie"
    """
    ppo_wins, a2c_wins = 0, 0

    for round_idx in range(MAX_EVAL_ROUNDS):
        obs_ppo, obs_a2c = env.reset()
        done = False

        print(f"\n=== Evaluation Round {round_idx + 1} ===")

        while not done:
            # deterministic evaluation
            a_ppo, _, _ = ppo.act(obs_ppo, greedy=True)
            a_a2c, _, _, _ = a2c.act(obs_a2c, greedy=True)

            next_obs_ppo, next_obs_a2c, _, _, done, _ = env.step(a_ppo, a_a2c)
            obs_ppo, obs_a2c = next_obs_ppo, next_obs_a2c

        p1_hp, p2_hp = env._read_hp()  # P1=first agent (PPO), P2=second agent (A2C)
        print(f"End HP -> PPO (P1): {p1_hp}, A2C (P2): {p2_hp}")

        if p1_hp > p2_hp:
            ppo_wins += 1
            print("PPO wins round!")
        elif p2_hp > p1_hp:
            a2c_wins += 1
            print("A2C wins round!")
        else:
            print("Tie round")

        if ppo_wins == 2 or a2c_wins == 2:
            break

    if ppo_wins > a2c_wins:
        return "PPO"
    elif a2c_wins > ppo_wins:
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

    # -------- TRAIN --------
    env = MK2TwoAgentEnv(render=False)

    obs_ppo, obs_a2c = env.reset()
    obs_dim = obs_ppo.shape[0]
    n_actions = env.n_actions

    ppo = PPOAgent(obs_dim, n_actions)
    a2c = A2CAgent(obs_dim, n_actions)

    num_episodes = 1000
    max_steps = 2000

    training_stats = {
        "episodes": [],
        "ppo_rewards": [],
        "a2c_rewards": [],
        "steps": [],
        "timestamp": timestamp,
    }

    print("\n" + "=" * 70)
    print(f"Training PPO vs A2C for {num_episodes} episodes.")
    print(f"Checkpoints will be saved every {SAVE_CHECKPOINT_EVERY} episodes")
    print("=" * 70)

    for ep in range(num_episodes):
        obs_ppo, obs_a2c = env.reset()
        done = False
        steps = 0
        ep_r_ppo = 0.0
        ep_r_a2c = 0.0

        while not done and steps < max_steps:
            steps += 1

            # training = exploration/sampling
            a_ppo, logp_ppo, value_ppo = ppo.act(obs_ppo, greedy=False)
            a_a2c, logp_a2c, value_a2c, ent_a2c = a2c.act(obs_a2c, greedy=False)

            next_obs_ppo, next_obs_a2c, r_ppo, r_a2c, done, _ = env.step(a_ppo, a_a2c)

            ep_r_ppo += float(r_ppo)
            ep_r_a2c += float(r_a2c)

            # --- PPO stores trajectory; updates after episode ---
            ppo.store((obs_ppo, a_ppo, logp_ppo, r_ppo, value_ppo, float(done)))

            # --- A2C online update ---
            with torch.no_grad():
                next_obs_t = torch.from_numpy(next_obs_a2c).float().unsqueeze(0).to(DEVICE) / 255.0
                _, next_value_a2c = a2c.net(next_obs_t)
                next_value_a2c = next_value_a2c.squeeze(0)

            a2c.train_step(
                logp=logp_a2c,
                value=value_a2c,
                entropy=ent_a2c,
                reward=float(r_a2c),
                next_value=next_value_a2c,
                done=float(done),
            )

            obs_ppo, obs_a2c = next_obs_ppo, next_obs_a2c

        # end episode: PPO update
        ppo.finish_trajectory_and_update()

        training_stats["episodes"].append(ep)
        training_stats["ppo_rewards"].append(float(ep_r_ppo))
        training_stats["a2c_rewards"].append(float(ep_r_a2c))
        training_stats["steps"].append(int(steps))

        print(f"Episode {ep:4d} | PPO: {ep_r_ppo:6.1f} | A2C: {ep_r_a2c:6.1f} | Steps: {steps:4d}")

        # checkpoints
        if (ep + 1) % SAVE_CHECKPOINT_EVERY == 0:
            save_agent_checkpoint(ppo, os.path.join(run_dir, f"ppo_ep{ep+1}.pt"), fallback_key="PPO")
            save_agent_checkpoint(a2c, os.path.join(run_dir, f"a2c_ep{ep+1}.pt"), fallback_key="A2C")
            save_json(training_stats, os.path.join(run_dir, "training_stats.json"))

    # final save
    save_agent_checkpoint(ppo, os.path.join(run_dir, "ppo_final.pt"), fallback_key="PPO")
    save_agent_checkpoint(a2c, os.path.join(run_dir, "a2c_final.pt"), fallback_key="A2C")
    save_json(training_stats, os.path.join(run_dir, "training_stats.json"))
    env.close()

    # -------- EVAL (10 matches, best-of-3 each) --------
    print("\n" + "=" * 70)
    print(f"Starting {NUM_EVAL_MATCHES} evaluation matches (Best-of-3 each).")
    print("=" * 70)

    eval_env = MK2TwoAgentEnv(render=True)

    ppo_total, a2c_total, ties = 0, 0, 0
    eval_results = []

    for i in range(NUM_EVAL_MATCHES):
        print(f"\n######## Evaluation Match {i + 1}/{NUM_EVAL_MATCHES} ########")
        winner = run_eval_match(eval_env, ppo, a2c)
        eval_results.append({"match": i + 1, "winner": winner})

        if winner == "PPO":
            ppo_total += 1
        elif winner == "A2C":
            a2c_total += 1
        else:
            ties += 1

        print(f"Match {i + 1} winner: {winner}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS ACROSS EVALUATION MATCHES")
    print("=" * 70)
    print(f"PPO wins: {ppo_total}")
    print(f"A2C wins: {a2c_total}")
    print(f"Ties    : {ties}")
    print("=" * 70)

    save_json(
        {
            "ppo_wins": ppo_total,
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
