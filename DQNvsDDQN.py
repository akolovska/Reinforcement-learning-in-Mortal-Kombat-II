import gym.utils.seeding as seeding
import torch
import cv2
import os
import json
from datetime import datetime

from agents.DQN import DQNAgent
from agents.DDQN import DDQNAgent
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
# Checkpoint settings
# ===============================
CHECKPOINT_DIR = "checkpoints"
SAVE_CHECKPOINT_EVERY = 100  # save every N episodes

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {path}")

def save_agent_checkpoint(agent, path):
    """
    Uses agent.save_checkpoint(path) if it exists.
    Otherwise saves common fields (q_net/target_net/optimizer) if present.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if hasattr(agent, "save_checkpoint") and callable(getattr(agent, "save_checkpoint")):
        agent.save_checkpoint(path)
        return

    payload = {}
    # common DQN-like naming
    if hasattr(agent, "q_net"):
        payload["q_net_state_dict"] = agent.q_net.state_dict()
    if hasattr(agent, "target_net"):
        payload["target_net_state_dict"] = agent.target_net.state_dict()
    if hasattr(agent, "optimizer"):
        payload["optimizer_state_dict"] = agent.optimizer.state_dict()
    # optionally store step counters / epsilon if present
    if hasattr(agent, "step_count"):
        payload["step_count"] = agent.step_count
    if hasattr(agent, "epsilon"):
        payload["epsilon"] = agent.epsilon

    torch.save(payload, path)
    print(f"Checkpoint saved to {path} (fallback saver)")

# ===============================
# Evaluation (Best-of-3)
# ===============================
def run_eval_match(env, dqn, ddqn):
    """
    One match = Best-of-3 rounds.
    Returns winner: "DQN" | "DDQN" | "Tie"
    """
    dqn_wins, ddqn_wins = 0, 0

    for round_idx in range(3):
        obs_dqn, obs_ddqn = env.reset()
        done = False

        while not done:
            # deterministic actions for eval (NO exploration)
            a_dqn = dqn.act(obs_dqn, greedy=True)
            a_ddqn = ddqn.act(obs_ddqn, greedy=True)

            obs_dqn, obs_ddqn, _, _, done, _ = env.step(a_dqn, a_ddqn)

        p1_hp, p2_hp = env._read_hp()  # P1 = DQN, P2 = DDQN

        if p1_hp > p2_hp:
            dqn_wins += 1
        elif p2_hp > p1_hp:
            ddqn_wins += 1

        if dqn_wins == 2 or ddqn_wins == 2:
            break

    if dqn_wins > ddqn_wins:
        return "DQN"
    elif ddqn_wins > dqn_wins:
        return "DDQN"
    else:
        return "Tie"

# ===============================
# Main
# ===============================
def main():
    # run dir like your template
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(CHECKPOINT_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Run dir: {run_dir}")
    print(f"Device: {DEVICE}")

    # -------- TRAIN --------
    env = MK2TwoAgentEnv(render=False)
    obs_dqn, obs_ddqn = env.reset()
    obs_dim = obs_dqn.shape[0]
    n_actions = env.n_actions

    USE_PER = True  # if your DDQN supports it
    dqn = DQNAgent(obs_dim, n_actions)
    ddqn = DDQNAgent(obs_dim, n_actions, use_per=USE_PER)

    num_episodes = 1000
    max_steps = 2000

    training_stats = {
        "episodes": [],
        "dqn_rewards": [],
        "ddqn_rewards": [],
        "steps": [],
        "timestamp": timestamp,
        "use_per": USE_PER,
    }

    print("\n" + "=" * 70)
    print(f"Training DQN vs DDQN (PER={USE_PER}) for {num_episodes} episodes...")
    print(f"Saving checkpoints every {SAVE_CHECKPOINT_EVERY} episodes")
    print("=" * 70)

    for ep in range(num_episodes):
        obs_dqn, obs_ddqn = env.reset()
        done = False
        steps = 0
        ep_r_dqn, ep_r_ddqn = 0.0, 0.0

        while not done and steps < max_steps:
            steps += 1

            # exploration ON during training
            a_dqn = dqn.act(obs_dqn, greedy=False)
            a_ddqn = ddqn.act(obs_ddqn, greedy=False)

            next_obs_dqn, next_obs_ddqn, r_dqn, r_ddqn, done, _ = env.step(a_dqn, a_ddqn)

            ep_r_dqn += r_dqn
            ep_r_ddqn += r_ddqn

            dqn.store(obs_dqn, a_dqn, r_dqn, next_obs_dqn, float(done))
            ddqn.store(obs_ddqn, a_ddqn, r_ddqn, next_obs_ddqn, float(done))

            dqn.train_step()
            ddqn.train_step()

            obs_dqn, obs_ddqn = next_obs_dqn, next_obs_ddqn

        # record + print like template
        training_stats["episodes"].append(ep)
        training_stats["dqn_rewards"].append(float(ep_r_dqn))
        training_stats["ddqn_rewards"].append(float(ep_r_ddqn))
        training_stats["steps"].append(int(steps))

        print(f"Episode {ep:4d} | DQN: {ep_r_dqn:6.1f} | DDQN: {ep_r_ddqn:6.1f} | Steps: {steps:4d}")

        # periodic saves
        if (ep + 1) % SAVE_CHECKPOINT_EVERY == 0:
            save_agent_checkpoint(dqn, os.path.join(run_dir, f"dqn_ep{ep+1}.pt"))
            save_agent_checkpoint(ddqn, os.path.join(run_dir, f"ddqn_ep{ep+1}.pt"))
            save_json(training_stats, os.path.join(run_dir, "training_stats.json"))

    env.close()

    # final save
    print("\n" + "=" * 70)
    print("Training finished! Saving final checkpoints...")
    print("=" * 70)

    save_agent_checkpoint(dqn, os.path.join(run_dir, "dqn_final.pt"))
    save_agent_checkpoint(ddqn, os.path.join(run_dir, "ddqn_final.pt"))
    save_json(training_stats, os.path.join(run_dir, "training_stats.json"))

    # -------- EVAL (10 matches) --------
    print("\n" + "=" * 70)
    print("Starting 10 evaluation matches (Best-of-3 each)...")
    print("=" * 70)

    eval_env = MK2TwoAgentEnv(render=True)

    dqn_total, ddqn_total, ties = 0, 0, 0
    num_matches = 10
    eval_results = []

    for i in range(num_matches):
        print(f"\n######## Evaluation Match {i+1}/{num_matches} ########")
        winner = run_eval_match(eval_env, dqn, ddqn)
        print(f"Match {i+1} winner: {winner}")

        if winner == "DQN":
            dqn_total += 1
        elif winner == "DDQN":
            ddqn_total += 1
        else:
            ties += 1

        eval_results.append({"match": i + 1, "winner": winner})

    print("\n" + "=" * 70)
    print("FINAL RESULTS ACROSS 10 MATCHES")
    print("=" * 70)
    print(f"DQN wins : {dqn_total}")
    print(f"DDQN wins: {ddqn_total}")
    print(f"Ties     : {ties}")
    print("=" * 70)

    save_json(
        {
            "dqn_wins": dqn_total,
            "ddqn_wins": ddqn_total,
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
