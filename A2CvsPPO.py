import gym.utils.seeding as seeding
import torch
import cv2

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
# Evaluation
# ===============================
def run_eval_match(env, ppo, a2c):
    ppo_wins, a2c_wins = 0, 0

    for _ in range(3):
        obs_ppo, obs_a2c = env.reset()
        done = False

        while not done:
            a_ppo, _, _ = ppo.act(obs_ppo, greedy=True)
            a_a2c, _, _ = a2c.act(obs_a2c, greedy=True)

            obs_ppo, obs_a2c, _, _, done, _ = env.step(a_ppo, a_a2c)

        p1_hp, p2_hp = env._read_hp()
        if p1_hp > p2_hp:
            ppo_wins += 1
        elif p2_hp > p1_hp:
            a2c_wins += 1

        if ppo_wins == 2 or a2c_wins == 2:
            break

    if ppo_wins > a2c_wins:
        return "PPO"
    elif a2c_wins > ppo_wins:
        return "A2C"
    else:
        return "Tie"

# ===============================
# Main
# ===============================
def main():
    env = MK2TwoAgentEnv(render=False)

    obs_ppo, obs_a2c = env.reset()
    obs_dim = obs_ppo.shape[0]
    n_actions = env.n_actions

    ppo = PPOAgent(obs_dim, n_actions)
    a2c = A2CAgent(obs_dim, n_actions)

    num_episodes = 1000
    max_steps = 2000

    print("Training PPO vs A2C...")

    for ep in range(num_episodes):
        obs_ppo, obs_a2c = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            steps += 1

            a_ppo, logp, value = ppo.act(obs_ppo, greedy=False)
            a_a2c, logp_a2c, v_a2c = a2c.act(obs_a2c, greedy=False)

            next_obs_ppo, next_obs_a2c, r_ppo, r_a2c, done, _ = env.step(a_ppo, a_a2c)

            ppo.store((obs_ppo, a_ppo, logp, r_ppo, value, float(done)))

            with torch.no_grad():
                _, next_value = a2c.net(
                    torch.from_numpy(next_obs_a2c).float().unsqueeze(0).to(DEVICE)
                )

            a2c.train_step(
                logp=logp_a2c,
                value=v_a2c,
                reward=torch.tensor(r_a2c).to(DEVICE),
                next_value=next_value.squeeze(0),
                done=float(done),
            )

            obs_ppo, obs_a2c = next_obs_ppo, next_obs_a2c

        ppo.finish_trajectory_and_update()
        print(f"Episode {ep} finished")

    env.close()

    # ===============================
    # Evaluation
    # ===============================
    eval_env = MK2TwoAgentEnv(render=True)

    ppo_total, a2c_total, ties = 0, 0, 0

    for i in range(10):
        winner = run_eval_match(eval_env, ppo, a2c)
        print(f"Match {i+1}: {winner}")
        if winner == "PPO":
            ppo_total += 1
        elif winner == "A2C":
            a2c_total += 1
        else:
            ties += 1

    print("\nFINAL RESULTS")
    print("PPO wins:", ppo_total)
    print("A2C wins:", a2c_total)
    print("Ties:", ties)

    eval_env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
