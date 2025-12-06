import cv2
import retro
import numpy as np

GAME = "MortalKombatII-Genesis"

P1_HP_ADDR = 0x00B622
P2_HP_ADDR = 0x00B712

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

class MK2TwoAgentEnv:
    """
    Wrapper around Retro:
    - P1 = DQN agent
    - P2 = PPO agent
    - Both have 10 discrete actions
    - Reward = damage dealt to opponent this step
    """

    def __init__(self, render=False):
        self.render_enabled = render
        # players=2 so Retro gives both controllers
        self.env = retro.make(GAME, players=2, use_restricted_actions=retro.Actions.ALL)
        self.buttons = self.env.unwrapped.buttons  # list of button names for 1 pad
        self.n_buttons = len(self.buttons)         # e.g. 12

        # 10 discrete actions (same for both agents)
        self.action_meanings = [
            [],                 # 0: NOOP
            ["LEFT"],           # 1
            ["RIGHT"],          # 2
            ["DOWN"],           # 3
            ["UP"],             # 4 jump
            ["A"],              # 5 punch
            ["B"],              # 6 kick
            ["C"],              # 7 block
            ["DOWN", "A"],      # 8 crouch punch
            ["DOWN", "B"],      # 9 crouch kick
        ]
        self.n_actions = len(self.action_meanings)

        # Track health
        self.prev_p1_hp = None
        self.prev_p2_hp = None

    def reset(self):
        obs = self.env.reset()
        self.prev_p1_hp, self.prev_p2_hp = self._read_hp()
        obs_proc = preprocess_obs(obs)
        return obs_proc, obs_proc  # obs_dqn, obs_ppo (same screen)

    def step(self, a_dqn, a_ppo):
        """
        a_dqn, a_ppo: discrete actions in [0, n_actions)
        Returns:
          next_obs_dqn, next_obs_ppo, r_dqn, r_ppo, done, info
        """
        joint_action = self._build_joint_action(a_dqn, a_ppo)
        obs, _, done, info = self.env.step(joint_action)

        if self.render_enabled:
            # Use get_screen() + OpenCV instead of env.render()
            frame = self.env.get_screen()
            show_frame(frame, delay=1)

        p1_hp, p2_hp = self._read_hp()

        # Rewards: damage dealt
        r_dqn = max(0, self.prev_p2_hp - p2_hp)  # DQN = P1 damaging P2
        r_ppo = max(0, self.prev_p1_hp - p1_hp)  # PPO = P2 damaging P1

        self.prev_p1_hp, self.prev_p2_hp = p1_hp, p2_hp

        # terminal if game signals done OR someone at 0 HP
        done = bool(done or p1_hp <= 0 or p2_hp <= 0)

        obs_proc = preprocess_obs(obs)
        return obs_proc, obs_proc, r_dqn, r_ppo, done, info

    def close(self):
        self.env.close()

    def _build_single_pad_action(self, discrete_action):
        """
        Convert one discrete index -> MultiBinary(n_buttons) for one controller.
        """
        arr = np.zeros(self.n_buttons, dtype=np.uint8)
        buttons_to_press = self.action_meanings[discrete_action]
        for b in buttons_to_press:
            if b in self.buttons:
                idx = self.buttons.index(b)
                arr[idx] = 1
        return arr

    def _build_joint_action(self, a1, a2):
        """
        Two discrete actions -> joint MultiBinary for 2 pads.
        Layout: [P1 buttons..., P2 buttons...]
        """
        pad1 = self._build_single_pad_action(a1)
        pad2 = self._build_single_pad_action(a2)
        return np.concatenate([pad1, pad2], axis=0)

    def _read_hp(self):
        """
        Read HP from RAM using addresses you found.
        Returns (p1_hp, p2_hp) as ints.
        """
        ram = self.env.get_ram()
        # defensive check in case RAM is smaller than addresses
        if P1_HP_ADDR < len(ram):
            p1 = int(ram[P1_HP_ADDR])
        else:
            p1 = 120
        if P2_HP_ADDR < len(ram):
            p2 = int(ram[P2_HP_ADDR])
        else:
            p2 = 120

        # Clamp to reasonable range
        p1 = max(0, min(120, p1))
        p2 = max(0, min(120, p2))
        return p1, p2