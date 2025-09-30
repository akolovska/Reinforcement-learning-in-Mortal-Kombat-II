import gym
import numpy as np

class Discretizer(gym.ActionWrapper):
    """
    Converts a MultiBinary Retro action space into a Discrete action space using predefined combos.
    """
    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary), \
            f"Expected MultiBinary, got {type(env.action_space)}"

        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []

        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()
