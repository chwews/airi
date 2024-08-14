import gymnasium as gym
import numpy as np


class DarkRoom(gym.Env):
    """
    Class for DarkRoom environment
    
    Attributes:
    size (int): size of the room
    agen_pos (tuple(int, int)): current position of the agent (x, y)
    goal_pos (tuple(int, int)): position of goal(exit)
    observation_space (int): number of possible positions(observations)
    action_space (int): number of possible actions
    terminate_on_goal (bool): flag for returning termination on goal reach
    random_start (bool): flag for start in random position or center
    render_mode (str): attribute for gym rendering
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 1}

    def __init__(self, size=9, goal=None, random_start=True, terminate_on_goal=False, render_mode="rgb_array"):
        self.size = size
        self.agent_pos = None

        if goal is not None:
            self.goal_pos = np.asarray(goal)
            assert self.goal_pos.ndim == 1
        else:
            self.goal_pos = self.generate_goal()

        self.observation_space = gym.spaces.Discrete(self.size**2)
        self.action_space = gym.spaces.Discrete(5)

        self.action_to_direction = {
            0: np.array((0, 0), dtype=np.float32),  # noop
            1: np.array((-1, 0), dtype=np.float32),  # up
            2: np.array((0, 1), dtype=np.float32),  # right
            3: np.array((1, 0), dtype=np.float32),  # down
            4: np.array((0, -1), dtype=np.float32),  # left
        }
        self.center_pos = (self.size // 2, self.size // 2)
        self.terminate_on_goal = terminate_on_goal
        self.render_mode = render_mode
        self.random_start = random_start

    def generate_pos(self):
        return self.np_random.integers(0, self.size, size=2).astype(np.float32)

    def generate_goal(self):
        return self.np_random.integers(0, self.size, size=2)

    def pos_to_state(self, pos):
        return int(pos[0] * self.size + pos[1])

    def state_to_pos(self, state):
        return np.array(divmod(state, self.size))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        if self.random_start:
            self.agent_pos = self.generate_pos()
        else:
            self.agent_pos = np.array(self.center_pos, dtype=np.float32)

        return self.pos_to_state(self.agent_pos), {}

    def step(self, action):
        self.agent_pos = np.clip(self.agent_pos + self.action_to_direction[action], 0, self.size - 1)

        reward = 1.0 if np.array_equal(self.agent_pos, self.goal_pos) else 0.0
        terminated = True if reward and self.terminate_on_goal else False

        return self.pos_to_state(self.agent_pos), reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            grid = np.full((self.size, self.size, 3), fill_value=(255, 255, 255), dtype=np.uint8)
            grid[self.goal_pos[0], self.goal_pos[1]] = (255, 0, 0)
            grid[int(self.agent_pos[0]), int(self.agent_pos[1])] = (0, 255, 0)
            return grid