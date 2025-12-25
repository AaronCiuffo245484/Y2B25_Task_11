import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=300, target_threshold=0.005):
        super(OT2Env, self).__init__()
        self.render_mode = render
        self.max_steps = max_steps
        self.target_threshold = target_threshold
        
        # Create simulation
        self.sim = Simulation(num_agents=1, render=render)
        
        # Action space: 3D [x, y, z] velocities
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space: 6D [current_pos, goal_pos] normalized to [-1, 1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        
        # Workspace bounds
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1700], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        self.steps = 0
        self.goal_position = None

    def _normalize_position(self, pos):
        return 2.0 * (pos - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0

    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        self.goal_position = np.random.uniform(self.workspace_low, self.workspace_high).astype(np.float32)
        state_dict = self.sim.reset(num_agents=1)
        robotId = list(sorted(state_dict.keys()))[0]
        current_pos = np.array(state_dict[robotId].get('pipette_position'), dtype=np.float32)
        self.steps = 0
        obs = np.concatenate([self._normalize_position(current_pos), self._normalize_position(self.goal_position)])
        return obs, {}

    def step(self, action):
        velocity = np.asarray(action, dtype=np.float32) * 2.0 
        state_dict = self.sim.run([[float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]])
        robotId = list(sorted(state_dict.keys()))[0]
        current_pos = np.array(state_dict[robotId].get('pipette_position'), dtype=np.float32)
        dist = np.linalg.norm(current_pos - self.goal_position)
        
        # PROGRESSIVE REWARD LOGIC
        max_dist = np.linalg.norm(self.workspace_high - self.workspace_low)
        reward = -((dist / max_dist) ** 2)
        if dist < self.target_threshold:
            reward += 100.0
        elif dist < 0.02: # 20mm proximity bonus
            reward += 10.0 * np.exp(-dist * 100)

        self.steps += 1
        terminated = bool(dist < self.target_threshold)
        truncated = bool(self.steps >= self.max_steps)
        obs = np.concatenate([self._normalize_position(current_pos), self._normalize_position(self.goal_position)])
        return obs, float(reward - 0.01), terminated, truncated, {"distance_mm": dist * 1000}

    def render(self, mode='human'): pass
    def close(self): self.sim.close()
