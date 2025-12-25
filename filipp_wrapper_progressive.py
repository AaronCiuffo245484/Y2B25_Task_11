"""
OT2 Gym Wrapper - PROGRESSIVE CURRICULUM REWARD
Level 2: Multi-stage hierarchical learning
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    """OT2 Environment with Progressive Curriculum Reward"""

    def __init__(self, render=False, max_steps=500, target_threshold=0.015):
        super(OT2Env, self).__init__()
        
        self.render_mode = render
        self.max_steps = max_steps
        self.target_threshold = target_threshold
        
        self.sim = Simulation(num_agents=1, render=render)
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        self.steps = 0
        self.goal_position = None
        
        print("=" * 70)
        print("REWARD ALGORITHM: PROGRESSIVE CURRICULUM (Level 2)")
        print("=" * 70)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.goal_position = np.random.uniform(
            self.workspace_low, self.workspace_high
        ).astype(np.float32)
        
        state_dict = self.sim.reset(num_agents=1)
        current_pos = self._extract_position(state_dict)
        observation = np.concatenate([current_pos, self.goal_position], dtype=np.float32)
        
        self.steps = 0
        return observation, {}

    def step(self, action):
        scaled_action = self.workspace_low + (action + 1.0) * (self.workspace_high - self.workspace_low) / 2.0
        full_action = [*scaled_action, 0]
        
        state_dict = self.sim.run([full_action])
        current_pos = self._extract_position(state_dict)
        observation = np.concatenate([current_pos, self.goal_position], dtype=np.float32)
        
        distance_to_goal = np.linalg.norm(current_pos - self.goal_position)
        
        reward = self._calculate_reward(current_pos, distance_to_goal)
        
        terminated = bool(distance_to_goal < self.target_threshold)
        truncated = bool(self.steps >= self.max_steps)
        
        info = {
            'distance_to_goal': float(distance_to_goal),
            'current_position': current_pos.tolist(),
            'goal_position': self.goal_position.tolist()
        }
        
        self.steps += 1
        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, current_pos, distance_to_goal):
        """
        PROGRESSIVE CURRICULUM REWARD (Level 2)
        Multi-stage reward with positive intermediate bonuses
        Theory: Curriculum learning guides agent through difficulty stages
        """
        if distance_to_goal < self.target_threshold:
            # Stage 4: Success!
            reward = 100.0
        elif distance_to_goal < 0.010:  # Within 10mm
            # Stage 3: Close zone - positive rewards for precision
            reward = 10.0 - distance_to_goal * 200
        elif distance_to_goal < 0.050:  # Within 50mm
            # Stage 2: Medium zone - moderate penalty
            reward = -distance_to_goal * 50
        else:
            # Stage 1: Far zone - gentle penalty
            reward = -distance_to_goal * 10
        
        # Small time penalty across all stages
        reward -= 0.01
        
        return float(reward)

    def _extract_position(self, state_dict):
        robotId = list(sorted(state_dict.keys()))[0]
        robot_state = state_dict.get(robotId, {})
        position = np.array(
            robot_state.get('pipette_position', [0.0, 0.0, 0.0]),
            dtype=np.float32
        )
        return position

    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.close()
