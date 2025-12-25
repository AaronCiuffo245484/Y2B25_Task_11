"""
OT2 Gym Wrapper - POTENTIAL-BASED SHAPING REWARD
Level 2: Theoretically grounded reward shaping (Ng et al., 1999)
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    """OT2 Environment with Potential-Based Shaping Reward"""

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
        self.previous_distance = None
        
        print("=" * 70)
        print("REWARD ALGORITHM: POTENTIAL-BASED SHAPING (Level 2)")
        print("Theory: Ng et al. 1999 - Preserves optimal policy")
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
        
        # Initialize previous distance
        self.previous_distance = np.linalg.norm(current_pos - self.goal_position)
        
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
        
        # Update previous distance for next step
        self.previous_distance = distance_to_goal
        
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
        POTENTIAL-BASED SHAPING REWARD (Level 2)
        Theory: Reward = R(s,a,s') + γΦ(s') - Φ(s)
        Where Φ is a potential function (here: negative squared distance)
        
        This guarantees optimal policy preservation (Ng et al., 1999)
        Rewards improvement in potential, not absolute distance
        """
        # Potential function: negative squared distance
        current_potential = -(distance_to_goal ** 2)
        previous_potential = -(self.previous_distance ** 2)
        
        # Shaping reward: change in potential (scaled up for visibility)
        shaping_reward = (current_potential - previous_potential) * 100
        
        # Base reward: success bonus
        base_reward = 100.0 if distance_to_goal < self.target_threshold else 0.0
        
        # Small time penalty
        time_penalty = -0.01
        
        total_reward = base_reward + shaping_reward + time_penalty
        
        return float(total_reward)

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
