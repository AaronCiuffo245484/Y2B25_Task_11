"""
OT2 Gym Wrapper - MULTI-OBJECTIVE EFFICIENCY REWARD
Level 3: Balances distance, velocity, and time efficiency
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    """OT2 Environment with Multi-Objective Efficiency Reward"""

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
        self.previous_position = None
        
        print("=" * 70)
        print("REWARD ALGORITHM: MULTI-OBJECTIVE EFFICIENCY (Level 3)")
        print("Objectives: Distance + Velocity + Time + Early Success Bonus")
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
        
        self.previous_position = current_pos.copy()
        self.steps = 0
        return observation, {}

    def step(self, action):
        scaled_action = self.workspace_low + (action + 1.0) * (self.workspace_high - self.workspace_low) / 2.0
        full_action = [*scaled_action, 0]
        
        state_dict = self.sim.run([full_action])
        current_pos = self._extract_position(state_dict)
        observation = np.concatenate([current_pos, self.goal_position], dtype=np.float32)
        
        distance_to_goal = np.linalg.norm(current_pos - self.goal_position)
        
        # Calculate velocity (change in position)
        velocity = current_pos - self.previous_position
        
        reward = self._calculate_reward(current_pos, distance_to_goal, velocity)
        
        # Update previous position for next step
        self.previous_position = current_pos.copy()
        
        terminated = bool(distance_to_goal < self.target_threshold)
        truncated = bool(self.steps >= self.max_steps)
        
        info = {
            'distance_to_goal': float(distance_to_goal),
            'current_position': current_pos.tolist(),
            'goal_position': self.goal_position.tolist(),
            'velocity_magnitude': float(np.linalg.norm(velocity))
        }
        
        self.steps += 1
        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, current_pos, distance_to_goal, velocity):
        """
        MULTI-OBJECTIVE EFFICIENCY REWARD (Level 3)
        Balances multiple competing objectives:
        1. Distance to goal (primary objective)
        2. Movement efficiency (velocity penalty for jerky motion)
        3. Time cost (step penalty)
        4. Early success bonus (incentivize fast convergence)
        
        Theory: Pareto optimization, multi-objective RL
        """
        # Objective 1: Distance component (primary)
        distance_component = -distance_to_goal * 10
        
        # Objective 2: Velocity penalty (smoothness)
        # Penalize excessive velocity to encourage smooth motion
        velocity_magnitude = np.linalg.norm(velocity)
        velocity_penalty = -0.1 * velocity_magnitude
        
        # Objective 3: Time penalty (efficiency)
        time_penalty = -0.01
        
        # Objective 4: Success bonus (sparse component)
        if distance_to_goal < self.target_threshold:
            success_bonus = 100.0
        else:
            success_bonus = 0.0
        
        # Objective 5: Early success bonus (efficiency incentive)
        # Reward reaching goal quickly
        if distance_to_goal < 0.020 and self.steps < 100:
            efficiency_bonus = 20.0 * (1.0 - self.steps / 100.0)  # Higher bonus if earlier
        else:
            efficiency_bonus = 0.0
        
        # Combine all objectives
        total_reward = (
            distance_component + 
            velocity_penalty + 
            time_penalty + 
            success_bonus + 
            efficiency_bonus
        )
        
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
