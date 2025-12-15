# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (GPU Balanced)
#     language: python
#     name: python3
# ---

# %reload_ext autoreload
# %autoreload 2

from library.simulation import Simulation
from library.robot_control import find_workspace, get_position, get_velocities
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000,
                 action_space=None,
                 target=None,
                 target_threshold=0.001):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=render)

        # Define action and observation space
        self.action_space = None
        self.observation_space = None
        
        self.target = target
        self.target_threshold = target_threshold

        # keep track of the number of steps
        self.steps = 0
        self.goal_position = None

    @property
    def observation_space(self):
        return self._observation_space
    
    @observation_space.setter
    def observation_space(self, obs_space):
        if not obs_space:
            low = -np.inf
            high = np.inf
            shape = (9, )
            d_type = np.float32

            obs_space = (low, high, shape, d_type)

        self._observation_space = spaces.Box(*obs_space)

    @property
    def action_space(self):
        return self._action_space
    
    @action_space.setter
    def action_space(self, action_space=None):
        if not action_space:
            # Action space is ALWAYS [-1, 1] for RL
            low = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
            high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            shape = (3, )
            d_type = np.float32
            action_space = (low, high, shape, d_type)
        
        self._action_space = spaces.Box(*action_space)

        # Store workspace bounds for scaling later
        workspace = find_workspace(self.sim)
        self.workspace_low = np.array([workspace['x_min'],
                                    workspace['y_min'],
                                    workspace['z_min']], dtype=np.float32)
        self.workspace_high = np.array([workspace['x_max'],
                                        workspace['y_max'],
                                        workspace['z_max']], dtype=np.float32)        
        

    def reset(self, seed=None):
        # being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        # set a random goal position for the agent, consisting of x, y, and z coordinates within the working area
        self.goal_position = np.array([
            np.random.uniform(self.workspace_low[0], self.workspace_high[0]),
            np.random.uniform(self.workspace_low[1], self.workspace_high[1]),
            np.random.uniform(self.workspace_low[2], self.workspace_high[2])], 
            dtype=np.float32)

        # Call the environment reset function
        state_dict = self.sim.reset(num_agents=1)
        
        # Extract observation
        current_pos, current_vel = self._extract_observation(state_dict)

        observation = np.concatenate([
            current_pos,
            current_vel,
            self.goal_position], 
            dtype=np.float32)

        # Reset the number of steps
        self.steps = 0

        info = {}
        return observation, info

    def step(self, action):
        # Scale action from [-1, 1] to workspace bounds
        scaled_action = self.workspace_low + (action + 1.0) * (self.workspace_high - self.workspace_low) / 2.0
        
        # Execute one time step within the environment
        full_action = [*scaled_action, 0]

        # Call the environment step function
        state_dict = self.sim.run([full_action])

        # Extract observations
        current_pos, current_vel = self._extract_observation(state_dict)

        observation = np.concatenate([
            current_pos,
            current_vel,
            self.goal_position], 
            dtype=np.float32
            )

        # Distance to Goal
        distance_to_goal = np.linalg.norm(current_pos - self.goal_position)

        # Calculate the reward with improved structure
        reward = self._calculate_reward(current_pos, current_vel, distance_to_goal)
        
        # Check if task is complete
        terminated = bool(distance_to_goal < self.target_threshold)
        
        # Check truncation - max steps reached
        truncated = bool(self.steps >= self.max_steps)
        
        # Info for debugging and logging
        info = {
            'distance_to_goal': float(distance_to_goal),
            'current_position': current_pos.tolist(),
            'goal_position': self.goal_position.tolist()
        }
        
        # increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, current_pos, current_vel, distance_to_goal):
        """
        Reward structure prioritizing:
        1. Precision - large bonus for reaching goal
        2. Distance shaping - quadratic penalty when far from goal
        3. Smoothness - penalty for excessive velocity
        4. Speed - small time penalty to encourage efficiency
        
        Parameters:
        -----------
        current_pos : np.ndarray
            Current XYZ position
        current_vel : np.ndarray
            Current XYZ velocities
        distance_to_goal : float
            Euclidean distance to goal
        
        Returns:
        --------
        float
            Computed reward
        """
        reward = 0.0
        
        # 1. Goal achievement bonus (most important)
        if distance_to_goal < self.target_threshold:  # <1mm
            reward += 100.0
        
        # 2. Distance shaping (quadratic - stronger penalty when far)
        # Scaled so max penalty ~-1.0 at workspace limits
        max_distance = np.linalg.norm(self.workspace_high - self.workspace_low)
        reward -= (distance_to_goal / max_distance) ** 2
        
        # 3. Velocity penalty (encourages smooth motion)
        velocity_magnitude = np.linalg.norm(current_vel)
        reward -= 0.01 * velocity_magnitude
        
        # 4. Time penalty (encourages efficiency)
        reward -= 0.01
        
        return float(reward)

    def render(self, mode='human'):
        pass
    
    def close(self):
        self.sim.close()

    @staticmethod
    def _extract_observation(state_dict, robotId=None):
        """
        Extract observation from state dictionary.
        
        Parameters:
        -----------
        state_dict : dict
            Dictionary from sim.get_states() or sim.run()
        robotId : str, optional
            Robot ID to extract from. If None, uses first robot.
        
        Returns:
        --------
        tuple
            (position, velocity) as numpy arrays
        """
        if not robotId:
            robotId = list(sorted(state_dict.keys()))[0]
        
        robot_state = state_dict.get(robotId, {})
        
        # Extract position
        position = np.array(
            robot_state.get('pipette_position', [None, None, None]),
            dtype=np.float32
        )
        
        # Extract velocities from joint states
        if 'joint_states' in robot_state:
            velocity = np.array(
                [robot_state['joint_states'][j]['velocity'] 
                for j in sorted(robot_state['joint_states'].keys())],
                dtype=np.float32
            )
        else:
            velocity = np.array([None, None, None], dtype=np.float32)
        
        return position, velocity