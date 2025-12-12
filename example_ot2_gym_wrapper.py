"""
Example OT2 Gym Wrapper
Simplified version aligned with course documentation
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from library.simulation import Simulation
from library.robot_control import find_workspace


class OT2Env(gym.Env):
    """
    Custom Gymnasium environment for OT-2 robot control.
    
    Goal: Train agent to move pipette tip to random target positions.
    
    Observation Space: 6D [current_x, current_y, current_z, goal_x, goal_y, goal_z]
    Action Space: 3D [x, y, z] normalized to [-1, 1]
    Reward: Negative distance to goal
    """
    
    def __init__(self, render=False, max_steps=1000, target_threshold=0.002):
        super(OT2Env, self).__init__()
        
        self.render_mode = render
        self.max_steps = max_steps
        self.target_threshold = target_threshold
        
        # Create simulation
        self.sim = Simulation(num_agents=1, render=render)
        
        # Define action space: normalized [-1, 1] for RL algorithms
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space: 6D [pos_x, pos_y, pos_z, goal_x, goal_y, goal_z]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )
        
        # Get workspace bounds for action scaling
        workspace = find_workspace(self.sim)
        self.workspace_low = np.array([
            workspace['x_min'],
            workspace['y_min'],
            workspace['z_min']
        ], dtype=np.float32)
        self.workspace_high = np.array([
            workspace['x_max'],
            workspace['y_max'],
            workspace['z_max']
        ], dtype=np.float32)
        
        # Episode tracking
        self.steps = 0
        self.goal_position = None
    
    def reset(self, seed=None):
        """
        Reset environment to initial state.
        
        Returns:
            observation: 6D array [current_pos, goal_pos]
            info: Empty dict
        """
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random goal within workspace
        self.goal_position = np.random.uniform(
            self.workspace_low,
            self.workspace_high
        ).astype(np.float32)
        
        # Reset simulation
        state_dict = self.sim.reset(num_agents=1)
        
        # Extract current position
        current_pos = self._extract_position(state_dict)
        
        # Create observation: [current_x, current_y, current_z, goal_x, goal_y, goal_z]
        observation = np.concatenate([current_pos, self.goal_position], dtype=np.float32)
        
        # Reset step counter
        self.steps = 0
        
        info = {}
        return observation, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: 3D array normalized to [-1, 1]
        
        Returns:
            observation: 6D array [current_pos, goal_pos]
            reward: Float reward value
            terminated: Bool, True if goal reached
            truncated: Bool, True if max steps reached
            info: Dict with debugging information
        """
        # Scale action from [-1, 1] to workspace bounds
        # Formula: scaled = low + (action + 1) * (high - low) / 2
        scaled_action = self.workspace_low + (action + 1.0) * (self.workspace_high - self.workspace_low) / 2.0
        
        # Append 0 for drop action (not used in this task)
        full_action = [*scaled_action, 0]
        
        # Execute action in simulation
        state_dict = self.sim.run([full_action])
        
        # Extract current position
        current_pos = self._extract_position(state_dict)
        
        # Create observation
        observation = np.concatenate([current_pos, self.goal_position], dtype=np.float32)
        
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(current_pos - self.goal_position)
        
        # Reward: negative distance (agent wants to minimize distance)
        reward = float(-distance_to_goal)
        
        # Terminated: goal reached
        terminated = bool(distance_to_goal < self.target_threshold)
        
        # Truncated: max steps reached
        truncated = bool(self.steps >= self.max_steps)
        
        # Info for debugging
        info = {
            'distance_to_goal': float(distance_to_goal),
            'current_position': current_pos.tolist(),
            'goal_position': self.goal_position.tolist()
        }
        
        self.steps += 1
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """Render is handled by simulation if render=True in __init__"""
        pass
    
    def close(self):
        """Close the simulation"""
        self.sim.close()
    
    def _extract_position(self, state_dict):
        """
        Extract pipette position from state dictionary.
        
        Args:
            state_dict: Dictionary from sim.reset() or sim.run()
        
        Returns:
            position: 3D numpy array [x, y, z]
        """
        # Get first robot ID
        robotId = list(sorted(state_dict.keys()))[0]
        robot_state = state_dict.get(robotId, {})
        
        # Extract pipette position
        position = np.array(
            robot_state.get('pipette_position', [0.0, 0.0, 0.0]),
            dtype=np.float32
        )
        
        return position
