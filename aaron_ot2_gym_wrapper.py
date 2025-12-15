"""
OT2 Gym Wrapper
Custom Gymnasium environment for OT-2 robot control

Author: Aaron Ciuffo
Course: ADS-AI Y2B Block B - Task 11
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    """
    Custom Gymnasium environment for OT-2 robot control.
    
    Goal: Train agent to move pipette tip to random target positions within workspace.
    
    Observation Space: 6D [current_x, current_y, current_z, goal_x, goal_y, goal_z]
    Action Space: 3D [x, y, z] normalized to [-1, 1]
    Reward: Negative distance to goal (baseline - can be modified)
    
    Parameters
    ----------
    render : bool
        Whether to render the simulation visually
    max_steps : int
        Maximum steps per episode before truncation
    target_threshold : float
        Distance threshold (meters) for successful goal achievement
    """
    
    def __init__(self, render=False, max_steps=1000, target_threshold=0.001):
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
        
        # OT-2 workspace bounds (fixed values from robot specifications)
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        # Episode tracking
        self.steps = 0
        self.goal_position = None
    
    def reset(self, seed=None):
        """
        Reset environment to initial state with new random goal.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        observation : np.ndarray
            6D array [current_pos, goal_pos]
        info : dict
            Empty dictionary (for Gymnasium compatibility)
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
        
        # Create observation
        observation = np.concatenate([current_pos, self.goal_position], dtype=np.float32)
        
        # Reset step counter
        self.steps = 0
        
        info = {}
        return observation, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Parameters
        ----------
        action : np.ndarray
            3D array normalized to [-1, 1]
        
        Returns
        -------
        observation : np.ndarray
            6D array [current_pos, goal_pos]
        reward : float
            Reward value for this step
        terminated : bool
            True if goal reached (task success)
        truncated : bool
            True if max steps reached (timeout)
        info : dict
            Dictionary with debugging information
        """
        # Scale action from [-1, 1] to workspace bounds
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
        
        # Calculate reward
        reward = self._calculate_reward(current_pos, distance_to_goal)
        
        # Terminated: goal reached
        terminated = bool(distance_to_goal < self.target_threshold)
        
        # Truncated: max steps reached
        truncated = bool(self.steps >= self.max_steps)
        
        # Info for debugging and callback logging
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
    
    # ========================================================================
    # REWARD FUNCTION - MODIFY THIS SECTION FOR EXPERIMENTS
    # ========================================================================
    
    def _calculate_reward(self, current_pos, distance_to_goal):
        """
        Calculate reward based on current state.
        
        BASELINE: Simple negative distance reward
        MODIFY THIS METHOD to experiment with different reward functions.
        
        Parameters
        ----------
        current_pos : np.ndarray
            Current pipette position [x, y, z]
        distance_to_goal : float
            Euclidean distance to goal
        
        Returns
        -------
        reward : float
            Reward value for this step
        
        Examples of alternative reward functions:
        --------------------------------------------------------
        # Sparse reward (only reward at goal)
        reward = 100.0 if distance_to_goal < self.target_threshold else -0.01
        
        # Shaped with goal bonus
        reward = -distance_to_goal
        if distance_to_goal < self.target_threshold:
            reward += 100.0 
        
        # Quadratic distance penalty + goal bonus + time penalty
        max_dist = np.linalg.norm(self.workspace_high - self.workspace_low)
        reward = -(distance_to_goal / max_dist) ** 2 - 0.01
        if distance_to_goal < self.target_threshold:
            reward += 100.0
        
        # Exponential shaping
        reward = -np.exp(distance_to_goal) - 0.01
        --------------------------------------------------------
        """
        # BASELINE: Simple negative distance
        # reward = float(-distance_to_goal)

        # Shaped with goal bonus
        # reward = -distance_to_goal
        # if distance_to_goal < self.target_threshold:
        #     reward += 100.0         
        

        # Quadratic distance penalty + goal bonus + time penalty
        max_dist = np.linalg.norm(self.workspace_high - self.workspace_low)
        reward = -(distance_to_goal / max_dist) ** 2 - 0.01
        if distance_to_goal < self.target_threshold:
            reward += 100.0
        return reward

    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _extract_position(self, state_dict):
        """
        Extract pipette position from state dictionary.
        
        Parameters
        ----------
        state_dict : dict
            Dictionary from sim.reset() or sim.run()
        
        Returns
        -------
        position : np.ndarray
            3D numpy array [x, y, z]
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
    
    # ========================================================================
    # FUTURE EXTENSIONS - Uncomment and modify as needed
    # ========================================================================
    
    # def _extract_velocity(self, state_dict):
    #     """
    #     Extract joint velocities from state dictionary.
    #     USE THIS if you want to add velocities to observation space.
    #     """
    #     robotId = list(sorted(state_dict.keys()))[0]
    #     robot_state = state_dict.get(robotId, {})
    #     
    #     if 'joint_states' in robot_state:
    #         velocity = np.array(
    #             [robot_state['joint_states'][j]['velocity'] 
    #              for j in sorted(robot_state['joint_states'].keys())],
    #             dtype=np.float32
    #         )
    #     else:
    #         velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    #     
    #     return velocity
    
    # def _calculate_velocity_penalty(self, velocity):
    #     """
    #     Calculate penalty for excessive velocity (encourages smooth motion).
    #     USE THIS if you want to add smoothness to reward.
    #     """
    #     velocity_magnitude = np.linalg.norm(velocity)
    #     penalty = 0.01 * velocity_magnitude
    #     return penalty
