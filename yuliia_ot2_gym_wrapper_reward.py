import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    
    def __init__(self, render=False, max_steps=300, target_threshold=0.001):
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
        
        # Define observation space: 6D normalized positions
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        
        # OT-2 workspace bounds (verified from simulation)
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1700], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        # Episode tracking
        self.steps = 0
        self.goal_position = None
        self.initial_distance = None
        
        # NEW: Settling behavior tracking
        self.steps_within_threshold = 0  # How many consecutive steps we've been settled
        self.required_settle_steps = 5   # Must stay settled for this many steps
        self.distance_history = []       # Track distances over episode
        self.velocity_history = []       # Track velocities over episode


    def reset(self, seed=None):
        """Reset environment to initial state with new random goal."""
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
        
        # Store initial distance for reward scaling
        self.initial_distance = float(np.linalg.norm(current_pos - self.goal_position))
        
        # Create normalized observation
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        # Reset step counter
        self.steps = 0
        
        # NEW: Reset settling tracking
        self.steps_within_threshold = 0
        self.distance_history = []
        self.velocity_history = []
        
        # Verify observation shape and dtype
        assert observation.shape == (6,), f"Observation shape is {observation.shape}, expected (6,)"
        assert observation.dtype == np.float32, f"Observation dtype is {observation.dtype}, expected float32"
        
        return observation, {}


    def step(self, action):
        """Execute one step in the environment."""
        # Ensure action is float32
        action = np.asarray(action, dtype=np.float32)
        
        # Scale action to velocity range
        max_velocity = 2.0
        velocity = action * max_velocity
        
        # NEW: Calculate velocity magnitude for reward and termination
        velocity_magnitude = np.linalg.norm(velocity)
        
        # Create full action array with gripper command (0)
        # Convert to list for sim.run() compatibility
        full_action = [float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]

        # Execute action in simulation
        state_dict = self.sim.run([full_action])
        
        # Extract current position
        current_pos = self._extract_position(state_dict)
        
        # Calculate distance to goal
        distance_to_goal = np.linalg.norm(current_pos - self.goal_position)
        
        # NEW: Track history for analysis
        self.distance_history.append(distance_to_goal)
        self.velocity_history.append(velocity_magnitude)
        
        # NEW: Calculate reward with velocity information
        reward = self._calculate_reward(distance_to_goal, velocity_magnitude)
        
        # NEW: Improved termination - require settling
        # Check if we're close AND moving slowly
        if distance_to_goal < self.target_threshold and velocity_magnitude < 0.04:
            self.steps_within_threshold += 1
        else:
            self.steps_within_threshold = 0  # Reset if we leave threshold or speed up
        
        # Success = stayed within threshold for required number of steps
        terminated = bool(self.steps_within_threshold >= self.required_settle_steps)
        
        # Increment step counter
        self.steps += 1
        
        # Check if max steps reached
        truncated = bool(self.steps >= self.max_steps)
        
        # Create observation
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        # Verify observation shape and dtype
        assert observation.shape == (6,), f"Observation shape is {observation.shape}, expected (6,)"
        assert observation.dtype == np.float32, f"Observation dtype is {observation.dtype}, expected float32"
        
        # Info for logging
        info = {
            'distance_to_goal': float(distance_to_goal),
            'current_position': current_pos.tolist(),
            'goal_position': self.goal_position.tolist(),
            'velocity_magnitude': float(velocity_magnitude),  # NEW: add velocity info
            'steps_settled': int(self.steps_within_threshold)  # NEW: add settling info
        }
        
        return observation, reward, terminated, truncated, info


    def _calculate_reward(self, distance_to_goal, velocity_magnitude):
        """
        IMPROVED REWARD FUNCTION
        
        Encourages settling behavior by:
        1. Progressive distance rewards (not binary)
        2. Penalizing high velocity when close
        3. Big bonus for settling (close + slow)
        4. Reduced time penalty
        """
        
        # ========== DISTANCE COMPONENT ==========
        # Progressive rewards - better precision = better reward
        if distance_to_goal < 0.001:  # Within 1mm - excellent!
            distance_reward = 100.0
        elif distance_to_goal < 0.002:  # Within 2mm - very good
            distance_reward = 50.0
        elif distance_to_goal < 0.005:  # Within 5mm - good
            distance_reward = 20.0
        else:  # Still far - linear penalty
            distance_reward = -10.0 * distance_to_goal
        
        # ========== VELOCITY COMPONENT ==========
        # When close to goal, penalize high velocity
        velocity_penalty = 0.0
        if distance_to_goal < 0.005:  # Only when close
            # Quadratic penalty: moving fast is really bad when close
            velocity_penalty = -50.0 * (velocity_magnitude ** 2)
        
        # ========== TIME COMPONENT ==========
        # Small time penalty - reduced from -0.1 to allow precision
        time_penalty = -0.05
        
        # ========== SETTLING BONUS ==========
        # Huge bonus for being close AND slow (the goal behavior!)
        settling_bonus = 0.0
        if distance_to_goal < self.target_threshold and velocity_magnitude < 0.04:
            settling_bonus = 200.0
        
        # Combine all components
        reward = distance_reward + velocity_penalty + time_penalty + settling_bonus
        
        return float(reward)
    

    def render(self, mode='human'):
        """Render is handled by simulation if render=True in __init__"""
        pass
    

    def close(self):
        """Close the simulation"""
        self.sim.close()
    

    def _extract_position(self, state_dict):
        """Extract pipette position from state dictionary."""
        robotId = list(sorted(state_dict.keys()))[0]
        robot_state = state_dict.get(robotId, {})
        position = np.array(
            robot_state.get('pipette_position', [0.0, 0.0, 0.0]),
            dtype=np.float32
        )
        return position


    def _normalize_position(self, position):
        """Normalize position from workspace bounds to [-1, 1]."""
        normalized = 2.0 * (position - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        return normalized.astype(np.float32)