"""
OT2 Gym Wrapper with Curriculum Learning
Automatically adapts difficulty based on agent performance

Author: Aaron Ciuffo
Course: ADS-AI Y2B Block B - Task 11
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    """
    Custom Gymnasium environment for OT-2 robot control with curriculum learning.
    
    The environment automatically adjusts the target threshold based on success rate:
    - Phase 1: 10mm threshold (easy)
    - Phase 2: 5mm threshold (medium)
    - Phase 3: 2mm threshold (hard)
    - Phase 4: 1mm threshold (final goal)
    
    Transitions to next phase when success rate > 80% over last 100 episodes.
    
    Parameters
    ----------
    render : bool
        Whether to render the simulation visually
    max_steps : int
        Maximum steps per episode before truncation
    target_threshold : float
        Final target threshold (1mm default), curriculum starts at 10x this
    curriculum : bool
        Whether to use curriculum learning (True recommended)
    """
    
    def __init__(self, render=False, max_steps=1000, 
                 target_threshold=0.001, curriculum=True):
        super(OT2Env, self).__init__()
        
        self.render_mode = render
        self.max_steps = max_steps
        self.final_threshold = target_threshold
        self.use_curriculum = curriculum
        
        # Curriculum learning setup
        if self.use_curriculum:
            # Start at 10x the final threshold (10mm if final is 1mm)
            self.current_threshold = self.final_threshold * 10.0
            self.curriculum_phase = 1
            self.phase_thresholds = [
                self.final_threshold * 10.0,  # Phase 1: 10mm
                self.final_threshold * 5.0,   # Phase 2: 5mm
                self.final_threshold * 2.0,   # Phase 3: 2mm
                self.final_threshold           # Phase 4: 1mm (final)
            ]
        else:
            self.current_threshold = target_threshold
            self.curriculum_phase = None
        
        # Success tracking for curriculum
        self.episode_count = 0
        self.success_count = 0
        self.success_history = []  # Track last 100 episodes
        
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
        
        # OT-2 workspace bounds
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        # Episode tracking
        self.steps = 0
        self.goal_position = None
    
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
        
        # Create observation
        observation = np.concatenate([current_pos, self.goal_position], dtype=np.float32)
        
        # Reset step counter
        self.steps = 0
        
        info = {}
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment."""
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
        
        # Terminated: goal reached (using current curriculum threshold)
        terminated = bool(distance_to_goal < self.current_threshold)
        
        # Truncated: max steps reached
        truncated = bool(self.steps >= self.max_steps)
        
        # Track success for curriculum
        if terminated:
            self._update_curriculum(success=True)
        elif truncated:
            self._update_curriculum(success=False)
        
        # Info for debugging and callback logging
        info = {
            'distance_to_goal': float(distance_to_goal),
            'current_position': current_pos.tolist(),
            'goal_position': self.goal_position.tolist(),
            'curriculum_threshold': float(self.current_threshold),
            'curriculum_phase': self.curriculum_phase if self.use_curriculum else 0
        }
        
        self.steps += 1
        
        return observation, reward, terminated, truncated, info
    
    def _update_curriculum(self, success):
        """
        Update curriculum based on success rate.
        Advances to next phase when success rate > 80% over last 100 episodes.
        """
        if not self.use_curriculum:
            return
        
        self.episode_count += 1
        self.success_history.append(1.0 if success else 0.0)
        
        # Keep only last 100 episodes
        if len(self.success_history) > 100:
            self.success_history.pop(0)
        
        # Check if we should advance curriculum (after at least 100 episodes)
        if len(self.success_history) >= 100:
            success_rate = np.mean(self.success_history)
            
            # Advance to next phase if success rate > 80%
            if success_rate > 0.80:
                next_phase = self.curriculum_phase
                
                # Move to next phase if not at final phase
                if next_phase < len(self.phase_thresholds):
                    self.curriculum_phase = next_phase + 1
                    self.current_threshold = self.phase_thresholds[self.curriculum_phase - 1]
                    
                    # Reset success tracking for new phase
                    self.success_history = []
                    
                    print(f"\n{'='*60}")
                    print(f"CURRICULUM ADVANCED!")
                    print(f"Phase {self.curriculum_phase}/{len(self.phase_thresholds)}")
                    print(f"New threshold: {self.current_threshold*1000:.2f} mm")
                    print(f"Previous success rate: {success_rate*100:.1f}%")
                    print(f"{'='*60}\n")
    
    def _calculate_reward(self, current_pos, distance_to_goal):
        """
        Reward with stability bonus for staying near goal.
        Uses current curriculum threshold for goal bonus.
        """
        max_dist = np.linalg.norm(self.workspace_high - self.workspace_low)
        reward = -(distance_to_goal / max_dist) ** 2 - 0.01
        
        # Large bonus for reaching current curriculum goal
        if distance_to_goal < self.current_threshold:
            reward += 100.0
        
        # Proximity bonus for getting close to current threshold
        # Scale proximity window to 10x current threshold
        proximity_threshold = self.current_threshold * 10.0
        if distance_to_goal < proximity_threshold:
            # Bonus scales with curriculum difficulty
            proximity_factor = 100.0 / self.current_threshold  # Higher bonus for smaller thresholds
            proximity_bonus = 10.0 * np.exp(-distance_to_goal * proximity_factor)
            reward += proximity_bonus
        
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
    
    def get_curriculum_info(self):
        """
        Get current curriculum status for logging/monitoring.
        
        Returns
        -------
        dict
            Dictionary with curriculum phase, threshold, and success rate
        """
        if not self.use_curriculum:
            return {
                'phase': 0,
                'threshold_mm': self.current_threshold * 1000,
                'success_rate': 0.0
            }
        
        success_rate = np.mean(self.success_history) if self.success_history else 0.0
        
        return {
            'phase': self.curriculum_phase,
            'max_phase': len(self.phase_thresholds),
            'threshold_mm': self.current_threshold * 1000,
            'success_rate': success_rate,
            'episodes_in_phase': len(self.success_history)
        }
