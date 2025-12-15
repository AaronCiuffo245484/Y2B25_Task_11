"""
OT2 RL Training Script with Custom Metrics Logging
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from clearml import Task
import argparse
from datetime import datetime
import numpy as np

# Import your wrapper
from aaron_ot2_gym_wrapper import OT2Env

# ============================================================================
# CONFIGURATION
# ============================================================================

# Generate timestamp for unique task name and model filename
timestamp = datetime.now().strftime("%y%m%d.%H%M")

# ============================================================================
# Custom Callback for OT2-Specific Metrics
# ============================================================================
class OT2Callback(BaseCallback):
    """
    Custom callback for logging OT2-specific metrics:
    - Episode reward (total reward per episode)
    - Episode length (steps to reach goal or timeout)
    - Success rate (% episodes reaching <1mm precision)
    - Final distance to goal
    """
    
    def __init__(self, threshold=0.001, verbose=0):
        super().__init__(verbose)
        self.threshold = threshold  # Success threshold in meters
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_final_distances = []
    
    def _on_step(self) -> bool:
        # Check if any environment finished an episode
        dones = self.locals.get('dones', [])
        
        for i, done in enumerate(dones):
            if done:
                # Get info for this environment
                infos = self.locals.get('infos', [])
                if i < len(infos):
                    info = infos[i]
                    
                    # Extract metrics from info dict
                    final_dist = info.get('distance_to_goal', float('inf'))
                    
                    # Episode ended - log metrics
                    # Note: episode_rewards and episode_lengths are tracked by SB3
                    ep_info = info.get('episode')
                    if ep_info is not None:
                        ep_reward = ep_info['r']
                        ep_length = ep_info['l']
                        
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
                        
                        # Success if final distance < threshold
                        success = float(final_dist < self.threshold)
                        self.episode_successes.append(success)
                        self.episode_final_distances.append(final_dist)
                        
                        # Log to tensorboard (will show in ClearML)
                        self.logger.record('ot2/episode_reward', ep_reward)
                        self.logger.record('ot2/episode_length', ep_length)
                        self.logger.record('ot2/final_distance_m', final_dist * 1000)  # Convert to mm
                        self.logger.record('ot2/success', success)
                        
                        # Rolling averages over last 100 episodes
                        if len(self.episode_successes) >= 10:
                            window = min(100, len(self.episode_successes))
                            self.logger.record('ot2/success_rate_100ep', 
                                             np.mean(self.episode_successes[-window:]))
                            self.logger.record('ot2/avg_length_100ep', 
                                             np.mean(self.episode_lengths[-window:]))
                            self.logger.record('ot2/avg_final_dist_mm_100ep', 
                                             np.mean(self.episode_final_distances[-window:]) * 1000)
        
        return True
    
    def _on_training_end(self) -> None:
        """Print summary statistics at end of training"""
        if len(self.episode_successes) > 0:
            print("\n" + "="*60)
            print("TRAINING SUMMARY")
            print("="*60)
            print(f"Total episodes: {len(self.episode_successes)}")
            print(f"Success rate: {100*np.mean(self.episode_successes):.1f}%")
            print(f"Average episode length: {np.mean(self.episode_lengths):.1f} steps")
            print(f"Average final distance: {1000*np.mean(self.episode_final_distances):.3f} mm")
            print(f"Successful episodes avg length: {np.mean([l for l, s in zip(self.episode_lengths, self.episode_successes) if s]):.1f} steps")
            print("="*60)


# ============================================================================
# ClearML Setup
# ============================================================================
task_name = f'OT2_RL_{PERSON_NAME}_{timestamp}'

task = Task.init(
    project_name='Mentor Group - Jason/Group 1', 
    task_name=task_name,
)

# Force HTTPS instead of SSH for GitHub
task.set_repo(
    repo='https://github.com/AaronCiuffo245484/Y2B25_Task_11.git',
    branch=BRANCH_NAME
)

task.set_base_docker('deanis/2023y2b-rl:latest')
task.set_packages(['tensorboard', 'clearml'])

# ============================================================================
# Command Line Arguments
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--total_timesteps", type=int, default=1000000)
args = parser.parse_args()

# Execute remotely AFTER capturing arguments
task.execute_remotely(queue_name='default')

# ============================================================================
# Generate Filename
# ============================================================================
def format_lr(lr):
    """Convert learning rate to scientific notation string for filename"""
    return f"{lr:.0e}".replace("+", "").replace("-0", "-")

lr_str = format_lr(args.learning_rate)
filename = f"{timestamp}_{PERSON_NAME}_lr{lr_str}_b{args.batch_size}_s{args.n_steps}"

print("="*60)
print(f"Training Configuration:")
print(f"  Person: {PERSON_NAME}")
print(f"  Learning Rate: {args.learning_rate}")
print(f"  Batch Size: {args.batch_size}")
print(f"  N Steps: {args.n_steps}")
print(f"  Total Timesteps: {args.total_timesteps:,}")
print(f"  Model Name: {filename}")
print("="*60)

# ============================================================================
# Environment Setup
# ============================================================================
env = OT2Env(render=False, max_steps=1000, target_threshold=0.001)

# ============================================================================
# Model Setup
# ============================================================================
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    verbose=1,
    tensorboard_log=f"runs/{PERSON_NAME}",
)

# ============================================================================
# Training with Custom Callback
# ============================================================================
ot2_callback = OT2Callback(threshold=0.001, verbose=1)

model.learn(
    total_timesteps=args.total_timesteps,
    callback=ot2_callback,
    tb_log_name=f"PPO_{filename}"
)

# ============================================================================
# Save and Upload Model
# ============================================================================
model_name = f"{filename}.zip"
model.save(model_name)
print(f"\nModel saved: {model_name}")

task.upload_artifact("model", artifact_object=model_name)
print(f"Artifact uploaded: {model_name}")

print("\nTraining complete!")

# Close environment safely
try:
    env.close()
except:
    pass