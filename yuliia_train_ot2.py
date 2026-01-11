import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from clearml import Task
import argparse
from datetime import datetime
import numpy as np

from yuliia_ot2_gym_wrapper_reward import OT2Env

PERSON_NAME = "yuliia"
BRANCH_NAME = "yuliia"  

timestamp = datetime.now().strftime("%y%m%d.%H%M")


# Custom Callback for OT2 Metrics

class OT2Callback(BaseCallback):
    """
    Callback for logging OT2-specific metrics during training.

    This callback tracks:
    - Success rate (% of episodes reaching goal)
    - Final distances (how close did it get)
    - Episode lengths (how many steps to complete)
    - Episode rewards (cumulative reward per episode)
    """
    
    def __init__(self, threshold=0.001, verbose=0):
        super().__init__(verbose)
        self.threshold = threshold
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_final_distances = []
    

    def _on_step(self) -> bool:
        """Called after each step in all environments"""
        # Checks which episodes finished.
        dones = self.locals.get('dones', [])
        
        for i, done in enumerate(dones):
            if done:
                infos = self.locals.get('infos', [])
                if i < len(infos):
                    info = infos[i]
                    
                    # Extracting metrics
                    final_dist = info.get('distance_to_goal', float('inf'))
                    
                    # Getting episode info from SB3
                    ep_info = info.get('episode')
                    if ep_info is not None:
                        ep_reward = ep_info['r']
                        ep_length = ep_info['l']
                        
                        # Store metrics
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
                        
                        success = float(final_dist < self.threshold)
                        self.episode_successes.append(success)
                        self.episode_final_distances.append(final_dist)
                        
                        # Logging to tensorboard
                        self.logger.record('ot2/episode_reward', ep_reward)
                        self.logger.record('ot2/episode_length', ep_length)
                        self.logger.record('ot2/final_distance_mm', final_dist * 1000)
                        self.logger.record('ot2/success', success)
                        
                        # Rolling averages
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
        """
        Called once when training finishes.
        Prints a summary of overall training performance.
        """
        if len(self.episode_successes) > 0:
            print(f"Total episodes: {len(self.episode_successes)}")
            print(f"Success rate: {100*np.mean(self.episode_successes):.1f}%")
            print(f"Average episode length: {np.mean(self.episode_lengths):.1f} steps")
            print(f"Average final distance: {1000*np.mean(self.episode_final_distances):.3f} mm")
            
            # Calculate stats for only successful episodes. This shows how many steps it takes when the agent DOES succeed.
            successful_lengths = [l for l, s in zip(self.episode_lengths, self.episode_successes) if s]
            if successful_lengths:
                print(f"Successful episodes avg length: {np.mean(successful_lengths):.1f} steps")
            



# CLEARML SETUP

task_name = f'OT2_RL_{PERSON_NAME}_{timestamp}'

# Initializing ClearML task. 
task = Task.init(
    project_name='Mentor Group - Jason/Group 1', 
    task_name=task_name,
)

# Link to git repository for code version control.
task.set_repo(
    repo='https://github.com/AaronCiuffo245484/Y2B25_Task_11.git',
    branch=BRANCH_NAME
)

# Set Docker image to use on remote server. This ensures the remote machine has all necessary dependencies.
task.set_base_docker('deanis/2023y2b-rl:latest')

# Install tensorboard and clearml. 
task.set_packages(['tensorboard', 'clearml'])



# COMMAND LINE ARGUMENTS: HYPERPARAMETERS

# argparse allows running the script with different settings without editing code.
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--total_timesteps", type=int, default=500000)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--max_steps_truncate", type=int, default=250)
parser.add_argument("--target_threshold", type=float, default=0.001)
args = parser.parse_args()

# Execute remotely
task.execute_remotely(queue_name='default')



# GENERATING FILENAME

def format_lr(lr):
    """Convert learning rate to scientific notation for compact filename."""
    return f"{lr:.0e}".replace("+", "").replace("-0", "-")

# Build descriptive filename with all important hyperparameters. his makes it easy to identify what settings were used for each model.
lr_str = format_lr(args.learning_rate)
filename = f"{timestamp}_{PERSON_NAME}_lr{lr_str}_b{args.batch_size}_s{args.n_steps}_th{int(args.target_threshold*1000)}mm"


print(f"Training Configuration:")
print(f"  Person: {PERSON_NAME}")
print(f"  Learning Rate: {args.learning_rate}")
print(f"  Batch Size: {args.batch_size}")
print(f"  N Steps: {args.n_steps}")
print(f"  Total Timesteps: {args.total_timesteps:,}")
print(f"  Max Episode Steps: {args.max_steps_truncate}")
print(f"  Target Threshold: {args.target_threshold*1000:.1f}mm")
print(f"  Model Name: {filename}")



# ENVIRONMENT SETUP

# Creating the gym environment with our custom wrapper. This is what the RL agent will interact with during training.
env = OT2Env(
    render=False, 
    max_steps=args.max_steps_truncate, 
    target_threshold=args.target_threshold
)


# MODEL SETUP

model = PPO(
    'MlpPolicy',    # Multi-Layer Perceptron policy (standard feedforward neural network)
    env,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=10,
    gamma=args.gamma,
    gae_lambda=0.95,    # Generalized Advantage Estimation parameter (variance reduction)
    clip_range=0.2,     # PPO clipping parameter (prevents too large policy updates)
    verbose=1,
    tensorboard_log=f"runs/{PERSON_NAME}"   
)


# TRAINING

# Creating callback instance to track our custom metrics.
ot2_callback = OT2Callback(threshold=args.target_threshold, verbose=1)

# This runs for total_timesteps and calls our callback.
model.learn(
    total_timesteps=args.total_timesteps,
    callback=ot2_callback,
    tb_log_name=f"PPO_{filename}"
)


# SAVE AND UPLOAD MODEL

model_name = f"{filename}.zip"
model.save(model_name)
print(f"\nModel saved: {model_name}")

task.upload_artifact("model", artifact_object=model_name)
print(f"Artifact uploaded: {model_name}")

print("\nTraining complete!")

# Close environment
try:
    env.close()
except:
    pass
