"""
OT2 RL Training Script
"""
import gymnasium as gym
from stable_baselines3 import PPO
from clearml import Task
import argparse
from pathlib import Path
from datetime import datetime
import os

# Import your wrapper - UPDATE THIS TO MATCH YOUR WRAPPER FILENAME
from your_name_ot2_gym_wrapper import OT2Env  # e.g., from aaron_ot2_gym_wrapper import OT2Env

# ============================================================================
# CONFIGURATION - UPDATE THESE
# ============================================================================
PERSON_NAME = "your_name"  # UPDATE WITH YOUR NAME (e.g., "aaron")
BRANCH_NAME = "your_branch"   # UPDATE WITH YOUR BRANCH NAME (e.g., "aaron_branch")
ENTRYPOINT = "your_name_train_ot2.py"  # UPDATE WITH THIS FILENAME (e.g., "aaron_train_ot2.py")

# Generate timestamp for unique task name and model filename
timestamp = datetime.now().strftime("%y%m%d.%H%M")

# ============================================================================
# ClearML Setup - DO NOT CHANGE BELOW THIS LINE
# ============================================================================
# Create unique task name
task_name = f'OT2_RL_{PERSON_NAME}_{timestamp}'

task = Task.init(
    project_name='Mentor Group - Jason/Group 1', 
    task_name=task_name,
)

task.set_base_docker('deanis/2023y2b-rl:latest')

task.set_script(
    repository='https://github.com/AaronCiuffo245484/Y2B25_Task_11.git',
    branch=BRANCH_NAME,
    working_dir='.',
    entry_point=ENTRYPOINT,
)

# Force tensorboard and clearml to install
task.set_packages(['tensorboard', 'clearml'])

# ============================================================================
# Command Line Arguments - BEFORE execute_remotely()
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
# Generate Filename with Scientific Notation
# ============================================================================
def format_lr(lr):
    """Convert learning rate to scientific notation string for filename"""
    return f"{lr:.0e}".replace("+", "").replace("-0", "-")

# Use timestamp from above and generate model filename (NO .zip extension)
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
env = OT2Env(render=False, max_steps=1000, target_threshold=0.002)

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
# Training
# ============================================================================
model.learn(
    total_timesteps=args.total_timesteps,
    tb_log_name=f"PPO_{filename}"
)

# ============================================================================
# Save and Upload Model
# ============================================================================
# Save model with .zip extension
model_name = f"{filename}.zip"
model.save(model_name)
print(f"\nModel saved: {model_name}")

# Upload artifact with .zip extension
task.upload_artifact("model", artifact_object=model_name)
print(f"Artifact uploaded: {model_name}")

print("\nTraining complete!")

# Close environment safely
try:
    env.close()
except:
    pass  # Already disconnected