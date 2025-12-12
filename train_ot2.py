"""
OT2 RL Training Script
"""
import gymnasium as gym
from stable_baselines3 import PPO
from clearml import Task
import argparse
from pathlib import Path
from datetime import datetime

# Import your wrapper
from ot2_gym_wrapper import OT2Env

# ============================================================================
# CONFIGURATION - UPDATE THESE
# ============================================================================
PERSON_NAME = "aaron_ciuffo"  # UPDATE WITH YOUR NAME
BRANCH_NAME = "aaron_branch"   # UPDATE WITH YOUR BRANCH NAME
OT2_WRAPPER = "aaron_ot2_gym_wrapper.py"

# ============================================================================
# ClearML Setup - DO NOT CHANGE THIS!
# ============================================================================
task = Task.init(
    project_name='Mentor Group - Jason/Group 1', 
    task_name='OT2_RL_Training',
)

task.set_base_docker('deanis/2023y2b-rl:latest')

task.set_script(
    repository='https://github.com/AaronCiuffo245484/Y2B25_Task_11.git',
    branch=BRANCH_NAME,
    working_dir='.',
    entry_point=OT2_WRAPPER,  # UPDATE WITH YOUR FILENAME
)

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

timestamp = datetime.now().strftime("%y%m%d.%H%M")
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

# Save model
model_dir = Path("models") / PERSON_NAME
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / filename
model.save(model_path)
print(f"\nModel saved: {model_path}")

# Upload to ClearML
task.upload_artifact(name='final_model', artifact_object=str(model_path) + '.zip')

print("\nTraining complete!")
env.close()