"""
OT2 RL Training Script
Train PPO agent to control OT-2 robot for precision positioning

Author: Aaron Ciuffo
Course: ADS-AI Y2B Block B - Task 11
"""
import gymnasium as gym
from stable_baselines3 import PPO
from clearml import Task
import argparse
from datetime import datetime
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# Import your wrapper
from aaron_ot2_gym_wrapper import OT2Env

# ============================================================================
# CONFIGURATION
# ============================================================================
PERSON_NAME = "aaron"
BRANCH_NAME = "aaron_new"
ENTRYPOINT = "aaron_train_ot2.py"

# Generate timestamp for unique task name and model filename
timestamp = datetime.now().strftime("%y%m%d.%H%M")

# ============================================================================
# ClearML Setup
# ============================================================================
task_name = f'OT2_RL_{PERSON_NAME}_{timestamp}'

task = Task.init(
    project_name='Mentor Group - Jason/Group 1', 
    task_name=task_name,
)

# Force HTTPS instead of SSH for GitHub (prevents authentication errors)
task.set_repo(
    repo='https://github.com/AaronCiuffo245484/Y2B25_Task_11.git',
    branch=BRANCH_NAME
)

task.set_base_docker('deanis/2023y2b-rl:latest')
# ============================================================================
# Command Line Arguments
# ============================================================================
parser = argparse.ArgumentParser()

# PPO hyperparameters (precision-oriented defaults)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--n_steps', type=int, default=8192)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--clip_range', type=float, default=0.15)
parser.add_argument('--ent_coef', type=float, default=0.0)

# Training control
parser.add_argument('--total_timesteps', type=int, default=1_000_000)

# Environment
parser.add_argument('--max_steps_truncate', type=int, default=1000)
parser.add_argument('--target_threshold', type=float, default=0.001)

args = parser.parse_args()

task.connect(vars(args))

# prevent weirdly sized minibatches
rollout_size = args.n_steps  # n_envs is 1 in this script

if args.batch_size > rollout_size:
    raise ValueError(f'batch_size ({args.batch_size}) must be <= n_steps ({args.n_steps})')

if rollout_size % args.batch_size != 0:
    raise ValueError(f'n_steps ({args.n_steps}) must be divisible by batch_size ({args.batch_size}) for clean PPO minibatches')


# Execute remotely AFTER capturing arguments
if Task.running_locally():
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
env = Monitor(OT2Env(render=False, max_steps=args.max_steps_truncate, target_threshold=args.target_threshold))
eval_env = Monitor(OT2Env(render=False, max_steps=args.max_steps_truncate, target_threshold=args.target_threshold))


# ============================================================================
# Model Setup
# ============================================================================
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=args.learning_rate,
    n_steps=args.n_steps,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    clip_range=args.clip_range,
    ent_coef=args.ent_coef,
    tensorboard_log='runs',
    verbose=1,
)

# ============================================================================
# Training with Custom Callback
# ============================================================================

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='models/best',
    log_path='models/eval',
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    render=False
)
model.learn(
    total_timesteps=args.total_timesteps,
    callback=eval_callback,
    tb_log_name=f'PPO_{filename}',
)

# ============================================================================
# Save and Upload Model
# ============================================================================
model_name = f"{filename}.zip"
model.save(model_name)
print(f"\nModel saved: {model_name}")

task.upload_artifact("model", artifact_object=Path(model_name).resolve())
print(f"Artifact uploaded: {model_name}")

print("\nTraining complete!")

# Close environment safely
try:
    env.close()
except:
    pass