# OT2 RL Training - Getting Started Guide


## Key Concepts

- ClearML makes it easy to test different hyper parameters on a remote server
- All of your work needs to reside on a git branch within the repo
- Make sure you commit your changes before starting a job!

## Prerequisites

1. **ClearML Setup (One-Time)**
   - Follow the setup instructions here: [Remote Training with ClearML](https://adsai.buas.nl/Study%20Content/Robotics%20and%20Reinforcement%20Learning/14.%20Remote%20RL%20Training.html#remote-training-with-clearml)
   - This includes installing ClearML and configuring credentials
   - Only needs to be done once per environment

2. **VPN Connection Required**
   - You MUST be connected to the BUAS VPN to access the ClearML server and GPU queue
   - Connect to VPN before running any training jobs

3. **Repository Setup**
   - Clone the repository: `git clone https://github.com/AaronCiuffo245484/Y2B25_Task_11`
   - Create your own branch: `git checkout -b your_name_branch`
   - Activate your y2b25 python environment
   - Install local dependencies: `pip install gymnasium stable-baselines3 pybullet numpy`

4. **Required Files in Your Branch**
   - `your_name_train_ot2.py` - Your training script
   - `your_name_ot2_gym_wrapper.py` - Your gym wrapper
   - `sim_class.py` - Simulation class (provided)
   - `requirements.txt` - Dependencies for remote server

## Setup Your Training Script

### 1. Copy and Rename the Template

```bash
cp train_ot2.py your_name_train_ot2.py
```

### 2. Update Configuration (Lines 15-17, 31)

Edit these lines in your training script:

```python
# Line 17
PERSON_NAME = "your_name"  # e.g., "jane_doe"

# Line 18  
BRANCH_NAME = "your_branch"  # e.g., "jane_branch"

# Line 19
entry_point='your_name_train_ot2.py',  # Match your filename
```

### 3. Copy and Rename the Example Wrapper 


```bash
cp example_ot2_gym_wrapper.py your_name_ot2_gym_wrapper.py
```

**Alternatively:** Copy your own wrapper into the repo. Make sure the name matches the convention `your_name_ot2_gym_wrapper.py`

Alter the import at the top of your `your_name_train_ot2.py` to match the name of your wrapper:

```Python
from aaron_ot2_gym_wrapper import OT2Env
```

### 4. Commit and Push

```bash
git add your_name_train_ot2.py your_name_ot2_gym_wrapper.py requirements.txt sim_class.py
git commit -m "Add training setup"
git push origin your_branch
```

## Update Your Wrapper

### Modify the Reward Function (Optional)

The key method to experiment with is `_calculate_reward()` around line 145:

```python
def _calculate_reward(self, current_pos, distance_to_goal):
    """
    MODIFY THIS METHOD to experiment with different reward functions.
    """
    # Simple negative distance reward
    reward = float(-distance_to_goal)
    
    # Try these alternatives:
    # Sparse reward:
    # reward = 1.0 if distance_to_goal < 0.01 else 0.0
    
    # Shaped reward with step penalty:
    # reward = -distance_to_goal - 0.001
    
    # Exponential reward:
    # reward = -np.exp(distance_to_goal)
    
    return reward
```

### Test Locally (Optional but Recommended)

```python
from stable_baselines3.common.env_checker import check_env
from your_name_ot2_gym_wrapper import OT2Env

env = OT2Env()
check_env(env)  # Verifies gym compliance
print("Wrapper passed checks!")
```

## Running Training Jobs

### Fast Test (~1 minute)
```bash
python your_name_train_ot2.py --total_timesteps 512 --n_steps 256 --batch_size 32
```

### Quick Test (~5 minutes)
```bash
python your_name_train_ot2.py --total_timesteps 5000 --n_steps 256 --batch_size 32
```

### Standard Training (30-60 minutes)
```bash
python your_name_train_ot2.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --total_timesteps 500000
```

### Full Training (2-4 hours)
```bash
python your_name_train_ot2.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --total_timesteps 2000000
```

### Experiment with Hyperparameters
```bash
# Higher learning rate
python your_name_train_ot2.py --learning_rate 0.001 --total_timesteps 100000

# Larger batch
python your_name_train_ot2.py --batch_size 128 --n_steps 4096 --total_timesteps 100000

# Quick test with different settings
python your_name_train_ot2.py --learning_rate 0.0001 --batch_size 32 --n_steps 512 --total_timesteps 10000
```

## Monitoring Your Training

### Access ClearML Dashboard
1. URL: http://194.171.191.227:8080
2. Login: Use your Coder credentials
3. Project: Mentor Group - Jason/Group 1

### What to Check
- **Console tab**: See training logs and progress
- **Scalars tab**: View reward curves and loss plots
- **Configuration tab**: Verify hyperparameters are correct
- **Artifacts tab**: Download saved models

## Model Naming Convention

Models are automatically saved with this format:
```
YYMMDD.HHMM_yourname_lrXeY_bZ_sN
```

Example: `241212.1430_aaron_ciuffo_lr3e-4_b64_s2048`

Where:
- `YYMMDD.HHMM`: Timestamp when training started
- `yourname`: From PERSON_NAME in your script
- `lr3e-4`: Learning rate in scientific notation
- `b64`: Batch size
- `s2048`: Number of steps

## File Structure

```
your_repo/
├── train_ot2.py                    # Template training script
├── your_name_train_ot2.py          # Your training script
├── example_ot2_gym_wrapper.py      # Example wrapper (reference)
├── your_name_ot2_gym_wrapper.py    # Your gym wrapper
├── sim_class.py                    # Simulation class (provided)
├── requirements.txt                # Remote dependencies
└── models/
    └── your_name/                  # Saved models (created during training)
```

## Required Files: requirements.txt

Create this file in your repository root:

```
clearml
tensorboard
```

## Troubleshooting

### "No module named 'clearml'"
- Check that `requirements.txt` exists in your repo root
- Verify it contains `clearml` and `tensorboard`
- Commit and push requirements.txt

### "Cannot connect to ClearML server"
- Verify you're connected to BUAS VPN
- Check VPN connection is active

### Arguments not updating in ClearML
- Verify argparse is BEFORE `task.execute_remotely()`
- Check the Configuration tab in ClearML dashboard
- The values should match what you passed on command line

### Job fails immediately
- Check ClearML Console tab for error message
- Common issues:
  - Wrapper file not in repository
  - BRANCH_NAME doesn't match actual branch
  - entry_point doesn't match filename
  - Missing sim_class.py file

### "ModuleNotFoundError: No module named 'sim_class'"
- Verify `sim_class.py` is committed to your branch
- Check that file exists in repository root

### "ModuleNotFoundError: No module named 'your_name_ot2_gym_wrapper'"
- Verify you updated the import in your training script
- Check that your wrapper file is committed
- Ensure the filename in the import matches your actual file

### Model file not found
- Check that `models/` directory was created
- Verify PERSON_NAME matches in your script
- Look in ClearML Artifacts tab for uploaded model

## Tips for Hyperparameter Search

### What to Experiment With

1. **Learning Rate** (--learning_rate)
   - Try: 0.0001, 0.0003, 0.001, 0.003
   - Lower = slower but more stable
   - Higher = faster but might be unstable

2. **Batch Size** (--batch_size)
   - Try: 32, 64, 128, 256
   - Smaller = noisier gradients, more updates
   - Larger = smoother gradients, fewer updates

3. **N Steps** (--n_steps)
   - Try: 512, 1024, 2048, 4096
   - More steps = more data per update
   - Must be divisible by batch_size

4. **Reward Function** (in wrapper)
   - Modify `_calculate_reward()` method
   - Test sparse vs dense rewards
   - Add penalties or bonuses

### Coordination with Team

- Use unique PERSON_NAME for each team member
- Check ClearML dashboard before starting long jobs
- Share successful hyperparameters with team
- Document experiments and results

## Example Workflow

```bash
# 1. Setup (one time)
git checkout -b aaron_branch
cp train_ot2.py aaron_train_ot2.py
# Edit aaron_train_ot2.py (update PERSON_NAME, BRANCH_NAME, entry_point)

# 2. Create wrapper
cp example_ot2_gym_wrapper.py aaron_ot2_gym_wrapper.py
# Edit aaron_train_ot2.py to import from aaron_ot2_gym_wrapper
# Optionally modify _calculate_reward() in wrapper

# 3. Commit
git add aaron_train_ot2.py aaron_ot2_gym_wrapper.py requirements.txt sim_class.py
git commit -m "Initial setup"
git push origin aaron_branch

# 4. Connect to VPN
# Connect to BUAS VPN now!

# 5. Quick test
python aaron_train_ot2.py --total_timesteps 512 --n_steps 256 --batch_size 32

# 6. Check ClearML dashboard
# Verify job is running correctly

# 7. Run full training
python aaron_train_ot2.py --total_timesteps 1000000

# 8. Monitor and iterate
# Check results, modify hyperparameters or reward function, repeat
```

## Getting Help

- Check ClearML Console tab for error messages
- Verify VPN connection
- Review this README
- Ask team members who have successful runs
- Check that all files are committed and pushed