"""
Quick Imports Test - Run this first if you don't have sim_class.py locally
Tests only that all Python packages are installed correctly
"""
import sys

print("="*60)
print("IMPORTS-ONLY TEST (No Simulation Required)")
print("="*60)

print("\n[1/5] Testing gymnasium...")
try:
    import gymnasium as gym
    print(f"  ✓ gymnasium {gym.__version__}")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    print("  Install: pip install gymnasium")
    sys.exit(1)

print("\n[2/5] Testing stable-baselines3...")
try:
    import stable_baselines3
    from stable_baselines3 import PPO
    print(f"  ✓ stable-baselines3 {stable_baselines3.__version__}")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    print("  Install: pip install stable-baselines3")
    sys.exit(1)

print("\n[3/5] Testing numpy...")
try:
    import numpy as np
    print(f"  ✓ numpy {np.__version__}")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    print("  Install: pip install numpy")
    sys.exit(1)

print("\n[4/5] Testing tensorboard...")
try:
    import tensorboard
    print(f"  ✓ tensorboard {tensorboard.__version__}")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    print("  Install: pip install tensorboard")
    sys.exit(1)

print("\n[5/5] Testing clearml...")
try:
    import clearml
    from clearml import Task
    print(f"  ✓ clearml {clearml.__version__}")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    print("  Install: pip install clearml")
    sys.exit(1)

print("\n" + "="*60)
print("ALL PACKAGE IMPORTS SUCCESSFUL ✓")
print("="*60)
print("\nNext: Run test_execution.py to test full setup")
print("(Requires sim_class.py to be available)")
print("="*60)
