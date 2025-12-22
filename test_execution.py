"""
Quick Test - Verify ClearML Job Will Execute
Run this locally to catch issues before remote training

This tests:
1. All imports work
2. Wrapper initializes correctly
3. PPO model can be created
4. Environment steps work
5. Training loop runs
6. Model can be saved
"""
import sys
import numpy as np

print("="*60)
print("OT2 RL EXECUTION TEST")
print("="*60)

# Test 1: Imports
print("\n[1/7] Testing imports...")
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from datetime import datetime
    print("  ✓ All imports successful")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Import wrapper
print("\n[2/7] Testing wrapper import...")
try:
    from aaron_ot2_wrapper import OT2Env
    print("  ✓ Wrapper import successful")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    print("  Make sure aaron_ot2_wrapper.py is in the same directory")
    sys.exit(1)

# Test 3: Initialize environment
print("\n[3/7] Testing environment initialization...")
try:
    env = OT2Env(render=False, max_steps=300, target_threshold=0.005)
    print(f"  ✓ Environment created")
    print(f"    Action space: {env.action_space}")
    print(f"    Observation space: {env.observation_space}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    print("  Check that sim_class.py is available")
    sys.exit(1)

# Test 4: Reset environment
print("\n[4/7] Testing environment reset...")
try:
    obs, info = env.reset()
    print(f"  ✓ Reset successful")
    print(f"    Observation shape: {obs.shape}")
    print(f"    Observation dtype: {obs.dtype}")
    print(f"    Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Environment step
print("\n[5/7] Testing environment step...")
try:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  ✓ Step successful")
    print(f"    Reward: {reward:.4f}")
    print(f"    Distance to goal: {info['distance_to_goal']*1000:.2f}mm")
    print(f"    Terminated: {terminated}")
    print(f"    Truncated: {truncated}")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Create PPO model
print("\n[6/7] Testing PPO model creation...")
try:
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=0.0003,
        batch_size=128,
        n_steps=2048,
        verbose=0
    )
    print("  ✓ PPO model created successfully")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Run minimal training
print("\n[7/7] Testing training loop (100 steps)...")
try:
    model.learn(total_timesteps=100, progress_bar=False)
    print("  ✓ Training loop successful")
    
    # Test model save
    model.save("test_model")
    print("  ✓ Model save successful")
    
    # Clean up
    import os
    if os.path.exists("test_model.zip"):
        os.remove("test_model.zip")
        print("  ✓ Cleanup complete")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Close environment
try:
    env.close()
except:
    pass

print("\n" + "="*60)
print("ALL TESTS PASSED ✓")
print("="*60)
print("\nYour setup is ready for ClearML remote execution.")
print("\nNext steps:")
print("1. Update BRANCH_NAME in aaron_train_ot2.py")
print("2. Run: python aaron_train_ot2.py")
print("3. Monitor training in ClearML dashboard")
print("="*60)
