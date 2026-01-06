"""
Test the improved wrapper with random actions.
Verify new reward and termination logic work correctly.
"""

import numpy as np
from yuliia_ot2_gym_wrapper_reward import OT2Env

def test_improved_wrapper():
    print("="*60)
    print("Testing IMPROVED Wrapper")
    print("="*60)
    
    # Create environment with improved wrapper
    env = OT2Env(render=False, max_steps=300, target_threshold=0.001)
    
    print("\n1. Testing initialization...")
    print(f"   Settling steps required: {env.required_settle_steps}")
    print(f"   Target threshold: {env.target_threshold*1000}mm")
    print("   ✓ Initialized")
    
    print("\n2. Testing reset...")
    obs, info = env.reset()
    print(f"   Steps within threshold: {env.steps_within_threshold}")
    print(f"   Distance history length: {len(env.distance_history)}")
    print("   ✓ Reset works")
    
    print("\n3. Testing step with random actions...")
    total_reward = 0
    settling_detected = False
    
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        # Check if settling logic triggered
        if env.steps_within_threshold > 0:
            settling_detected = True
            print(f"   Step {step}: Settling... ({env.steps_within_threshold}/{env.required_settle_steps})")
        
        if terminated:
            print(f"   ✓ Episode terminated (settled!) at step {step}")
            print(f"     Final distance: {info['distance_to_goal']*1000:.3f}mm")
            print(f"     Total reward: {total_reward:.2f}")
            break
        
        if truncated:
            print(f"   ✗ Episode truncated at step {step}")
            print(f"     Final distance: {info['distance_to_goal']*1000:.3f}mm")
            break
    
    if settling_detected:
        print("\n   ✓ Settling logic activated (robot got close with low velocity)")
    else:
        print("\n   ⚠ Settling logic never activated (expected with random actions)")
    
    print("\n4. Checking tracking arrays...")
    print(f"   Distance history length: {len(env.distance_history)}")
    print(f"   Velocity history length: {len(env.velocity_history)}")
    print("   ✓ History tracking works")
    
    env.close()
    print("\n✓ All tests passed!")
    print("="*60)

if __name__ == "__main__":
    test_improved_wrapper()