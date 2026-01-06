"""
Test settling behavior with controlled movements.
Manually move robot to target to verify settling detection.
"""

import numpy as np
from yuliia_ot2_gym_wrapper_reward import OT2Env

def test_settling_detection():
    print("\n" + "="*60)
    print("Testing Settling Detection")
    print("="*60)
    
    env = OT2Env(render=False, max_steps=300, target_threshold=0.001)
    obs, info = env.reset()
    
    print(f"\nGoal position: {env.goal_position}")
    
    # Get current position
    state = env.sim.get_states()
    robot_id = list(state.keys())[0]
    current_pos = np.array(state[robot_id]["pipette_position"])
    print(f"Start position: {current_pos}")
    
    # Calculate direction to goal
    direction = env.goal_position - current_pos
    distance = np.linalg.norm(direction)
    print(f"Initial distance: {distance*1000:.2f}mm")
    
    print("\n--- Phase 1: Fast approach ---")
    # Move fast toward goal
    for step in range(100):
        direction = env.goal_position - current_pos
        distance = np.linalg.norm(direction)
        
        if distance < 0.010:  # Within 10mm, slow down
            break
        
        # Fast movement: action = 0.5 (scaled to 1.0 m/s)
        action = direction / np.linalg.norm(direction) * 0.5
        action = np.clip(action, -1, 1)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        state = env.sim.get_states()
        robot_id = list(state.keys())[0]
        current_pos = np.array(state[robot_id]["pipette_position"])
        
        if step % 10 == 0:
            print(f"  Step {step}: distance={distance*1000:.2f}mm, reward={reward:.2f}")
    
    print(f"\n--- Phase 2: Precise settling ---")
    print("Now moving slowly to trigger settling...")
    
    # Move slowly toward goal to trigger settling
    for step in range(100, 200):
        direction = env.goal_position - current_pos
        distance = np.linalg.norm(direction)
        
        # Very slow movement: action = 0.05 (scaled to 0.1 m/s)
        action = direction / np.linalg.norm(direction) * 0.05
        action = np.clip(action, -1, 1)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        state = env.sim.get_states()
        robot_id = list(state.keys())[0]
        current_pos = np.array(state[robot_id]["pipette_position"])
        
        print(f"  Step {step}: distance={distance*1000:.3f}mm, "
              f"settling={env.steps_within_threshold}/{env.required_settle_steps}, "
              f"reward={reward:.2f}")
        
        if terminated:
            print(f"\n✓ SUCCESS: Settled at step {step}!")
            print(f"  Final distance: {distance*1000:.3f}mm")
            break
        
        if truncated:
            print(f"\n✗ TIMEOUT at step {step}")
            break
    
    env.close()
    print("="*60)

if __name__ == "__main__":
    test_settling_detection()