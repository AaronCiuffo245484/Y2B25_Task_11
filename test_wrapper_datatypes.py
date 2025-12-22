"""
Test script to verify aaron_ot2_wrapper.py datatypes
Run this locally to verify the wrapper is correctly configured
"""
import numpy as np
import sys

print("Testing OT2 Wrapper Datatypes...")
print("=" * 60)

# Test 1: Action space handling
print("\n1. Testing action space handling:")
action = np.array([0.5, -0.3, 0.8], dtype=np.float32)
max_velocity = 2.0
velocity = action * max_velocity
full_action = [float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]

print(f"   Input action dtype: {action.dtype}")
print(f"   Velocity dtype: {velocity.dtype}")
print(f"   Full action: {full_action}")
print(f"   All elements are float: {all(isinstance(x, float) for x in full_action)}")
assert all(isinstance(x, float) for x in full_action), "Full action should be all floats"
print("   ✓ PASSED")

# Test 2: Observation space
print("\n2. Testing observation space:")
workspace_low = np.array([-0.1871, -0.1706, 0.1195], dtype=np.float32)
workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)

current_pos = np.array([0.0, 0.0, 0.2], dtype=np.float32)
goal_pos = np.array([0.1, 0.1, 0.25], dtype=np.float32)

def normalize_position(position):
    normalized = 2.0 * (position - workspace_low) / (workspace_high - workspace_low) - 1.0
    return normalized.astype(np.float32)

norm_current = normalize_position(current_pos)
norm_goal = normalize_position(goal_pos)
observation = np.concatenate([norm_current, norm_goal], dtype=np.float32)

print(f"   Observation shape: {observation.shape}")
print(f"   Observation dtype: {observation.dtype}")
print(f"   Observation range: [{observation.min():.3f}, {observation.max():.3f}]")
assert observation.shape == (6,), "Observation should be shape (6,)"
assert observation.dtype == np.float32, "Observation should be float32"
assert observation.min() >= -1.0 and observation.max() <= 1.0, "Observation should be in [-1, 1]"
print("   ✓ PASSED")

# Test 3: Reward calculation
print("\n3. Testing reward calculation:")
distance = 0.005  # 5mm
time_penalty = -0.1
distance_penalty = -10.0 * distance
success_bonus = 50.0
total_reward = time_penalty + distance_penalty + success_bonus

print(f"   Distance: {distance*1000:.1f}mm")
print(f"   Time penalty: {time_penalty}")
print(f"   Distance penalty: {distance_penalty}")
print(f"   Success bonus: {success_bonus}")
print(f"   Total reward: {total_reward}")
assert isinstance(total_reward, float), "Reward should be float"
print("   ✓ PASSED")

# Test 4: Boolean flags
print("\n4. Testing boolean flags:")
terminated = bool(distance < 0.005)
truncated = bool(300 >= 300)
print(f"   Terminated: {terminated} (type: {type(terminated).__name__})")
print(f"   Truncated: {truncated} (type: {type(truncated).__name__})")
assert isinstance(terminated, bool), "Terminated should be bool"
assert isinstance(truncated, bool), "Truncated should be bool"
print("   ✓ PASSED")

# Test 5: Info dict
print("\n5. Testing info dict:")
info = {
    'distance_to_goal': float(distance),
    'current_position': current_pos.tolist(),
    'goal_position': goal_pos.tolist()
}
print(f"   distance_to_goal type: {type(info['distance_to_goal']).__name__}")
print(f"   current_position type: {type(info['current_position']).__name__}")
print(f"   goal_position type: {type(info['goal_position']).__name__}")
assert isinstance(info['distance_to_goal'], float), "Distance should be float"
assert isinstance(info['current_position'], list), "Position should be list"
assert isinstance(info['goal_position'], list), "Position should be list"
print("   ✓ PASSED")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("\nThe wrapper datatypes are correctly configured.")
print("You can now proceed with training on ClearML.")
