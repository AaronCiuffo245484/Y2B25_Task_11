from sim_class import Simulation
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Create simulation (render=False for speed, or render=True to watch)
sim = Simulation(num_agents=1, render=True)

# Workspace bounds
WORKSPACE_LOW = np.array([-0.1871, -0.1706, 0.1700], dtype=np.float32)
WORKSPACE_HIGH = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)

def normalize_position(position):
    """Normalize position from workspace bounds to [-1, 1]."""
    position = np.asarray(position, dtype=np.float32)
    normalized = 2.0 * (position - WORKSPACE_LOW) / (WORKSPACE_HIGH - WORKSPACE_LOW) - 1.0
    return normalized.astype(np.float32)

def scale_action_to_velocity(action, max_velocity=2.0):
    """Scale normalized action [-1, 1] to velocity commands."""
    action = np.asarray(action, dtype=np.float32)
    velocity = action * max_velocity
    return velocity.astype(np.float32)

def move_to_target_rl(sim, model, target_pos, max_steps=300, tolerance=0.0013):
    """
    Move to target using RL controller.
    Similar structure to PID move_to_target function.
    
    Returns:
        distances: List of distances at each step
        steps: Number of steps taken
        success: True if reached target
    """
    # Resetting simulation and getting starting position
    state = sim.reset(num_agents=1)
    robot_id = list(state.keys())[0]
    current_pos = np.array(state[robot_id]["pipette_position"], dtype=np.float32)
    
    # Normalize target: Converting target to numpy array and normalize it to [-1, 1] range. (RL model was trained with normalized positions, so we must normalize here too).
    goal_position = np.array(target_pos, dtype=np.float32)
    goal_normalized = normalize_position(goal_position)
    
    distances = []
    settle_count = 0  # To track settling 
    
    for step in range(max_steps):
        # Normalizing current position to [-1, 1]
        current_normalized = normalize_position(current_pos)
        
        # Building observation [current_x, current_y, current_z, goal_x, goal_y, goal_z]
        obs = np.concatenate([current_normalized, goal_normalized], dtype=np.float32)
        
        # Getting action from RL model. Model outputs normalized action in [-1, 1] range and deterministic parameter controls if model adds exploration noise.
        action, _states = model.predict(obs, deterministic=True)
        
        # Scaling action to velocity. Model outputs [-1, 1], we convert to [-2, 2] m/s.
        velocity = scale_action_to_velocity(action)
        
        # Calculate distance
        distance = np.linalg.norm(current_pos - goal_position)
        distances.append(distance)
        
        # Check settling (similar to my PID controller, 20 steps within tolerance to make sure it does not overshoot)
        if distance < tolerance:
            settle_count += 1
            if settle_count >= 20:
                print(f"  Target reached at step {step}")
                return distances, step, True
        else:
            settle_count = 0
        
        # Send velocity command (convert to list format)
        actions = [[float(velocity[0]), float(velocity[1]), float(velocity[2]), 0]]
        state = sim.run(actions, num_steps=1)
        
        # Getting new position for next iteration
        robot_id = list(state.keys())[0]
        current_pos = np.array(state[robot_id]["pipette_position"], dtype=np.float32)
    
    # If timeout
    print(f"  Max steps reached")
    return distances, max_steps, False


# Load trained model
model = PPO.load(r"C:\Users\USER\Documents\GitHub\Y2B25_Task_11\models\260105.1425_yuliia_lr1e-4_b128_s2048_th1mm.zip")
print("Model loaded")

# Generating 5 random targets with teh same seed as PID for truthful comparison.
np.random.seed(42)
targets = []
for i in range(5):
    target = [
        np.random.uniform(-0.18, 0.25),  # X range
        np.random.uniform(-0.17, 0.21),  # Y range
        np.random.uniform(0.17, 0.28)    # Z range
    ]
    targets.append(target)

print("\nGenerated 5 random targets:")
for i, target in enumerate(targets, 1):
    print(f"  Target {i}: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]")
print()

# Testing each target
all_results = []
colors = ['blue', 'green', 'orange', 'red', 'purple']

for i, target in enumerate(targets, 1):
    print(f"Moving to Target {i}: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]")
    distances, steps, success = move_to_target_rl(sim, model, target)
    all_results.append({
        'target': target,
        'distances': distances,
        'steps': steps,
        'success': success
    })
    print()

# Close simulation
sim.close()

# Plot all trajectories in the same format as PID plot
plt.figure(figsize=(12, 6))

for i, result in enumerate(all_results):
    target = result['target']
    label = f"Target {i+1}: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]"
    plt.plot(result['distances'], linewidth=2, color=colors[i], label=label)

plt.xlabel('Step', fontsize=12)
plt.ylabel('Distance to Target (m)', fontsize=12)
plt.title('RL Controller - Multiple Random Targets', fontsize=14, fontweight='bold')
plt.axhline(y=0.001, color='black', linestyle='--', linewidth=1.5, label='Tolerance (1mm)')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rl_multiple_targets.png', dpi=150, bbox_inches='tight')
print("Plot saved: rl_multiple_targets.png")
plt.close()

# Print summary
print("\nSummary:")
for i, result in enumerate(all_results, 1):
    status = "SUCCESS" if result['success'] else "TIMEOUT"
    final_error = result['distances'][-1]
    print(f"Target {i}: {result['steps']} steps, "
          f"Final error: {final_error:.6f} m ({final_error*1000:.3f} mm) - {status}")

# Additional statistics
total_success = sum(1 for r in all_results if r['success'])
print(f"\nOverall Success Rate: {total_success}/5 ({total_success/5*100:.0f}%)")

avg_steps = np.mean([r['steps'] for r in all_results])
print(f"Average steps: {avg_steps:.1f}")

avg_final_error = np.mean([r['distances'][-1] for r in all_results])
print(f"Average final error: {avg_final_error:.6f} m ({avg_final_error*1000:.3f} mm)")