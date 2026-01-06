from sim_class import Simulation
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

sim = Simulation(num_agents=1, render=True)

# Workspace bounds, which must match training exactly.
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

def dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def move_to_target_rl(sim, model, target_pos, max_steps=300, tolerance=0.001, 
                      velocity_threshold=0.04, deterministic=False):
    """
    Test RL controller - similar to my PID function but uses RL model.
    
    Args:
        sim: Simulation object
        model: Trained PPO model
        target_pos: Target [x, y, z] in meters
        max_steps: Maximum steps before timeout
        tolerance: Distance threshold (0.001 = 1mm)
        velocity_threshold: Max velocity for success
        deterministic: True/False for model prediction
    
    Returns:
        (success, distances, velocities, final_distance)
    """
    # Resetting simulation and getting starting position
    state = sim.reset(num_agents=1)
    robot_id = list(state.keys())[0]  # Get the actual robot ID dynamically
    current_pos = np.array(state[robot_id]["pipette_position"], dtype=np.float32)
    print(f"Starting position: {current_pos}")
    
    # Normalize target: Converting target to numpy array and normalize it to [-1, 1] range. (RL model was trained with normalized positions, so we must normalize here too).
    goal_position = np.array(target_pos, dtype=np.float32)
    goal_normalized = normalize_position(goal_position)
    
    distances = []
    velocities = []
    
    for step in range(max_steps):
        # Normalizing current position to [-1, 1]
        current_normalized = normalize_position(current_pos)
        
        # Building observation [current_x, current_y, current_z, goal_x, goal_y, goal_z]
        obs = np.concatenate([current_normalized, goal_normalized], dtype=np.float32)
        
        # Getting action from RL model. Model outputs normalized action in [-1, 1] range and deterministic parameter controls if model adds exploration noise.
        action, _states = model.predict(obs, deterministic=deterministic)
        
        # Scaling action to velocity. Model outputs [-1, 1], we convert to [-2, 2] m/s.
        velocity = scale_action_to_velocity(action)
        
        # Calculate metrics
        distance = np.linalg.norm(current_pos - goal_position)
        velocity_magnitude = np.linalg.norm(velocity)
        
        distances.append(distance)
        velocities.append(velocity_magnitude)
        
        # Checking if target is reached.
        if distance <= tolerance and velocity_magnitude <= velocity_threshold:
            print(f'Target reached at step {step}')
            print(f'Distance: {distance*1000:.3f}mm')
            print(f'Velocity: {velocity_magnitude:.4f}m/s')
            return True, distances, velocities, distance
        
        # Send velocity command (convert to list format)
        actions = [[float(velocity[0]), float(velocity[1]), float(velocity[2]), 0]]
        state = sim.run(actions, num_steps=1)
        # Getting new position for next iteration
        robot_id = list(state.keys())[0]  
        current_pos = np.array(state[robot_id]["pipette_position"], dtype=np.float32)
    
    # If timeout reached
    final_distance = np.linalg.norm(current_pos - goal_position)
    print(f'Max steps reached')
    print(f'Final distance: {final_distance*1000:.3f}mm')
    return False, distances, velocities, final_distance


print("Loading trained model...")
model = PPO.load(r"C:\Users\USER\Documents\GitHub\Y2B25_Task_11\models\260105.1425_yuliia_lr1e-4_b128_s2048_th1mm.zip") 

# Testting RL controller with same target as for PID controller.
target = [0.2, 0.2, 0.2]

print("\n=== Testing with deterministic=False ===")
success_false, dist_false, vel_false, final_false = move_to_target_rl(
    sim, model, target, deterministic=False
)

# Reset simulation
print("\n=== Resetting simulation for second test ===")
state = sim.reset(num_agents=1)

print("\n=== Testing with deterministic=True ===")
success_true, dist_true, vel_true, final_true = move_to_target_rl(
    sim, model, target, deterministic=True
)

# Plot comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Distance over time
ax1.plot(dist_false, label='RL (deterministic=False)', linewidth=2)
ax1.plot(dist_true, label='RL (deterministic=True)', linewidth=2)
ax1.axhline(y=0.001, color='r', linestyle='--', label='1mm tolerance')

settling_threshold = 0.002  
settling_steps_false = [i for i, d in enumerate(dist_false) if d < settling_threshold]
settling_steps_true = [i for i, d in enumerate(dist_true) if d < settling_threshold]

# Add shaded region to show settling phase
if settling_steps_false:
    ax1.axvspan(min(settling_steps_false), max(settling_steps_false), 
                alpha=0.2, color='blue', label='Settling phase (det=False)')
if settling_steps_true:
    ax1.axvspan(min(settling_steps_true), max(settling_steps_true), 
                alpha=0.2, color='orange', label='Settling phase (det=True)')

ax1.set_xlabel('Step')
ax1.set_ylabel('Distance (m)')
ax1.set_title('RL Controller: Distance to Target')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Velocity over time
ax2.plot(vel_false, label='RL (deterministic=False)', linewidth=2)
ax2.plot(vel_true, label='RL (deterministic=True)', linewidth=2)
ax2.axhline(y=0.04, color='r', linestyle='--', label='Velocity threshold')
ax2.set_xlabel('Step')
ax2.set_ylabel('Velocity (m/s)')
ax2.set_title('RL Controller: Velocity Magnitude')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rl_evaluation.png', dpi=150)
print("\n Plot saved: rl_evaluation.png")



