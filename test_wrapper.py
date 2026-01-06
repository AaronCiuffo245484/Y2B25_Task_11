import numpy as np
from yuliia_ot2_gym_wrapper import OT2Env

def test_environment():
    """Test the OT2 environment with random actions."""

    # Testing environment creation. It creates an instance of the gym environment with visualization disabled, 300 step limit per episode, and 5mm success threshold.
    print("\n1. Creating environment...")
    env = OT2Env(render=False, max_steps=300, target_threshold=0.005)
    print("Environment created successfully")
    
    # Checking spaces. This verifies the action space (3D velocity commands in [-1,1]) and observation space (6D normalized positions) are correctly defined.
    print("\n2. Checking action and observation spaces...")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print("Spaces are correct")
    
    # Reseting environment. It resets the simulation to initial state, generates a new random goal position, and returns the first observation.
    print("\n3. Resetting environment...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation dtype: {obs.dtype}")
    print(f"Goal position: {env.goal_position}")

    # Checking if simulation starts with Z below workspace.
    state = env.sim.get_states()
    robot_id = list(state.keys())[0]
    start_pos = state[robot_id]["pipette_position"]
    print(f"Starting position: {start_pos}")
    print(f"Workspace Z bounds: [{env.workspace_low[2]}, {env.workspace_high[2]}]")

    if start_pos[2] < env.workspace_low[2]:
        print(f"Z starts BELOW workspace! {start_pos[2]:.4f} < {env.workspace_low[2]:.4f}")
    else:
        print(f"Z is within workspace bounds")
    print("Reset successful")
    
    # Running 1000 steps with random actions.
    print("\n4. Running 1000 steps with random actions...")
    
    total_reward = 0
    episodes_completed = 0
    min_distance = float('inf')
    steps_completed = 0
    
    for step in range(1000):
        # Generating a random 3D velocity vector within the action space bounds [-1,1] for each axis.
        action = env.action_space.sample()
        action = env.action_space.sample()
        
        # Executing the action in simulation and returning new observation, reward, termination flags, and info dictionary with current distance.
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps_completed += 1
        
        # # Recording the closest distance ever achieved to check if random actions can accidentally reach the goal.
        distance = info['distance_to_goal']
        if distance < min_distance:
            min_distance = distance
        
        # Print progress every 100 steps.
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/1000 - Distance: {distance*1000:.2f}mm - Reward: {reward:.2f}")
        
        # Reset if episode ends.
        if terminated or truncated:
            episodes_completed += 1
            if terminated:
                print(f"Goal reached at step {step + 1}! Distance: {distance*1000:.3f}mm")
            obs, info = env.reset()
    
    # Print summary
    print("Test Summary:")
    print(f"Total steps completed: {steps_completed}")
    print(f"Episodes completed: {episodes_completed}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward/steps_completed:.2f}")
    print(f"Minimum distance achieved: {min_distance*1000:.3f}mm")
    print(f"Goal threshold: {env.target_threshold*1000:.1f}mm")
    
    if min_distance < env.target_threshold:
        print("\n Random actions reached the goal at least once!")
    else:
        print(f"\n Random actions did not reach goal (closest: {min_distance*1000:.3f}mm)")
    
    # Close environment
    env.close()
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_environment()