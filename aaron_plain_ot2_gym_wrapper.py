import numpy as np
import gymnasium as gym
from gymnasium import spaces

class OT2Env(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, sim, max_steps=1000, target_threshold=0.01, render=False):
        super().__init__()
        self.sim = sim
        self.max_steps = int(max_steps)
        self.target_threshold = float(target_threshold)
        self.render_enabled = bool(render)

        self.step_count = 0
        self.goal_position = None
        self.prev_distance = None

        low = float(np.cos(np.pi))
        high = float(1.0)

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(9,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(3,),
            dtype=np.float32
        )

        self.max_velocity = float(2.0)

        self.workspace_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.workspace_high = np.array([0.2, 0.2, 0.2], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        self.sim.reset()

        self.goal_position = self.np_random.uniform(
            low=self.workspace_low,
            high=self.workspace_high
        ).astype(np.float32)

        state = self.sim.get_states()
        current_pos = self._get_pos(state)
        current_vel = self._get_joint_vel(state)

        distance = float(np.linalg.norm(current_pos - self.goal_position))
        self.prev_distance = distance

        obs = self._build_obs(current_pos, current_vel)

        info = {}
        info['distance_to_goal'] = distance
        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32)
        vel = action * np.float32(self.max_velocity)
        full_action = [float(vel[0]), float(vel[1]), float(vel[2]), 0]
        self.sim.run([full_action])

        state = self.sim.get_states()
        current_pos = self._get_pos(state)
        current_vel = self._get_joint_vel(state)

        distance = float(np.linalg.norm(current_pos - self.goal_position))

        progress = float(self.prev_distance - distance)
        self.prev_distance = distance

        reward = float(progress * 10.0)
        action_penalty = float(np.sum(np.square(action)) * 0.0001)
        reward = float(reward - action_penalty)

        success = bool(distance < self.target_threshold)

        terminated = success
        truncated = bool(self.step_count >= self.max_steps)

        obs = self._build_obs(current_pos, current_vel)

        info = {}
        info['distance_to_goal'] = distance
        info['success'] = float(success)
        info['progress'] = progress
        info['reward'] = reward

        return obs, reward, terminated, truncated, info

    def _build_obs(self, current_pos, current_vel):
        pos_n = self._norm_pos(current_pos)
        goal_n = self._norm_pos(self.goal_position)
        vel_n = self._norm_vel(current_vel)

        obs = np.concatenate([pos_n, goal_n, vel_n]).astype(np.float32)
        low = np.float32(np.cos(np.pi))
        high = np.float32(1.0)
        obs = np.clip(obs, low, high)
        return obs

    def _norm_pos(self, pos):
        pos = np.asarray(pos, dtype=np.float32)
        scale = (self.workspace_high - self.workspace_low)
        out = (pos - self.workspace_low) / scale
        out = out * np.float32(2.0) + np.float32(np.cos(np.pi))
        return out.astype(np.float32)

    def _norm_vel(self, vel):
        vel = np.asarray(vel, dtype=np.float32)
        out = vel / np.float32(self.max_velocity)
        return out.astype(np.float32)

    def _get_pos(self, state_dict):
        robot_id = list(sorted(state_dict.keys()))[0]
        rs = state_dict.get(robot_id, {})
        return np.asarray(rs.get('pipette_position', [0.0, 0.0, 0.0]), dtype=np.float32)

    def _get_joint_vel(self, state_dict):
        robot_id = list(sorted(state_dict.keys()))[0]
        rs = state_dict.get(robot_id, {})
        js = rs.get('joint_states', {})
        v = []
        for j in (0, 1, 2):
            rec = js.get(j, {})
            v.append(rec.get('velocity', 0.0))
        return np.asarray(v, dtype=np.float32)