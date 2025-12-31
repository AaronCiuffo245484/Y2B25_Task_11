import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from clearml import Task
from datetime import datetime
import numpy as np
from filipp_ot2_wrapper import OT2Env

PERSON_NAME = "filipp"
BRANCH_NAME = "Filipp"
REWARD_TYPE = "normalized_progress"
timestamp = datetime.now().strftime("%y%m%d.%H%M")

LEARNING_RATE = 0.001
BATCH_SIZE = 128
N_STEPS = 2048
TOTAL_TIMESTEPS = 2000000
GAMMA = 0.99
MAX_STEPS_TRUNCATE = 250
TARGET_THRESHOLD = 0.001

class OT2Callback(BaseCallback):
    def __init__(self, threshold=0.001, verbose=0):
        super().__init__(verbose)
        self.threshold = threshold
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_final_distances = []
    
    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        for i, done in enumerate(dones):
            if done:
                infos = self.locals.get('infos', [])
                if i < len(infos):
                    info = infos[i]
                    final_dist = info.get('distance_to_goal', float('inf'))
                    ep_info = info.get('episode')
                    if ep_info is not None:
                        ep_reward = ep_info['r']
                        ep_length = ep_info['l']
                        
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)
                        success = float(final_dist < self.threshold)
                        self.episode_successes.append(success)
                        self.episode_final_distances.append(final_dist)
                        
                        self.logger.record('ot2/episode_reward', ep_reward)
                        self.logger.record('ot2/episode_length', ep_length)
                        self.logger.record('ot2/final_distance_mm', final_dist * 1000)
                        self.logger.record('ot2/success', success)
                        
                        if len(self.episode_successes) >= 10:
                            window = min(100, len(self.episode_successes))
                            self.logger.record('ot2/success_rate_100ep', 
                                             np.mean(self.episode_successes[-window:]))
                            self.logger.record('ot2/avg_length_100ep', 
                                             np.mean(self.episode_lengths[-window:]))
                            self.logger.record('ot2/avg_final_dist_mm_100ep', 
                                             np.mean(self.episode_final_distances[-window:]) * 1000)
        return True

task_name = f'OT2_RL_{PERSON_NAME}_{REWARD_TYPE}_{timestamp}'
task = Task.init(project_name='Mentor Group - Jason/Group 1', task_name=task_name)
task.set_repo(repo='https://github.com/AaronCiuffo245484/Y2B25_Task_11.git', branch=BRANCH_NAME)
task.set_base_docker('deanis/2023y2b-rl:latest')
task.set_packages(['tensorboard', 'clearml'])
task.execute_remotely(queue_name='default')

lr_str = f"{LEARNING_RATE:.0e}".replace("+", "").replace("-0", "-")
filename = f"{timestamp}_{PERSON_NAME}_{REWARD_TYPE}_lr{lr_str}_b{BATCH_SIZE}_s{N_STEPS}_th{int(TARGET_THRESHOLD*1000)}mm"

env = OT2Env(render=False, max_steps=MAX_STEPS_TRUNCATE, target_threshold=TARGET_THRESHOLD, reward_type=REWARD_TYPE)

model = PPO('MlpPolicy', env, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, n_steps=N_STEPS,
            n_epochs=10, gamma=GAMMA, gae_lambda=0.95, clip_range=0.2, verbose=1,
            tensorboard_log=f"runs/{PERSON_NAME}/{REWARD_TYPE}")

ot2_callback = OT2Callback(threshold=TARGET_THRESHOLD, verbose=1)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=ot2_callback, tb_log_name=f"PPO_{filename}")

model_name = f"{filename}.zip"
model.save(model_name)
task.upload_artifact("model", artifact_object=model_name)

try:
    env.close()
except:
    pass