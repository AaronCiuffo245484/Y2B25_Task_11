from clearml import Task
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from sim_class import Simulation
from aaron_ot2_gym_wrapper import OT2Env

def main():
    task = Task.init(project_name='ot2', task_name='ppo_ot2_stock')

    parser = argparse.ArgumentParser(prefix_chars='/')
    parser.add_argument('/learning_rate', type=float, default=0.0001)
    parser.add_argument('/n_steps', type=int, default=8192)
    parser.add_argument('/batch_size', type=int, default=256)
    parser.add_argument('/n_epochs', type=int, default=10)
    parser.add_argument('/gamma', type=float, default=0.99)
    parser.add_argument('/clip_range', type=float, default=0.2)
    parser.add_argument('/ent_coef', type=float, default=0.0)
    parser.add_argument('/total_timesteps', type=int, default=1000000)
    parser.add_argument('/max_steps', type=int, default=400)
    parser.add_argument('/target_threshold', type=float, default=0.01)
    args = parser.parse_args()

    task.connect(vars(args))

    sim = Simulation(num_agents=1, render=False)

    env = OT2Env(sim=sim, max_steps=args.max_steps, target_threshold=args.target_threshold, render=False)
    env = Monitor(env)

    eval_sim = Simulation(num_agents=1, render=False)
    eval_env = OT2Env(sim=eval_sim, max_steps=args.max_steps, target_threshold=args.target_threshold, render=False)
    eval_env = Monitor(eval_env)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path='models',
        log_path='eval',
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True
    )

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        tensorboard_log='runs',
        verbose=1
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_cb,
        tb_log_name='ppo_ot2'
    )

    model.save('ppo_ot2_final')

if __name__ == '__main__':
    main()