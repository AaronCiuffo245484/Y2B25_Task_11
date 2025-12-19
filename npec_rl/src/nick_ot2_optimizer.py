from clearml import Task
from clearml.automation import HyperParameterOptimizer, DiscreteParameterRange
from clearml.automation import GridSearch
from datetime import datetime

# Generate timestamp for unique task name and model filename
timestamp = datetime.now().strftime("%y%m%d.%H%M")

PERSON_NAME = "NickBelterman"
BRANCH_NAME = "batch_size_nick"
ENTRYPOINT = "nick_train_ot2.py"

task_name = f'OT2_RL_{PERSON_NAME}_{timestamp}'
# 1. Initialize a Task for the Optimizer itself
task = Task.init(
    project_name='Mentor Group - Jason/Group 1',
    task_name=task_name+"_batch_size_optimization",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)


# 2. Define the search space
param_distribution = [
    # --- Batching & Buffer ---
    # batch_size must divide n_steps! 
    # Since n_steps is 2048, these are all valid:
    DiscreteParameterRange('Args/batch_size', [256, 512, 1024]),
    DiscreteParameterRange('Args/n_steps', [2048]),

    # --- Learning Rate ---
    # 3e-4 is standard; 1e-4 is safer/slower; 1e-3 is aggressive
    # DiscreteParameterRange('Args/learning_rate', [0.0001, 0.0003, 0.0007]),

    # --- Discount Factor (Gamma) ---
    # How much the agent cares about future rewards. 0.99 is standard.
    # 0.999 is better for long-horizon tasks (like complex gantry paths).
    # DiscreteParameterRange('Args/gamma', [0.95, 0.98, 0.99, 0.999]),

    # --- Entropy Coefficient ---
    # Controls exploration. Higher = more random movement (prevents getting stuck).
    # Critical for robotics to ensure the agent doesn't just sit still.
    # DiscreteParameterRange('Args/ent_coef', [0.01, 0.05]),

    # --- GAE Lambda ---
    # Variance vs Bias trade-off. 0.95 is the gold standard for PPO.
    # DiscreteParameterRange('Args/gae_lambda', [0.9, 0.95, 1.0]),

    # --- Clip Range ---
    # How much the policy can change in one update. Lower is more stable.
    DiscreteParameterRange('Args/clip_range', [0.1, 0.2, 0.3])
]

# 3. Setup the Optimizer
optimizer = HyperParameterOptimizer(
    base_task_id='e45b3c1b96884b9c9ec8be2473c55235', # Change this value to the parent task ID of a previous run of your train script
    hyper_parameters=param_distribution,
    objective_metric_title='ot2',
    objective_metric_series='success_rate_100ep',
    objective_metric_sign='max',
    optimizer_class=GridSearch,
    execution_queue='default',
    max_number_of_concurrent_tasks=1,
    total_max_jobs=9
)

optimizer.start()

optimizer.wait()

optimizer.stop()