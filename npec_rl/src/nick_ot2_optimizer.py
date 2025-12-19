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
# We use DiscreteParameterRange because PPO batch sizes should be powers of 2
param_distribution = [
    DiscreteParameterRange('Args/batch_size', [128, 256, 512, 1024]),
    DiscreteParameterRange('Args/n_steps', [2048]) 
]

# 3. Setup the Optimizer
optimizer = HyperParameterOptimizer(
    base_task_id='aeb472795604444eae90c234f888e37c', 
    hyper_parameters=param_distribution,
    objective_metric_title='ot2',
    objective_metric_series='success_rate_100ep',
    objective_metric_sign='max',
    optimizer_class=GridSearch,
    execution_queue='default',
    max_number_of_concurrent_tasks=1,
    total_max_jobs=4 
)

optimizer.start()

optimizer.wait()

optimizer.stop()