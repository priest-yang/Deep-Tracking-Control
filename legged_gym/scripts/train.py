import sys
import os
cur_path = os.getcwd()
sys.path.append(os.path.join(cur_path, "../"))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import get_args
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

import inspect
def save_config(class_obj, path, file_name):
    with open(f"{path}/{file_name}", 'w') as file:
        for name, obj in inspect.getmembers(class_obj):
            if not name.startswith('_') and not inspect.isfunction(obj) and not inspect.ismethod(obj):
                file.write(f"{name} = {obj}\n")

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env.debug_viz = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    os.makedirs(ppo_runner.log_dir, exist_ok=True)
    log_dir = ppo_runner.log_dir
    save_config(env_cfg.rewards.scales, log_dir, "reward_scale.ini")
    
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
