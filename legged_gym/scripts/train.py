import sys
import os
cur_path = os.getcwd()
sys.path.append(os.path.join(cur_path, "../"))

from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.utils.helpers import get_args
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR



def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env.debug_viz = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    train(args)
