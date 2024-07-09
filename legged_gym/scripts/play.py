import sys
import os
from time import sleep
cur_path = os.getcwd()
sys.path.append(os.path.join(cur_path, "../"))

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

from rsl_rl.env.wrappers.history_wrapper import HistoryWrapper

import torch

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.terrain_length = 8
    env_cfg.terrain.terrain_width = 8
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 2
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True
    env_cfg.env.play_teacher = True
    env_cfg.env.play_commond = False
    
    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap_terrain, pit_terrain]
    env_cfg.terrain.terrain_proportions = [0., 0., .2, .2, .3, .3, .0]
    env_cfg.terrain.terrain_proportions = [0., 0., .25, .25, .25, .25, .0]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.debug_viz = True
    env = HistoryWrapper(env)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(env_cfg.env.play_teacher,device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    # if EXPORT_POLICY:
    #     path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    #     # export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    #     #! changed by wz
    #     # export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    #     # print('Exported policy as jit script to: ', path)

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs)
        # actions=torch.zeros(env.num_envs,env.num_actions,dtype=torch.float32,device=env.device)
        # print("obs:",obs)
        obs,  rews, dones, infos = env.step(actions.detach())
        sleep(0.02)

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
