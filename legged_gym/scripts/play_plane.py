import sys
sys.path.append("/home/ysc/dr_gym/")
from legged_gym import LEGGED_GYM_ROOT_DIR
import os



from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

from rsl_rl.env.wrappers.history_wrapper import HistoryWrapper


import torch

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False

    env_cfg.terrain.mesh_type = 'plane'

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True
    env_cfg.env.play_teacher = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
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

        print("time:", env.episode_length_buf[env.lookat_id].item() / 50, 
              "cmd vx", env.commands[env.lookat_id, 0].item(),
              "actual vx", env.base_lin_vel[env.lookat_id, 0].item(), 
              "cmd vy", env.commands[env.lookat_id, 1].item(),
              "actual vy", env.base_lin_vel[env.lookat_id, 1].item(),
              "cmd wz", env.commands[env.lookat_id, 2].item(),
              "actual wz", env.base_ang_vel[env.lookat_id, 2].item(),
              "cmd heading", env.commands[env.lookat_id, 3].item(),
              "actual_height", env.base_pos[env.lookat_id, 2].item(),
            )

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
