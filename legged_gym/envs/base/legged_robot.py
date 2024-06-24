# import torch.nn.functional
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch.nn.functional
import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from scipy.spatial.transform import Rotation as R

from collections import OrderedDict, defaultdict
import cv2
import torchvision.transforms as T
import time
import itertools


# import open3d as o3d

def euler_from_quaternion(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False 

        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.lidar_global_counter = 0
        self.height_global_counter = 0
        self.j = 0

        ####PMTrajectoryGenerator test:#####
        self.robot = None
        task_name = 'lite'      
    def clock(self):#PMTrajectoryGenerator
        return self.gym.get_sim_time(self.sim)
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            # print('t = ',time.time())
            # print('torques = ',self.torques)
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)

            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        # print('T = ',time.time())
        # return clipped obs, clipped states (None), rewards, dones and infos
        
        # breakpoint()
        
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    #! added by shaoze
    @staticmethod
    def rotate_positions(base_pos, thetas):
        
        def batch_rotation_matrix_z(thetas):
            cos_thetas = torch.cos(thetas)
            sin_thetas = torch.sin(thetas)
            # Create a batch of rotation matrices
            zeros = torch.zeros_like(thetas)
            ones = torch.ones_like(thetas)
            Rz = torch.stack([
                cos_thetas, -sin_thetas, zeros,
                sin_thetas, cos_thetas, zeros,
                zeros, zeros, ones
            ], dim=1).reshape(-1, 3, 3)
            return Rz

        Rz = batch_rotation_matrix_z(thetas)
        
        rotated_pos = torch.bmm(Rz, base_pos.transpose(1, 2)).transpose(1, 2)
        return rotated_pos

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)


        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.last_base_ang_vel = self.base_ang_vel.clone()
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10]) #robot velocity
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        
        #to  estimate the pos 
        self.base_pos[:] = self.root_states[:, :3]
        #test
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13
                                                          )[:, self.feet_indices, 7:10]

        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3]
        
        ##PMTrajectoryGenerator test:###
        # self.cpg_phase_information = self.pmtg.update_observation()

        self._post_physics_step_callback()
        
        ###########################################################################
        #! added by shaoze
        self.hip_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.hip_indices, 0:3]
        
        #! for DTC foothold prediction
        hip_to_base = self.hip_positions - self.base_pos.unsqueeze(1).repeat(1,4,1)
        yaw_vel_cmd = self.commands[:, 2]
        rotated_hip_to_base = self.rotate_positions(hip_to_base, yaw_vel_cmd)
        p_shoulder_i = self.base_pos.unsqueeze(1).repeat(1,4,1) + rotated_hip_to_base
        t_stance = self.cfg.sim.dt * self.cfg.control.decimation
        fdbk_gain_k = 0.03
        cmd_lin_vel = torch.cat((self.commands[:, :2], torch.zeros(self.num_envs, 1, device=self.device)), dim=1)
        
        p_symmetric = t_stance / 2 * self.base_lin_vel.unsqueeze(1).repeat(1,4,1) + \
            fdbk_gain_k * (self.base_lin_vel.unsqueeze(1).repeat(1,4,1) - cmd_lin_vel.unsqueeze(1).repeat(1,4,1))
        
        
        self.pred_footholds = p_shoulder_i + p_symmetric
        
        pred_footholds_to_robot = self.pred_footholds - self.base_pos.unsqueeze(1).repeat(1,4,1)
        self.pred_footholds_to_robot = torch.zeros_like(pred_footholds_to_robot)
        for i in range(4):
            self.pred_footholds_to_robot[:,i,:] = quat_rotate_inverse(self.base_quat, pred_footholds_to_robot[:,i,:])
        
        
        #! DTC foothold score computation based on terrain
        if self.cfg.terrain.measure_heights:
            #! foothold score based on "Perceptive Locomotion in Rough Terrain"

            measure_height_to_base = self.measured_heights - self.base_pos[:, 2].unsqueeze(1).repeat(1, len(self.cfg.terrain.measured_points_x)*len(self.cfg.terrain.measured_points_y))
            measured_heights_grid = measure_height_to_base.clone().view(self.num_envs, len(self.cfg.terrain.measured_points_x), len(self.cfg.terrain.measured_points_y))
            
            #! filter exceptional points (compared to root)
            exception_heights = (measured_heights_grid > 1) | (measured_heights_grid < -1)
            
            measured_heights_grid = measured_heights_grid.clamp_(min=-0.5, max=0.5)
            d_x,d_y = torch.gradient(measured_heights_grid, dim=[1,2], spacing = 0.05) #! Note: spacing should be changed when the resolution of the terrain is changed
            self.slope = torch.sqrt(d_x**2 + d_y**2)
            h_mean = torch.mean(measured_heights_grid, dim=(1,2))
            roughness = measured_heights_grid - h_mean.unsqueeze(1).unsqueeze(2).repeat(1, len(self.cfg.terrain.measured_points_x), len(self.cfg.terrain.measured_points_y))
            roughness = torch.abs(roughness)
            # roughness_score = torch.nn.functional.normalize(roughness, p=2, dim=(1,2))
            # slope_score = torch.nn.functional.normalize(self.slope, p=2, dim=(1,2))
            edge = torch.sqrt(torch.var(measured_heights_grid, dim=(1,2))).unsqueeze(1).unsqueeze(2).repeat(1, len(self.cfg.terrain.measured_points_x), len(self.cfg.terrain.measured_points_y))
            edge = edge.clamp_(min=0.0, max=0.3)
            
            lambda_1, lambda_2, lambda_3 = 0.2, 1, 0.3
            foothold_score = lambda_1 * edge + lambda_2 * self.slope + lambda_3 * roughness
            
            foothold_score = foothold_score.view(self.num_envs, -1)
            foothold_score = torch.where(foothold_score < 0.1, foothold_score, torch.tensor(10.0, dtype=torch.float, device=self.device) )
            
            # take distance to nominal foothold
            # breakpoint()
            height_points_flattened = self.height_points.flatten(end_dim = -2)
            quat_flattened = self.base_quat.repeat(1, self.measured_heights.shape[1]).flatten()
            heights_world = quat_apply_yaw(quat_flattened, height_points_flattened) 
            self.heights_world = heights_world.reshape(self.num_envs, -1, 3) + self.base_pos.unsqueeze(1).repeat(1, len(self.cfg.terrain.measured_points_x)*len(self.cfg.terrain.measured_points_y), 1)
            self.heights_world[:, :, 2] = self.measured_heights
            
            footholds_repeated_envs = self.pred_footholds.unsqueeze(1).repeat(1, len(self.cfg.terrain.measured_points_x)*len(self.cfg.terrain.measured_points_y), 1, 1)
            heights_world_repeated4 = self.heights_world.unsqueeze(2).repeat(1,1,4,1)
            
            dis_to_nominal = torch.norm(footholds_repeated_envs[:, :, :, :-1] - heights_world_repeated4[:, :, :, :-1], dim=-1)            
            # dis_to_nominal: [num_envs, num_points, 4]
            # clip the dis with filling dis > 0.1 with 1.0, these points will be filtered
            dis_to_nominal = torch.where(dis_to_nominal < 0.16, dis_to_nominal, torch.tensor(10.0, dtype=torch.float, device=self.device) )
            
            #! for debug & visualize
            self.nominal_footholds_indice = dis_to_nominal.min(dim=1)[1]
            
            foothold_score = foothold_score.unsqueeze(2).repeat(1,1,4)
            foothold_score = foothold_score * 0.2 + dis_to_nominal * 0.8
            
            #! filter exceptional points
            foothold_score = torch.where(exception_heights.view(self.num_envs, -1).unsqueeze(2).repeat(1,1,4), torch.tensor(10.0, dtype=torch.float, device=self.device), foothold_score)
            
            self.foothold_score = foothold_score
            # foothold_score: [num_envs, num_points, 4] (each point respect to each foothold)
            
            # get the top n footholds
            ktop_num = 1  #! do not change
            optimal_values, optimal_indices = torch.topk(foothold_score, k=ktop_num, dim=1, largest=False, sorted=True)
            self.optimal_foothold_indice = optimal_indices
            
            #! compute for observation
            x_indices = torch.remainder(optimal_indices, self.cfg.terrain.measured_y_dim)
            y_indices = torch.div(optimal_indices, self.cfg.terrain.measured_y_dim, rounding_mode='trunc')

            # Map matrix indices to actual x, y values
            measured_points_x = torch.tensor(self.cfg.terrain.measured_points_x).unsqueeze(0).unsqueeze(0).repeat(self.num_envs, ktop_num, 4).to(self.device)
            measured_points_y = torch.tensor(self.cfg.terrain.measured_points_y).unsqueeze(0).unsqueeze(0).repeat(self.num_envs, ktop_num, 4).to(self.device)
            
            decoded_x_values = torch.gather(measured_points_x, -1, x_indices)
            decoded_y_values = torch.gather(measured_points_y, -1, y_indices)
            
            foothold_obs = torch.cat((decoded_x_values, decoded_y_values), dim=-2)
            self.foothold_obs = foothold_obs.view(self.num_envs, -1)
            
            #! compute for rewards
            if ktop_num == 1:
                optimal_indices = optimal_indices.squeeze(1)
                batch_indices = torch.arange(optimal_indices.size(0)).unsqueeze(-1).expand_as(optimal_indices)
                self.optimal_footholds_world = self.heights_world[batch_indices, optimal_indices]

        ###########################################################################

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        self.last_actions_2[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_foot_velocities[:] = self.foot_velocities[:]

        #! added by wz
        self.base_ang_vel_last = self.base_ang_vel.clone()
        self.base_lin_vel_last = self.base_lin_vel.clone()
        
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 100., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        # self.reset_buf |= (self.projected_gravity[:,2] > -0.2) 

        #! stand up
        # self.reset_buf |= (self.projected_gravity[:,2] > 0.2)  
        #! X30
        # self.reset_buf |= ((torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights[:, 55:132], dim=1)) < 0.25)
        #! lite3
        
        self.reset_buf |= ((torch.mean(self.root_states[:, 2].unsqueeze(1) - 
                                       self.measured_heights[:, 10 * 21: (33-10)*21], 
                                       dim=1)) < 0.1) #! 55-132 changed according to terrain resolution
        
        self.reset_buf |= ((torch.mean(self.root_states[:, 2].unsqueeze(1) - 
                                       self.foot_positions[:, :, 2], 
                                       dim=1)) < 0.1) 
        
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # self.pmtg.reset(env_ids)#PMTrajectoryGenerator test:###

        self._resample_commands(env_ids)

        self._randomize_dof_props(env_ids)

        self.height_noise_offset[env_ids] = self.height_noise_offset[env_ids]*0.0
        self.height_noise_offset[env_ids] += np.random.normal(0,0.02)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_actions_2[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1.
        self.feet_contact_time[env_ids] = 0.
        self.last_scale_actions[env_ids] = 0
        self.last_scale_actions2[env_ids] = 0
        self.pitch_est[env_ids] = 0

        #! added by wz
        self.base_ang_vel_last[env_ids] = 0.
        self.base_lin_vel_last[env_ids] = 0.

        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0
        for i in range(len(self.stumb_buffer)):
            self.stumb_buffer[i][env_ids, :] = 0
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
            
        #! added by shaoze
        self.contact_filt[env_ids] = False
        self.last_contacts[env_ids] = False
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    ############camera test
    # def _get_forward_depth_obs(self):
    #     return torch.stack(self.sensor_tensor_dict["forward_depth"]).flatten(start_dim= 1)
    
    # def _get_backward_depth_obs(self):
    #     return torch.stack(self.sensor_tensor_dict["backward_depth"]).flatten(start_dim= 1)
    
    # def _crop_depth_images(self, depth_images):
    #     H, W = depth_images.shape[-2:]
    #     return depth_images[...,
    #         self.cfg.sensor.forward_camera.crop_top_bottom[0]: H - self.cfg.sensor.forward_camera.crop_top_bottom[1],
    #         self.cfg.sensor.forward_camera.crop_left_right[0]: W - self.cfg.sensor.forward_camera.crop_left_right[1],
    #     ]
      
        
    # def _normalize_depth_images(self, depth_images):
    #     depth_images = torch.clip(
    #         depth_images,
    #         self.cfg.sensor.forward_camera.depth_range[0],
    #         self.cfg.sensor.forward_camera.depth_range[1],
    #     )
    #     # normalize depth image to (0, 1)
    #     depth_images = (depth_images - self.cfg.sensor.forward_camera.depth_range[0]) / (
    #         self.cfg.sensor.forward_camera.depth_range[1] - self.cfg.sensor.forward_camera.depth_range[0]
    #     )
    #     return depth_images    
    ####################### 

    def compute_observations(self):
        """ Computes observations
        """ 
        # self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
        #                             self.base_ang_vel  * self.obs_scales.ang_vel,
        #                             self.projected_gravity,
        #                             self.commands[:, :3] * self.commands_scale,
        #                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
        #                             self.dof_vel * self.obs_scales.dof_vel,
        #                             self.actions
        #                             ),dim=-1)
        # add perceptive inputs if not blind
        # if self.cfg.terrain.measure_heights:
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        #     self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        #

        #PMTrajectoryGenerator test:###
        cpg_phase_information = self.cpg_phase_information
        
        #! no teacher-student arch
        self.obs_buf = torch.cat((  
                                
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity, #* self.obs_scales.gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    # cpg_phase_information *1.0 #PMTrajectoryGenerator test:###
                                    ###############################
                                    #! DTC
                                    self.foothold_obs, # 2 * 4 = 8
                                    ###############################
                                    ),dim=-1)
        
        if self.cfg.terrain.measure_heights:
            self.heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.rewards.base_height_target - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
        self.privileged_obs_buf = torch.cat(
            (
                self.heights +(2 * torch.rand_like(self.heights) - 1) * 0.1 + self.height_noise_offset,
                # self.base_lin_vel * self.obs_scales.lin_vel,
                # (self.friction_coeffs.unsqueeze(1) - friction_coeffs_shift).squeeze(-1).squeeze(-1) * friction_coeffs_scale,  # friction coeff
                # (self.restitutions.unsqueeze(1) - restitutions_shift).squeeze(-1).squeeze(-1) * restitutions_scale,  # friction coeff
                # (self.payloads.unsqueeze(1) - payloads_shift).squeeze(-1).squeeze(-1) * payloads_scale,  # payload
                self.forces[:,0,:]* self.obs_scales.force,
                # self.force_positions[:,0,:]* self.obs_scales.force,
                # (self.com_displacements.squeeze(-1) - com_displacements_shift) * com_displacements_scale, 
                # self.measured_foot_clearance,

                # (self.contact_forces[:, self.feet_indices, 0]+ (2 * torch.rand_like(self.contact_forces[:, self.feet_indices, 0]) - 1) * 5)*self.obs_scales.force_contact ,
                # (self.contact_forces[:, self.feet_indices, 1]+ (2 * torch.rand_like(self.contact_forces[:, self.feet_indices, 1]) - 1) * 5)*self.obs_scales.force_contact ,
                # (self.contact_forces[:, self.feet_indices, 2]+ (2 * torch.rand_like(self.contact_forces[:, self.feet_indices, 2]) - 1) * 5)*self.obs_scales.force_contact,
                
                # 1.0*(torch.norm(self.contact_forces[:, self.collision_contact_indices, :], dim=-1) > 0.1),
                self.heights ,
                # (self.motor_strengths - motor_strengths_shift) * motor_strengths_scale,  # motor strength

                
                
            ), dim=1)

        # add noise if needed
        if self.add_noise:
            # print('self.noise_scale_vec: ',self.noise_scale_vec)
            # print('obs_buf',self.obs_buf)
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec[:53]
            # print('obs_buf_after',self.obs_buf)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """

        if self.cfg.domain_rand.randomize_base_mass:
            if env_id==0:
                # prepare friction randomization
                added_mass_range = self.cfg.domain_rand.added_mass_range
                num_buckets = 64*5
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                mass_buckets = torch_rand_float(added_mass_range[0], added_mass_range[1], (num_buckets,1), device=self.device)
                self.payloads = mass_buckets[bucket_ids]

        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64*5
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device=self.device)
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare friction randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                num_buckets = 64*5
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets,1), device=self.device)
                self.restitutions = restitution_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].restitution = self.restitutions[env_id]
        return props
    

    
    def _randomize_dof_props(self, env_ids):
        if self.cfg.domain_rand.randomize_motor_strength:
            min_strength, max_strength = self.cfg.domain_rand.motor_strength
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_strength - min_strength) + min_strength

        if self.cfg.domain_rand.randomize_Kp_factor:
            min_Kp_factor, max_Kp_factor = self.cfg.domain_rand.kp_range
            self.Kp_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kp_factor - min_Kp_factor) + min_Kp_factor
        if self.cfg.domain_rand.randomize_Kd_factor:
            min_Kd_factor, max_Kd_factor = self.cfg.domain_rand.kd_range
            self.Kd_factors[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device,
                                                     requires_grad=False).unsqueeze(1) * (
                                                  max_Kd_factor - min_Kd_factor) + min_Kd_factor
  
    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = 0.8*props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        self.default_body_mass = props[0].mass
        props[0].mass = self.default_body_mass + self.payloads[env_id]
        #! added by wz
        self.robot_mass[env_id] +=props[0].mass
        props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1],
                                   self.com_displacements[env_id, 2])
        #! added by wz
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            scale = np.random.uniform(rng[0], rng[1])
            for i in range(1, len(props)):
                props[i].mass *= scale
                self.robot_mass[env_id] +=props[i].mass

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1.5, 1.5)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

            
        self.height_global_counter += 1
        
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval in range(2)):
            if self.common_step_counter % self.cfg.domain_rand.push_interval == 0:
                max_force = self.cfg.domain_rand.max_push_force_xy
                max_offset = self.cfg.domain_rand.max_push_force_offset
                self.forces[:,0,0:2] = torch_rand_float(-max_force, max_force, (self.num_envs,2), device=self.device)
                self.force_positions[:,0,0:3] = torch_rand_float(-max_offset, max_offset, (self.num_envs, 3), device=self.device)
            # print(self.common_step_counter, self.forces[1,:])
            self._push_robots()
        else :
            self.forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            self.force_positions = self.rb_positions.clone()
            
        #! added by shaoze
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)        
        # #temp set the command by ourself
        if self.cfg.env.play_commond == True:
            self.commands[env_ids, 0] = 0.5
            self.commands[env_ids, 1] = 0.0
            if self.cfg.commands.heading_command:
                # self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
                self.commands[env_ids, 3] = 0
            else:
                self.commands[env_ids, 2] = 0.

        # set small commands to zero

        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.1).unsqueeze(1)
        self.forces[env_ids, :] = torch.zeros((len(env_ids), self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.force_positions = self.rb_positions.clone()

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        #filter 
        choice = np.random.randint(1,5)

        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
        goal_actions_limit_scale = 1.0
        goal_actions = torch.clip(self.lag_buffer[choice] + self.default_dof_pos, goal_actions_limit_scale*self.dof_pos_limits[:,0], goal_actions_limit_scale*self.dof_pos_limits[:,1]).to(self.device)
        # print("no limit goal action = ",self.lag_buffer[choice] + self.default_dof_pos)
        if control_type=="P":
            # torques = self.p_gains* self.Kp_factors *(actions_scaled + self.default_dof_pos - self.dof_pos) -  self.d_gains* self.Kd_factors *self.dof_vel
            torques = self.p_gains* self.Kp_factors *(goal_actions - self.dof_pos + self.motor_offsets) -  self.d_gains* self.Kd_factors *self.dof_vel            # self.last_scale_actions = actions_scaled.clone()

            # self.last_scale_actions = actions_scaled.clone()
            # self.last_scale_actions2 = self.last_scale_actions.clone()
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        torques = torques * self.motor_strengths
        return torch.clip(torques , -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            # self.base_init_state[2]=torch_rand_float(self.cfg.init_state.pos_z_range[0], self.cfg.init_state.pos_z_range[1], (1,1),device=self.device)
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # self.root_states[env_ids, 0] = self.root_states[env_ids, 0]-10
    
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        
        #need to check reset forces when we don't push robot
        # max_force = self.cfg.domain_rand.max_push_force_xy
        # max_offset = self.cfg.domain_rand.max_push_force_offset
        # self.forces = self.forces.clone()
        # # self.forces[:,0,0:2] = torch_rand_float(-max_force, max_force, (self.num_envs,2), device=self.device)
        # self.force_positions = self.force_positions.clone()
        # # self.force_positions[:,0,0:3] = torch_rand_float(-max_offset, max_offset, (self.num_envs, 3), device=self.device)
        # self.gym.apply_rigid_body_force_at_pos_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), gymtorch.unwrap_tensor(self.force_positions), gymapi.ENV_SPACE)


    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length*0.6
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        # print(self.terrain_levels[env_ids])
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
        # self.env_origins[env_ids][0] = 45
        # self.env_origins[env_ids][1] = 45

    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:53] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[53:746] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)
        self.rb_positions = self.rigid_body_state[:, 0:3].view(self.num_envs, self.num_bodies, 3)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.stair_vector = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,
                               self.feet_indices,
                               7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices,
                              0:3] 
        #to estimate the pos
        self.base_pos = self.root_states[:, :3]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_torques2 = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_torques3 = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions_est = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions_2 = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_scale_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_scale_actions2 = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_foot_velocities = torch.zeros_like(self.foot_velocities)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])

        #! changed by wz
        self.base_lin_vel_last = torch.zeros_like(self.base_lin_vel)
        
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        #! changed by wz
        self.base_ang_vel_last = torch.zeros_like(self.base_ang_vel)

        self.last_base_ang_vel = torch.zeros_like(self.base_ang_vel)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.pitch_est = torch.zeros_like(self.gravity_vec[:,0])
        self.feet_contact_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)

        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(5+1)]
        self.stumb_buffer = [torch.zeros_like(self.last_contacts) for i in range(5) ]
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.measured_foot_clearance = 0

        self.forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.force_positions = self.rb_positions.clone()
        
        self.height_noise_offset = torch.zeros(self.num_envs, 693, device=self.device, requires_grad=False)

        #! added by shaoze
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        
        # test camera
        # self.sensor_tensor_dict = defaultdict(list)
        # self.forward_depth_resize_transform = T.Resize(
        #     self.cfg.sensor.forward_camera.resized_resolution,
        #     interpolation= T.InterpolationMode.BICUBIC,
        # )
        # for env_i, env_handle in enumerate(self.envs):
        #     self.sensor_tensor_dict["forward_depth"].append(gymtorch.wrap_tensor(
        #         self.gym.get_camera_image_gpu_tensor(
        #             self.sim,
        #             env_handle,
        #             self.sensor_handles[env_i]["forward_camera"],
        #             gymapi.IMAGE_DEPTH,
        #     )))
        #     self.sensor_tensor_dict["backward_depth"].append(gymtorch.wrap_tensor(
        #         self.gym.get_camera_image_gpu_tensor(
        #             self.sim,
        #             env_handle,
        #             self.sensor_handles[env_i]["backward_camera"],
        #             gymapi.IMAGE_DEPTH,
        #     )))
        #######################

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        ###PMTrajectoryGenerator test:
        self.delta_phi = torch.zeros(self.num_envs,
                                4,
                                dtype=torch.float,
                                device=self.device,
                                requires_grad=False)
        self.residual_angle = torch.zeros(self.num_envs,
                                          12,
                                          dtype=torch.float,
                                          device=self.device,
                                          requires_grad=False)
        self.cpg_phase_information = torch.ones((self.num_envs, 13),
                                                device=self.device,
                                                dtype=torch.float,
                                                requires_grad=False)  
              

    def _init_custom_buffers__(self):
        # domain randomization properties
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.payloads = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                     requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                    requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)

        self.robot_mass = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                         requires_grad=False)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
        # slope_p = np.array([0.0, 0.0, 0.0])
        # slope_size = np.array([1.0, 1.0, 0.1])
        # slop_actor = self.gym.create_actor()
        # slope_l = 1.0
        # self.gym.add_slope(self.sim ,0.0 ,0,0, slope_h, slope_l)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        
        
        
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_sensors(self, env_handle=None, actor_handle= None):
        """ attach necessary sensors for each actor in each env
        Considering only one robot in each environment, this method takes only one actor_handle.
        Args:
            env_handle: env_handle from gym.create_env
            actor_handle: actor_handle from gym.create_actor
        Return:
            sensor_handle_dict: a dict of sensor_handles with key as sensor name (defined in cfg["sensor"])
        """
        sensor_handle_dict = dict()

        camera_handle = self._create_onboard_camera(env_handle, actor_handle, "forward_camera")
        sensor_handle_dict["forward_camera"] = camera_handle
        
        camera_handle = self._create_onboard_camera(env_handle, actor_handle, "backward_camera")
        sensor_handle_dict["backward_camera"] = camera_handle

        return sensor_handle_dict
    
    def _create_onboard_camera(self, env_handle, actor_handle, sensor_name):
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height = getattr(self.cfg.sensor, sensor_name).resolution[0]
        camera_props.width = getattr(self.cfg.sensor, sensor_name).resolution[1]
        camera_props.use_collision_geometry = True
        if hasattr(getattr(self.cfg.sensor, sensor_name), "horizontal_fov"):
            camera_props.horizontal_fov = np.random.uniform(
                getattr(self.cfg.sensor, sensor_name).horizontal_fov[0],
                getattr(self.cfg.sensor, sensor_name).horizontal_fov[1],
            ) if isinstance(getattr(self.cfg.sensor, sensor_name).horizontal_fov, (tuple, list)) else getattr(self.cfg.sensor, sensor_name).horizontal_fov
            # vertical_fov = horizontal_fov * camera_props.height / camera_props.width
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        local_transform = gymapi.Transform()

        if sensor_name == 'forward_camera':
            position = [0.26914, 0.0, 0.00985]
            rotation = [1.5708, 1.39626, -3.14159]
            local_transform.p = gymapi.Vec3(*position)
            local_transform.r = gymapi.Quat.from_euler_zyx(*rotation)
        else:
            position = [-0.26914, 0.0, 0.00985] # position in base_link
            rotation = [1.5708, 1.39626, 3.52623e-16] # ZYX Euler angle in base_link
            local_transform.p = gymapi.Vec3(*position)
            local_transform.r = gymapi.Quat.from_euler_zyx(*rotation)

        self.gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            actor_handle,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )
        
        return camera_handle
        
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        # self.default_restitution = 1.
        self._init_custom_buffers__()

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        # hip_names = ["FL_HipX_joint", "FR_HipX_joint", "HL_HipX_joint", "HR_HipX_joint"]

        hip_names = [s for s in body_names if self.cfg.asset.hip_name in s]
        
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        
        
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = body_names.index(name)
        

        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        
        collision_contact_names = []
        for name in self.cfg.asset.collision_state:
            collision_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        # camera test
        self.sensor_handles = []
        def str_to_bit(s):
            return int(s.replace(' ', ''), 2)
        #! lite3
        collision_filter = {
            'base': str_to_bit('1000 1000 1000 0000'),
            'fl_hip': str_to_bit('1100 0000 0000 0000'),
            'fl_thigh': str_to_bit('1110 0000 0000 0000'),
            'fl_shank': str_to_bit('0011 0000 0000 0000'),
            'fl_foot': str_to_bit('0001 0000 0000 0000'),

            'fr_hip': str_to_bit('1000 0100 0000 0000'),
            'fr_thigh': str_to_bit('1000 0110 0000 0000'),
            'fr_shank': str_to_bit('0000 0011 0000 0000'),
            'fr_foot': str_to_bit('0000 0001 0000 0000'),

            'hl_hip': str_to_bit('1000 0000 0100 0000'), 
            'hl_thigh': str_to_bit('1000 0000 0110 0000'),
            'hl_shank': str_to_bit('0000 0000 0011 0000'),
            'hl_foot': str_to_bit('0000 0000 0001 0000'),

            'hr_hip': str_to_bit('1000 0000 0000 0100'),
            'hr_thigh': str_to_bit('1000 0000 0000 0110'),
            'hr_shank': str_to_bit('0000 0000 0000 0011'),
            'hr_foot': str_to_bit('0000 0000 0000 0001'),
        }        

        #! x30fff
        # collision_filter = {
        #     'base': str_to_bit('1000 1000 1000 1000'),
        #     'fl_hip': str_to_bit('1100 0000 0000 0000'),
        #     'fl_thigh': str_to_bit('1110 0000 0000 0000'),
        #     'fl_shank': str_to_bit('1111 0000 0000 0000'),
        #     'fl_shank1': str_to_bit('1111 0000 0000 0000'),
        #     'fl_shank2': str_to_bit('1111 0000 0000 0000'),
        #     'fl_foot': str_to_bit('1111 0000 0000 0000'),

        #     'fr_hip': str_to_bit('1000 0100 0000 0000'),
        #     'fr_thigh': str_to_bit('1000 0110 0000 0000'),
        #     'fr_shank': str_to_bit('0000 1111 0000 0000'),
        #     'fr_shank1': str_to_bit('0000 1111 0000 0000'),
        #     'fr_shank2': str_to_bit('0000 1111 0000 0000'),
        #     'fr_foot': str_to_bit('0000 1111 0000 0000'),

        #     'hl_hip': str_to_bit('1000 0000 0100 0000'), 
        #     'hl_thigh': str_to_bit('1000 0000 0110 0000'),
        #     'hl_shank': str_to_bit('0000 0000 1111 0000'),
        #     'hl_shank1': str_to_bit('0000 0000 1111 0000'),
        #     'hl_shank2': str_to_bit('0000 0000 1111 0000'),
        #     'hl_foot': str_to_bit('0000 0000 1111 0000'),

        #     'hr_hip': str_to_bit('1000 0000 0000 0100'),
        #     'hr_thigh': str_to_bit('1000 0000 0000 0110'),
        #     'hr_shank': str_to_bit('0000 0000 0000 1111'),
        #     'hr_shank1': str_to_bit('0000 0000 0000 1111'),
        #     'hr_shank2': str_to_bit('0000 0000 0000 1111'),
        #     'hr_foot': str_to_bit('0000 0000 0000 1111'),
        # }

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            # print(rigid_shape_props)
            for shape_id in range(len(rigid_shape_props)):
                rigid_shape_props[shape_id].filter = list(collision_filter.values())[shape_id]

            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            
            rsp = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            for shape_id in range(len(rigid_shape_props)):
                rsp[shape_id].filter = list(collision_filter.values())[shape_id]
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rsp)

            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            #temp

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            #camera test
            # sensor_handle_dict = self._create_sensors(env_handle, actor_handle)
            # self.sensor_handles.append(sensor_handle_dict)


        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            # print(penalized_contact_names)
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        
        self.collision_contact_indices = torch.zeros(len(collision_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(collision_contact_names)):
            self.collision_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], collision_contact_names[i])


    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]


        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        
        # hip_sphere_geom = gymutil.WireframeSphereGeometry(radius=0.05, color=(1, 1, 0))
        
        base_sphere_geom = gymutil.WireframeSphereGeometry(radius=0.1, color=(0, 0, 1))
        foothold_nominal_sphere_geom= gymutil.WireframeSphereGeometry(radius=0.035, color=(0, 0, 1))
        foothold_edge_sphere_geom = gymutil.WireframeSphereGeometry(radius=0.03, color=(1, 0, 0))
        foothold_optimal_sphere_geom = gymutil.WireframeSphereGeometry(radius=0.03, color=(0, 1, 0))
        sphere_geom = gymutil.WireframeSphereGeometry(0.03, 4, 4, None, color=(1, 1, 0))
        command_hip = gymutil.WireframeSphereGeometry(radius=0.05, color=(1, 1, 0))
        
        # draw height lines
        if self.terrain.cfg.measure_heights:
            # self.gym.clear_lines(self.viewer)
            
            i = self.lookat_id
            

            # in_range_all = None
            # min_dis = None
            # for pos in self.pred_footholds[i, :]:
            #     dist = (pos[0].cpu() - x_all)**2 + (pos[1].cpu() - y_all)**2
            #     in_range = torch.where(dist < 0.08)[0]
            #     if in_range_all is not None:
            #         in_range_all = torch.cat((in_range_all, in_range))
            #     else:
            #         in_range_all = in_range
                    
            #     if min_dis is not None:
            #         min_dis = torch.min(min_dis, dist)
            #     else: 
            #         min_dis = dist
            
            # min_dis = 1 / min_dis
            # min_dis = torch.clamp(min_dis, 0, 5)
            # min_dis_score = torch.nn.functional.normalize(min_dis, dim=0)
            
            # foothold_score = foothold_score.cpu()*0.8 + min_dis_score.cpu()*0.2

            x_all = self.heights_world[i, :, 0].cpu().numpy()
            y_all = self.heights_world[i, :, 1].cpu().numpy()
            z_all = self.heights_world[i, :, 2].cpu().numpy()
            
            foothold_score = torch.min(self.foothold_score, dim=2)[0][i, :].cpu()

            # in_range_points_x = x_all[in_range_all]
            # in_range_points_y = y_all[in_range_all]
            # in_range_points_z = z_all[in_range_all]
            # scores = foothold_score[in_range_all]
            
            for x,y,z, score in zip(x_all, y_all, z_all, foothold_score):
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                if score > 0.9 and score < 8:
                    gymutil.draw_lines(foothold_edge_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
                # else:
                #     gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
                # gymutil.draw_lines(foothold_edge_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
            
            
            #! nominal foot_hold
            nominal_footholds_indice = self.nominal_footholds_indice[i, :].cpu().numpy()
            nominal_footholds_x = x_all[nominal_footholds_indice]
            nominal_footholds_y = y_all[nominal_footholds_indice]
            nominal_footholds_z = z_all[nominal_footholds_indice]
            
            for x,y,z in zip(nominal_footholds_x, nominal_footholds_y, nominal_footholds_z):
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(foothold_nominal_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
            
            
            #! draw optimal indices
            optimal_foothold_x = self.optimal_footholds_world[i, :, 0].cpu().numpy()
            optimal_foothold_y = self.optimal_footholds_world[i, :, 1].cpu().numpy()
            optimal_foothold_z = self.optimal_footholds_world[i, :, 2].cpu().numpy()
            
            for x,y,z in zip(optimal_foothold_x, optimal_foothold_y, optimal_foothold_z):
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(foothold_optimal_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
            

        #! shaoze
        i = self.lookat_id
        
        #! hip position
        # for pos in self.hip_positions[i, :]:
        #     sphere_pose = gymapi.Transform(gymapi.Vec3(*pos), r=None)
        #     gymutil.draw_lines(hip_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        
        for foothold in self.pred_footholds[i, :]:
            sphere_pose = gymapi.Transform(gymapi.Vec3(*foothold), r=None)
            gymutil.draw_lines(command_hip, self.gym, self.viewer, self.envs[i], sphere_pose) 

        
    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)
        
        # breakpoint()

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale


    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    # def _reward_orientation(self):
    #     # Penalize non flat base orientation
    #     return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        # base_height = self.root_states[:, 2]
        # return torch.square(base_height - self.cfg.rewards.base_height_target)

        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights[:, 55:132], dim = 1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)

        lin_vel_error = torch.sum(torch.square((self.commands[:, :2] - self.base_lin_vel[:, :2])/ self.command_ranges["lin_vel_x"][1]), dim=1)
        #! changed by wz
        # vel_flag = (self.base_lin_vel[:, :2]-self.commands[:, :2])<=0
        # vel_flag = (self.base_lin_vel[:, :2]-self.commands[:, :2])*self.commands[:, :2]<=0
        # lin_vel_error = torch.sum(torch.abs((self.commands[:, :2] - self.base_lin_vel[:, :2])*vel_flag), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    #! added by wz
    def _reward_tracking_lin_vel_2(self):
        delta_vel = self.base_lin_vel[:, :2] - self.base_ang_vel_last[:, :2]
        vel_positive_flag = (self.commands[:, :2]-self.base_lin_vel[:, :2]) >= 1.0
        vel_negative_flag = (self.commands[:, :2]-self.base_lin_vel[:, :2]) <= 1.0

        return torch.sum(torch.abs(delta_vel)*vel_positive_flag+ torch.abs(delta_vel)*vel_negative_flag, dim=1)

    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    #! added by wz
    def _reward_tracking_ang_vel_2(self):
        delta_ang_vel = self.base_ang_vel[:, 2] - self.base_ang_vel_last[:, 2]
        ang_vel_positive_flag = (self.commands[:, 2]-self.base_ang_vel[:, 2])>=0.5
        ang_vel_negative_flag = (self.commands[:, 2]-self.base_ang_vel[:, 2])<0.5

        return torch.abs(delta_ang_vel)*ang_vel_positive_flag+ torch.abs(delta_ang_vel)*ang_vel_negative_flag


    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        
        #! added by wz
        # contact_left_up = contact[:,0]
        # contact_right_up = contact[:,1]
        # contact_left_down = contact[:,2]
        # contact_right_down = contact[:,3]
        # contact_all = torch.logical_and(torch.logical_and(torch.logical_and(contact_left_up, contact_right_up),contact_left_down),contact_right_down)

        # contact_left_up_last =  self.last_contacts[:,0]
        # contact_right_up_last =  self.last_contacts[:,1]
        # contact_left_down_last =  self.last_contacts[:,1]
        # contact_right_down_last =  self.last_contacts[:,3]
        # contact_all_last = torch.logical_and(torch.logical_and(torch.logical_and(contact_left_up_last, contact_right_up_last),contact_left_down_last),contact_right_down_last)

        contact_left_up_filt = torch.logical_or(contact[:,0],self.last_contacts[:,0])
        contact_right_up_filt = torch.logical_or(contact[:,1],self.last_contacts[:,1])
        contact_left_down_filt = torch.logical_or(contact[:,2],self.last_contacts[:,2])
        contact_right_down_filt = torch.logical_or(contact[:,3],self.last_contacts[:,3])

        
        # contact_all_filt = torch.logical_or(contact_all, contact_all_last) 
        # contact_all_filt = torch.logical_and(contact_all, contact_all_last) 
        contact_all_filt =torch.logical_and(torch.logical_and(torch.logical_and(contact_left_up_filt, contact_right_up_filt),contact_left_down_filt),contact_right_down_filt)

        rew_airTime_stand = contact_all_filt * (torch.norm(self.commands[:, :2], dim=1) < 0.1) * 10.0
        #!!!!!!!!!!!!!!
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        # rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.2 #no reward for zero command
        self.feet_air_time *= ~contact_filt

        #! changed by wz !!!!!!!!!!!!!!
        return rew_airTime
        # return rew_airTime_stand
        # return rew_airTime+rew_airTime_stand
    
    
    
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    def get_base_vel(self):
        base_vel = self.base_lin_vel * self.obs_scales.lin_vel
        return base_vel
    

    #! added by wz
    def _reward_power(self):
        # return torch.sum(torch.abs(self.torques)*torch.abs(self.dof_vel), dim=1)
        return torch.sum(torch.clip(self.torques * self.dof_vel, min = 0), dim=1)
    
    #! added by wz
    def _reward_smooth(self):
        return torch.sum(torch.square(self.actions - 2*self.last_actions+self.last_actions_2), dim=1)
    

    #! DTC 
    # tracking optimal footholds
    def _reward_tracking_optimal_footholds(self):
        dis = self.foot_positions[:, :, :-1] - self.optimal_footholds_world[:, :, :-1]
        dis = torch.norm(dis, dim = -1)
        contact = self.contact_filt.float() # * 2 - 1.0, # 0:4
        epsilon = 0.8
        reward_per_foot = -torch.log(epsilon + dis)
        reward_filt = torch.where(contact == 1, reward_per_foot, torch.tensor(0., dtype = torch.float32, device=self.device))
        reward_sum = torch.sum(reward_filt, dim = -1)
        
        return reward_sum
        
        


    def get_plane_norm(self):
        # points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
        # print('p',points)
        # print(self.height_points)
        heights = self.measured_heights.clone()
        # print('h',heights)
        A = self.height_points.clone()
        A[:, :, 2] = 1
        A_T = A.transpose(1, 2)
        A1 = torch.bmm(A_T, A)
        A2 = torch.linalg.inv(A1)
        A3 = torch.bmm(A2, A_T)
        X = torch.bmm(A3, heights.unsqueeze(-1))
        Ax = X[:, 0, :]
        By = X[:, 1, :]
        Cz = -torch.ones_like(By)

        plane_vector = torch.cat([Ax, By, Cz], 1).reshape(-1, 3)
        plane_vector = plane_vector/torch.norm(plane_vector, dim = -1, keepdim = True)

        return plane_vector
    


    def _reward_orientation(self):
        p_norm = -self.get_plane_norm()


        pitch_est = torch.atan(p_norm[:,0])
        roll_est = -torch.atan(p_norm[:,1])
        # print(pitch_est)
        pitch_est_clip = torch.where((pitch_est >= -0.1)&(pitch_est <= 0.1), torch.tensor(0., dtype = torch.float32, device=self.device), pitch_est)
        roll_est_clip = torch.where((roll_est >= -0.1)&(roll_est <= 0.1), torch.tensor(0., dtype = torch.float32, device=self.device), roll_est)
        # print(pitch_est_clip)
        yaw_est = torch.zeros_like(roll_est)
        self.pitch_est = self.pitch_est.clone()*0.2+0.8*pitch_est_clip
        quat = quat_from_euler_xyz(roll_est_clip, self.pitch_est, yaw_est)


        p_norm_local = quat_rotate_inverse(quat, self.gravity_vec)

        return torch.sum(torch.square(self.projected_gravity[:, :1] - p_norm_local[:, :1]), dim=1)


    def _reward_orientation_roll(self):
        p_norm = -self.get_plane_norm()


        pitch_est = torch.atan(p_norm[:,0])
        roll_est = -torch.atan(p_norm[:,1])
        # print(pitch_est)
        pitch_est_clip = torch.where((pitch_est >= -0.1)&(pitch_est <= 0.1), torch.tensor(0., dtype = torch.float32, device=self.device), pitch_est)
        roll_est_clip = torch.where((roll_est >= -0.1)&(roll_est <= 0.1), torch.tensor(0., dtype = torch.float32, device=self.device), roll_est)
        # print(pitch_est_clip)
        yaw_est = torch.zeros_like(roll_est)
        self.pitch_est = self.pitch_est.clone()*0.2+0.8*pitch_est_clip
        quat = quat_from_euler_xyz(roll_est_clip, self.pitch_est, yaw_est)


        p_norm_local = quat_rotate_inverse(quat, self.gravity_vec)

        return torch.abs(self.projected_gravity[:, 1] - p_norm_local[:, 1])



    def _reward_pos_acc(self):
            result = [list(item) for item in itertools.product([-1, 1], repeat=3)]
            #! x30
            # acc_point = np.array(result) * [0.3, 0.2, 0.15]
            #! lite3
            acc_point = np.array(result) * [0.3, 0.2, 0.15]/2.0
            self.acc_point = to_torch(acc_point, device=self.device).view(1, 8, 3).repeat(self.num_envs, 1, 1)
            acc_point_vel = self.base_lin_vel.reshape(self.num_envs, 1, 3).repeat(1, 8, 1) + \
                                            torch.cross(
                                                self.base_ang_vel.reshape(self.num_envs, 1, 3).repeat(1, 8, 1),
                                                self.acc_point)  
            return  torch.sum(torch.square(torch.norm(acc_point_vel, dim=-1)), dim=1)
    
    def _reward_powerchange(self):
        # Penalize power
        target_v = self.commands[:,0].clone()
        smooth_co = target_v.clip(min=1.0)
        # return (torch.sum(torch.abs(self.torques)*torch.abs(self.dof_vel), dim=1)/(self.robot_mass * 9.815 * smooth_co))**2
        return (torch.sum((self.torques*self.dof_vel).clip(min=0.0), dim=1)/(self.robot_mass * 9.815 * smooth_co))**2