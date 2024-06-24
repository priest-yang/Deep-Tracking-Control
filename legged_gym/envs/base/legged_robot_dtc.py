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
import torchvision.transforms as T
import time
from .legged_robot_config import LeggedRobotCfg
from legged_gym.envs import LeggedRobot



class LeggedRobotDTC(LeggedRobot):
    cfg : LeggedRobotCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        

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
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 100., dim=1)
        
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf = self.time_out_buf
        # self.reset_buf |= (self.projected_gravity[:,2] > -0.2) 

        #! stand up
        self.reset_buf |= (self.projected_gravity[:,2] > 0.2)  
        #! X30
        # self.reset_buf |= ((torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights[:, 55:132], dim=1)) < 0.25)
        #! lite3
        
        self.reset_buf |= ((torch.mean(self.root_states[:, 2].unsqueeze(1) - 
                                       self.measured_heights[:, 10 * 21: (33-10)*21], 
                                       dim=1)) < 0.1) #! 55-132 changed according to terrain resolution
        
        self.reset_buf |= ((torch.mean(self.root_states[:, 2].unsqueeze(1) - 
                                       self.foot_positions[:, :, 2], 
                                       dim=1)) < 0.1) 
        

    def compute_observations(self):
        """ Computes observations
        """ 
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
                self.forces[:,0,:]* self.obs_scales.force,
                self.heights ,
            ), dim=1)

        # add noise if needed
        if self.add_noise:
            # print('self.noise_scale_vec: ',self.noise_scale_vec)
            # print('obs_buf',self.obs_buf)
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec[:53]
            # print('obs_buf_after',self.obs_buf)
            
            
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
            self.root_states[env_ids, :2] += torch_rand_float(-.5, .5, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
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

            
    

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        base_sphere_geom = gymutil.WireframeSphereGeometry(radius=0.1, color=(0, 0, 1))
        foothold_nominal_sphere_geom= gymutil.WireframeSphereGeometry(radius=0.035, color=(0, 0, 1))
        foothold_edge_sphere_geom = gymutil.WireframeSphereGeometry(radius=0.03, color=(1, 0, 0))
        foothold_optimal_sphere_geom = gymutil.WireframeSphereGeometry(radius=0.03, color=(0, 1, 0))
        command_hip = gymutil.WireframeSphereGeometry(radius=0.05, color=(1, 1, 0))
        
        # draw height lines
        if self.terrain.cfg.measure_heights:
            
            i = self.lookat_id

            x_all = self.heights_world[i, :, 0].cpu().numpy()
            y_all = self.heights_world[i, :, 1].cpu().numpy()
            z_all = self.heights_world[i, :, 2].cpu().numpy()
            
            foothold_score = torch.min(self.foothold_score, dim=2)[0][i, :].cpu()
            
            for x,y,z, score in zip(x_all, y_all, z_all, foothold_score):
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                if score > 0.9 and score < 8:
                    gymutil.draw_lines(foothold_edge_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
                # else:
                #     gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
                # gymutil.draw_lines(foothold_edge_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
            
            
            # nominal foot_hold
            nominal_footholds_indice = self.nominal_footholds_indice[i, :].cpu().numpy()
            nominal_footholds_x = x_all[nominal_footholds_indice]
            nominal_footholds_y = y_all[nominal_footholds_indice]
            nominal_footholds_z = z_all[nominal_footholds_indice]
            
            for x,y,z in zip(nominal_footholds_x, nominal_footholds_y, nominal_footholds_z):
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(foothold_nominal_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
            
            
            # draw optimal indices
            optimal_foothold_x = self.optimal_footholds_world[i, :, 0].cpu().numpy()
            optimal_foothold_y = self.optimal_footholds_world[i, :, 1].cpu().numpy()
            optimal_foothold_z = self.optimal_footholds_world[i, :, 2].cpu().numpy()
            
            for x,y,z in zip(optimal_foothold_x, optimal_foothold_y, optimal_foothold_z):
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(foothold_optimal_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        
        # hip position
        # for pos in self.hip_positions[i, :]:
        #     sphere_pose = gymapi.Transform(gymapi.Vec3(*pos), r=None)
        #     gymutil.draw_lines(hip_sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        
        for foothold in self.pred_footholds[i, :]:
            sphere_pose = gymapi.Transform(gymapi.Vec3(*foothold), r=None)
            gymutil.draw_lines(command_hip, self.gym, self.viewer, self.envs[i], sphere_pose) 

    
    #! DTC tracking optimal footholds reward
    def _reward_tracking_optimal_footholds(self):
        dis = self.foot_positions[:, :, :-1] - self.optimal_footholds_world[:, :, :-1]
        dis = torch.norm(dis, dim = -1)
        contact = self.contact_filt.float() # * 2 - 1.0, # 0:4
        epsilon = 0.8
        reward_per_foot = -torch.log(epsilon + dis)
        reward_filt = torch.where(contact == 1, reward_per_foot, torch.tensor(0., dtype = torch.float32, device=self.device))
        reward_sum = torch.sum(reward_filt, dim = -1)
        
        return reward_sum
    
    