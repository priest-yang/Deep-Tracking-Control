# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from rsl_rl.modules import ActorCritic_TS
from rsl_rl.modules import ActorCriticDecoder
from rsl_rl.storage import RolloutStorage
import itertools
from rsl_rl.utils import unpad_trajectories

class PPO:
    actor_critic: ActorCriticDecoder

    def __init__(self,
                 actor_critic,
                #  lidar_encoder, lidar_actor, lidar_vae,#! deleted by wz
                 num_learning_epochs=5,
                 num_mini_batches=4,
                 clip_param=0.2,
                 gamma=0.99,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.01,
                 learning_rate=5.e-4,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="adaptive",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        for name, param in self.actor_critic.named_parameters():
            print('name:', name)
            print('shape: ',param.shape)
        # self.optimizer = optim.Adam(itertools.chain(self.actor_critic.actor_body.parameters(),
        #                             self.actor_critic.critic_body.parameters(), {'params': self.actor_critic.std}), lr=learning_rate)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)  
        self.vae_optimizer = optim.Adam(self.actor_critic.vae.parameters(), lr=5.e-4)  
        
        #! deleted by wz
        # self.if_lidar = lidar_encoder != None
        # self.lidar_encoder = lidar_encoder
        # self.lidar_actor = lidar_actor
        # self.lidar_vae = lidar_vae
        # if self.if_lidar:
        #     self.lidar_actor_optimizer = optim.Adam([*self.lidar_actor.parameters(), *self.lidar_encoder.parameters(), * self.lidar_vae.parameters()], lr=learning_rate)
        #!!!!!!!!!!!!
            # self.lidar_actor_optimizer = optim.Adam(self.lidar_encoder.parameters(), lr=learning_rate)
        # self.vae_optimizer = optim.Adam(itertools.chain(self.actor_critic.cenet_encoder.parameters(),self.actor_critic.cenet_decoder.parameters(), 
        #                                             self.actor_critic.latent_mu.parameters(),self.actor_critic.latent_var.parameters()),lr=5.e-4)
        
        # self.adaptation_module_optimizer = optim.Adam(itertools.chain(self.actor_critic.memory_a.parameters(),
        #                                             self.actor_critic.ga_encoder.parameters(),self.actor_critic.ter_decoder.parameters(),
                                                    # self.actor_critic.ga_decoder.parameters(),self.actor_critic.gb_encoder.parameters(),
                                            # self.actor_critic.actor_student.parameters()),
                    # lr=self.learning_rate)
        # self.optimizer_ter = optim.Adam(itertools.chain(self.actor_critic.ter_decoder.parameters(),
        #                                                 self.actor_critic.terrain_encoder.parameters()), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        #tsnet
        self.num_adaptation_module_substeps = 1
    
    #temp to delete:
    def init_student_optimizer(self):
        # self.adaptation_module_optimizer = optim.Adam(itertools.chain(self.actor_critic.memory_a.parameters(),self.actor_critic.adaptation_encoder.parameters(),
        #                                                               self.actor_critic.actor_student.parameters()),
        #                                         lr=self.learning_rate)
        # self.adaptation_module_optimizer = optim.Adam(itertools.chain(self.actor_critic.memory_a.parameters(),
        #                                                         self.actor_critic.ga_encoder.parameters(),self.actor_critic.ter_decoder.parameters(),
                                                                # self.actor_critic.ga_decoder.parameters(),self.actor_critic.gb_encoder.parameters(),
                                                        # self.actor_critic.actor_student.parameters()),
                                # lr=self.learning_rate)
        self.actor_critic.to(self.device)
        
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, obs_history_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, privileged_obs, obs_history,base_vel, rew_buf):#todo
        # if self.actor_critic.is_recurrent: #  lstm todo
        #     self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, obs_history, privileged_obs, rew_buf).detach()

        # self.actor_critic.act_student(obs,obs_history)

        self.transition.values = self.actor_critic.evaluate(obs, privileged_obs,base_vel).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = obs
        self.transition.privileged_observations = privileged_obs
        self.transition.observation_histories = obs_history
        self.transition.base_vel = base_vel 
        return self.transition.actions
    
    def process_env_step(self, rewards, dones,next_obs, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_observations = next_obs
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs, last_critic_privileged_obs,last_base_vel):
        last_values= self.actor_critic.evaluate(last_critic_obs, last_critic_privileged_obs,last_base_vel).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_adaptation_module_loss = 0
        mean_entropy_loss = 0
        mean_recons_loss = 0
        mean_vel_loss = 0
        mean_kld_loss = 0
        mean_height_loss = 0
        loss_decoder = 0 
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, privileged_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch,base_vel_batch,next_obs_batch, hid_states_batch, masks_batch,rew_buf_batch in generator:
                ####################################for teacher training ##################################################
            #! height
            # latent_mu, latent_var, z, height_latent = self.actor_critic.vae.cenet_forward(obs_history_batch, privileged_obs_batch[..., 0:693])
            # recons = self.actor_critic.vae.cenet_decoder(torch.cat([z, latent_mu[:,:3], height_latent], dim = 1))

            #! changed by wz 2
            latent_mu, latent_var, z = self.actor_critic.vae.cenet_forward(obs_history_batch)
            
            l_t = self.actor_critic.vae.terrain_encoder(privileged_obs_batch[:,:693])
            
            recons = self.actor_critic.vae.cenet_decoder(torch.cat([z, latent_mu[:,:3],l_t], dim = 1))
            # recons = self.actor_critic.vae.cenet_decoder(latent_mu)
            

           
            # print('a:' ,actions_batch.size())
            # print('b: ', base_vel_batch.size())
            # print('b: ', obs_history_batch.size())
            # print('delta:',(recons - next_obs_batch).mean(0))
            # print('latent',self.actor_critic.latent)
            # delta_recon = torch.cat(((recons - next_obs_batch)[:,:33], 0.5*(recons - next_obs_batch)[:,33:]),dim = 1)
            # recons_loss =torch.pow((recons - next_obs_batch),2).mean(-1).mean()#F.mse_loss(recons, next_obs_batch)
            delta_recon = recons - next_obs_batch
            #! next_obs loss
            recons_loss =torch.pow(delta_recon,2).mean(-1).mean()

            #! height
            height_recon = self.actor_critic.vae.terrain_decoder(l_t)

            #! terrian loss

            #! height
            height_loss = F.mse_loss(height_recon, privileged_obs_batch[..., 693+3:])


            # print('now: ', recons[0])
            # print('next: ', next_obs_batch[0])
            # print('delta1: ', (recons - next_obs_batch)[0])
            # # print('delta3: ', (recons - next_obs_batch).pow(2.)[0])
            # print('delta4: ', (torch.pow((recons - next_obs_batch),2.))[0])
            # recons_loss = F.mse_loss(recons, next_obs_batch)
            # print('loss1: ',recons_loss)
            # print('loss2: ', (recons - next_obs_batch).pow(2).mean(-1).mean())
            # print((recons - next_obs_batch).pow(2).mean(-1).size())
            #! vel loss
            vel_loss = F.mse_loss(latent_mu[:,:3], base_vel_batch)

            #! kl loss
            # kld_loss = torch.mean(-0.5 * torch.sum(1 + latent_var - latent_mu[:,3:].pow(2) - latent_var.exp(), dim = 1))
            #! changed by wz 2
            kld_loss = torch.mean(-0.5 * torch.sum(1 + latent_var - latent_mu[:,3:].pow(2) - latent_var.exp(), dim = 1))

            #! height
            # vae_loss = recons_loss+vel_loss + 4*kld_loss + height_loss

            #! changed by wz 2
            vae_loss = recons_loss+vel_loss+ 4*kld_loss + height_loss



            self.vae_optimizer.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
            vae_loss.backward()

            nn.utils.clip_grad_norm_(self.actor_critic.vae.parameters(), self.max_grad_norm)
            self.vae_optimizer.step()
            
            mean_recons_loss += recons_loss.item()
            mean_vel_loss += vel_loss.item()
            mean_kld_loss += kld_loss.item()

            # mean_kld_loss += 0

            #! height
            # mean_height_loss += height_loss.item()
            
            self.actor_critic.act(obs_batch, obs_history_batch, privileged_obs_batch, rew_buf_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            ####auto decoder test:
            # latent_e = self.actor_critic.terrain_encoder(privileged_obs_batch[:,:216])
            # height_decoder = self.actor_critic.ter_decoder(latent_e)
            # loss_decoder = F.mse_loss(privileged_obs_batch[:,249:249+216], height_decoder)
            
            # encoded = self.actor_critic.cenet_encoder(obs_history_batch)
            # latent_var = self.actor_critic.latent_var(encoded)
            # vel_var = self.actor_critic.vel_var(encoded)
            # latent_mu = self.actor_critic.latent_mu(encoded)
            # vel_mu = self.actor_critic.vel_mu(encoded)
            # vel = self.actor_critic.reparameterize(self.actor_critic.vel_mu_, self.actor_critic.vel_var_)
            # latent = self.actor_critic.reparameterize(self.actor_critic.latent_mu_, self.actor_critic.latent_var_)
            #try without noise and use one optimizer
                    # print('kld_loss: ', -0.5 * torch.sum(1 + self.actor_critic.latent_var_ - self.actor_critic.latent_mu_.pow(2) - self.actor_critic.latent_var_.exp(), dim = 1))
            # print('lll:', -0.5 * torch.sum(1 + self.actor_critic.latent_var_ - self.actor_critic.latent_mu_.pow(2) - self.actor_critic.latent_var_.exp(), dim = 1))
            # print('var: ', self.actor_critic.latent_var_ )
            # print('mu: ', self.actor_critic.latent_mu_.pow(2))
            # print('vare: ', self.actor_critic.latent_var_.exp())
            # print('kld_loss',-0.5 * torch.sum(1 + self.actor_critic.latent_var_ - self.actor_critic.latent_mu_ ** 2 - self.actor_critic.latent_var_.exp(),dim = 0))
            ####
            

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, privileged_obs_batch,base_vel_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl != None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate


            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                            1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()#+ \
                #    recons_loss+vel_loss + kld_loss
                #    loss_decoder*0.05
                    
                    

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy_loss += entropy_batch.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy_loss /= num_updates
        # loss_decoder /= num_updates
        mean_adaptation_module_loss /= (num_updates * self.num_adaptation_module_substeps)
        mean_recons_loss /= (num_updates * self.num_adaptation_module_substeps)
        mean_vel_loss /= (num_updates * self.num_adaptation_module_substeps)
        mean_kld_loss /= (num_updates * self.num_adaptation_module_substeps)
        mean_height_loss /= (num_updates * self.num_adaptation_module_substeps)
        # mean_adaptation_module_loss /= (self.num_learning_epochs * self.num_adaptation_module_substeps)
        self.storage.clear()

        # return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss ,loss_decoder, \
        #         mean_recons_loss, mean_vel_loss, mean_kld_loss, mean_height_loss
        #! changed by wz
        return mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss ,loss_decoder, \
                mean_recons_loss, mean_vel_loss, mean_kld_loss



    def update_student(self,height_map_buffer, lidar_recon_buffer, actions_student_buffer,actions_teacher_buffer):

        loss = 0
        teri_loss = 0
        actor_loss = 0
        teri_loss = F.mse_loss(height_map_buffer.detach(), lidar_recon_buffer)
        actor_loss = F.mse_loss(actions_student_buffer, actions_teacher_buffer.detach())
        # teri_loss = lidar_recon_buffer.norm()
        # actor_loss = actions_student_buffer.norm()
        loss = teri_loss + 0.1*actor_loss
        self.lidar_actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_([*self.lidar_actor.parameters(), *self.lidar_encoder.parameters(), * self.lidar_vae.parameters()], self.max_grad_norm)
        # nn.utils.clip_grad_norm_(self.lidar_encoder.parameters(), self.max_grad_norm)
        self.lidar_actor_optimizer.step()
        # encoder_loss = teri_loss
        # self.adaptation_module_optimizer.zero_grad()
        # depth_loss.backward()
        # self.adaptation_module_optimizer.step()

        return teri_loss.item(),actor_loss.item()
