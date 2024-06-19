import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from params_proto.proto import PrefixProto
from rsl_rl.utils import unpad_trajectories
import copy
import torch.nn.functional as F
class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    # actor_hidden_dims = [256, 128, 64]
    actor_hidden_dims = [512, 256, 128]#paper
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    adaptation_encoder_branch_input_dims = [100]# lstm output: 256  add depth image:256+173  
    adaptation_encoder_branch_hidden_dims = [[64, 64]]
    adaptation_encoder_branch_latent_dims = [36] # original: 16+8
    adaptation_encoder_branch_latent_dims1 = [36]

    privileged_encoder_branch_input_dims = [15 +12 - 24]
    privileged_encoder_branch_latent_dims = [12]
    privileged_encoder_branch_hidden_dims = [[64, 32]]

    terrain_encoder_branch_input_dims = [693]# original: 693
    terrain_encoder_branch_latent_dims = [64]# original: 16
    terrain_encoder_branch_hidden_dims = [[128, 64]]

    terrain_decoder_branch_input_dims = [64]# original: 693
    terrain_decoder_branch_output_dims = [693]# original: 16
    terrain_decoder_branch_hidden_dims = [[64, 128]]
    #CENet
    cenet_encoder_branch_input_dims = [53*5]# 45*5
    cenet_encoder_branch_latent_dims = [64]# original: 16
    cenet_encoder_branch_hidden_dims = [[128]]

    cenet_decoder_branch_input_dims = [19+64]
    cenet_decoder_branch_output_dims = [53]#obs_next
    cenet_decoder_branch_hidden_dims = [[64, 128]]
    
    Lidar_encoder_branch_input_dims = [16000]
    Lidar_encoder_branch_latent_dims = [64]
    Lidar_encoder_branch_hidden_dims = [[512, 256, 128]]
    
    Lidar_decoder_branch_input_dims = [64]
    Lidar_decoder_branch_latent_dims = [693]
    Lidar_decoder_branch_hidden_dims = [[256]]


    # temp to test:
    ga_encoder_input_dims = [128]
    ga_encoder_hidden_dims =  [[128]]
    ga_encoder_latent_dims = [64]

    ga_decoder_input_dims = [64]
    ga_decoder_hidden_dims = [[128]]
    ga_decoder_latent_dims = [693] 

    gb_encoder_input_dims = [128]
    gb_encoder_hidden_dims = [[128]] 
    gb_encoder_latent_dims = [64]

    # gb_decoder_input_dims = [693]
    # gb_decoder_hidden_dims = [[128, 64]] 
    # gb_decoder_latent_dims = [64]

    memory_mlp_input_dims = [53*5 + 64]
    memory_mlp_hidden_dims = [[128, 64]]
    memory_mlp_latent_dims = [64] 

    #LSTM todo
    rnn_type = 'gru'#lstm
    rnn_num_layers = 2
    rnn_hidden_size = 50


class Vae(nn.Module):
    def __init__(self, 
                **kwargs):

        super().__init__()

        activation = nn.ReLU()
        for i, (branch_input_dim, branch_hidden_dims, branch_latent_dim) in enumerate(
            zip(AC_Args.cenet_encoder_branch_input_dims,
                AC_Args.cenet_encoder_branch_hidden_dims,
                AC_Args.cenet_encoder_branch_latent_dims)):
        # Env factor encoder
            env_factor_encoder_layers = []
            env_factor_encoder_layers.append(nn.Linear(branch_input_dim, branch_hidden_dims[0]))
            env_factor_encoder_layers.append(activation)
            for l in range(len(branch_hidden_dims)):
                if l == len(branch_hidden_dims) - 1:
                    env_factor_encoder_layers.append(self.layer_init(
                        nn.Linear(branch_hidden_dims[l], branch_latent_dim),0.01))
                else:
                    env_factor_encoder_layers.append(self.layer_init(
                        nn.Linear(branch_hidden_dims[l],
                                    branch_hidden_dims[l + 1]),0.01))
                    env_factor_encoder_layers.append(activation)
            self.cenet_encoder = nn.Sequential(*env_factor_encoder_layers)
            self.add_module(f"cenet_encoder", self.cenet_encoder)

        self.latent_mu = self.layer_init(nn.Linear(16 * 4, 19),0.01)
        self.latent_var = self.layer_init(nn.Linear(16 * 4, 16),0.01)

        # self.vel_mu = nn.Linear(16 * 4, 3)
        # self.vel_var = nn.Linear(16 * 4, 3)
        
        for i, (branch_input_dim, branch_hidden_dims, branch_latent_dim) in enumerate(
                zip(AC_Args.cenet_decoder_branch_input_dims,
                    AC_Args.cenet_decoder_branch_hidden_dims,
                    AC_Args.cenet_decoder_branch_output_dims)):
            # Env factor encoder
            env_factor_encoder_layers = []
            print('branch_input_dim = ',branch_input_dim)#dim=19
            env_factor_encoder_layers.append(nn.Linear(branch_input_dim, branch_hidden_dims[0]))
            env_factor_encoder_layers.append(activation)
            for l in range(len(branch_hidden_dims)):
                if l == len(branch_hidden_dims) - 1:
                    env_factor_encoder_layers.append(self.layer_init(
                        nn.Linear(branch_hidden_dims[l], branch_latent_dim),0.01))
                else:
                    env_factor_encoder_layers.append(self.layer_init(
                        nn.Linear(branch_hidden_dims[l],
                                  branch_hidden_dims[l + 1]),0.01))
                    env_factor_encoder_layers.append(activation)
        self.cenet_decoder = nn.Sequential(*env_factor_encoder_layers)
        self.add_module(f"cenet_decoder", self.cenet_decoder)
        
        for i, (branch_input_dim, branch_hidden_dims, branch_latent_dim) in enumerate(
                zip(AC_Args.terrain_encoder_branch_input_dims,
                    AC_Args.terrain_encoder_branch_hidden_dims,
                    AC_Args.terrain_encoder_branch_latent_dims)):
            # Env factor encoder
            env_factor_encoder_layers = []
            print('branch_input_dim = ',branch_input_dim)#dim=19
            env_factor_encoder_layers.append(nn.Linear(branch_input_dim, branch_hidden_dims[0]))
            env_factor_encoder_layers.append(activation)
            for l in range(len(branch_hidden_dims)):
                if l == len(branch_hidden_dims) - 1:
                    env_factor_encoder_layers.append(self.layer_init(
                        nn.Linear(branch_hidden_dims[l], branch_latent_dim),0.01))
                else:
                    env_factor_encoder_layers.append(self.layer_init(
                        nn.Linear(branch_hidden_dims[l],
                                  branch_hidden_dims[l + 1]),0.01))
                    env_factor_encoder_layers.append(activation)
        self.terrain_encoder = nn.Sequential(*env_factor_encoder_layers)
        self.add_module(f"terrain_encoder", self.terrain_encoder)




        for i, (branch_input_dim, branch_hidden_dims, branch_latent_dim) in enumerate(
                zip(AC_Args.terrain_decoder_branch_input_dims,
                    AC_Args.terrain_decoder_branch_hidden_dims,
                    AC_Args.terrain_decoder_branch_output_dims)):
            # Env factor encoder
            env_factor_encoder_layers = []
            print('branch_input_dim = ',branch_input_dim)#dim=19
            env_factor_encoder_layers.append(nn.Linear(branch_input_dim, branch_hidden_dims[0]))
            env_factor_encoder_layers.append(activation)
            for l in range(len(branch_hidden_dims)):
                if l == len(branch_hidden_dims) - 1:
                    env_factor_encoder_layers.append(self.layer_init(
                        nn.Linear(branch_hidden_dims[l], branch_latent_dim),0.01))
                else:
                    env_factor_encoder_layers.append(self.layer_init(
                        nn.Linear(branch_hidden_dims[l],
                                  branch_hidden_dims[l + 1]),0.01))
                    env_factor_encoder_layers.append(activation)
        self.terrain_decoder = nn.Sequential(*env_factor_encoder_layers)
        self.add_module(f"terrain_decoder", self.terrain_decoder)

        for i, (branch_input_dim, branch_hidden_dims, branch_latent_dim) in enumerate(
                zip(AC_Args.memory_mlp_input_dims,
                    AC_Args.memory_mlp_hidden_dims,
                    AC_Args.memory_mlp_latent_dims)):
            # Env factor encoder
            env_factor_encoder_layers = []
            print('branch_input_dim = ',branch_input_dim)#dim=19
            env_factor_encoder_layers.append(nn.Linear(branch_input_dim, branch_hidden_dims[0]))
            env_factor_encoder_layers.append(activation)
            for l in range(len(branch_hidden_dims)):
                if l == len(branch_hidden_dims) - 1:
                    env_factor_encoder_layers.append(self.layer_init(
                        nn.Linear(branch_hidden_dims[l], branch_latent_dim),0.01))
                else:
                    env_factor_encoder_layers.append(self.layer_init(
                        nn.Linear(branch_hidden_dims[l],
                                  branch_hidden_dims[l + 1]),0.01))
                    env_factor_encoder_layers.append(activation)
        self.memory_mlp = nn.Sequential(*env_factor_encoder_layers)
        self.add_module(f"memory_mlp", self.memory_mlp)

        # Adaptation module todo lstm
        for i, (branch_input_dim, branch_hidden_dims, branch_latent_dim) in enumerate(
                zip(AC_Args.ga_decoder_input_dims,
                    AC_Args.ga_decoder_hidden_dims,
                    AC_Args.ga_decoder_latent_dims)):
            # Env factor encoder
            env_factor_encoder_layers = []
            env_factor_encoder_layers.append(nn.Linear(branch_input_dim, branch_hidden_dims[0]))
            env_factor_encoder_layers.append(activation)
            for l in range(len(branch_hidden_dims)):
                if l == len(branch_hidden_dims) - 1:
                    env_factor_encoder_layers.append(self.layer_init(
                        nn.Linear(branch_hidden_dims[l], branch_latent_dim),0.01))
                else:
                    env_factor_encoder_layers.append(self.layer_init(
                        nn.Linear(branch_hidden_dims[l],
                                  branch_hidden_dims[l + 1]),0.01))
                    env_factor_encoder_layers.append(activation)
            sig = get_activation('sigmoid')
            env_factor_encoder_layers.append(sig)

       # self.ga_decoder = nn.Sequential(*env_factor_encoder_layers)
        #self.add_module(f"ga_decoder", self.ga_decoder)

        # Adaptation module todo lstm
        for i, (branch_input_dim, branch_hidden_dims, branch_latent_dim) in enumerate(
                zip(AC_Args.gb_encoder_input_dims,
                    AC_Args.gb_encoder_hidden_dims,
                    AC_Args.gb_encoder_latent_dims)):
            # Env factor encoder
            env_factor_encoder_layers = []
            env_factor_encoder_layers.append(nn.Linear(branch_input_dim, branch_hidden_dims[0]))
            env_factor_encoder_layers.append(activation)
            for l in range(len(branch_hidden_dims)):
                if l == len(branch_hidden_dims) - 1:
                    env_factor_encoder_layers.append(self.layer_init(
                        nn.Linear(branch_hidden_dims[l], branch_latent_dim),0.01))
                else:
                    env_factor_encoder_layers.append(self.layer_init(
                        nn.Linear(branch_hidden_dims[l],
                                  branch_hidden_dims[l + 1]),0.01))
                    env_factor_encoder_layers.append(activation)
        self.gb_encoder = nn.Sequential(*env_factor_encoder_layers)
        self.add_module(f"gb_encoder", self.gb_encoder)


        print(f"cenet_encoder Module: {self.cenet_encoder}")
        print(f"cenet_decoder Module: {self.cenet_decoder}")
        print(f"terrain_encoder Module: {self.terrain_encoder}")
        print(f"terrain_decoder Module: {self.terrain_decoder}")

        #print(f"ga_decoder Module: {self.ga_decoder}")
        # print(f"ga_encoder Module: {self.ga_encoder}")
        print(f"gb_encoder Module: {self.gb_encoder}")
        # print(f"gb_decoder Module: {self.gb_decoder}")

    #! added by wz
    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def cenet_forward(self, observations_history):
        latent_e = self.cenet_encoder(observations_history)#64
        # vel_var = self.vel_var(latent_e)
        latent_var = self.latent_var(latent_e)
        # vel_mu = self.vel_mu(latent_e)
        latent_mu = self.latent_mu(latent_e)
        # height_latent = self.terrain_encoder(height_map)
        
        # self.vae.latent_mu_, self.vae.latent_var_ = self.vae.cenet_forward(observations_history)
        # self.vel = self.reparameterize(self.latent_mu_[:,:3], self.latent_var_[:,:3])
        z = self.reparameterize(latent_mu[:,3:], latent_var)
        return latent_mu, latent_var, z#, height_latent
        #return self.latent_distribution.sample()

class ActorCriticDecoder(nn.Module):
    is_recurrent = False  #  teacher:False  student:true

    def __init__(self, num_obs,
                num_critic_obs,
                num_actions,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super().__init__()

        activation = get_activation(AC_Args.activation)
        num_actions = num_actions  #PMTrajectoryGenerator test:###
        self.vae = Vae()
        self.bootstrap_threshold = 0.1

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(AC_Args.cenet_decoder_branch_input_dims[0] + num_obs, AC_Args.actor_hidden_dims[0]))#19+45
        actor_layers.append(activation)
        for l in range(len(AC_Args.actor_hidden_dims)):
            if l == len(AC_Args.actor_hidden_dims) - 1:
                actor_layers.append(self.layer_init(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions),0.01))
            else:
                actor_layers.append(self.layer_init(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]),0.01))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)
        self.add_module(f"actor_body", self.actor_body)




        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(693 + num_obs +3+ 15 +12 - 24 , AC_Args.critic_hidden_dims[0])) #todo :try latent dim = 36
        critic_layers.append(activation)
        for l in range(len(AC_Args.critic_hidden_dims)):
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(self.layer_init(nn.Linear(AC_Args.critic_hidden_dims[l], 1),0.01))
            else:
                critic_layers.append(self.layer_init(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]),0.01))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)
        self.add_module(f"critic_body", self.critic_body)

        # print(f"privileged_encoder Encoder: {self.privileged_encoder}")

        # print(f"terrain_encoder Encoder: {self.terrain_encoder}")
        # print(f"pri_decoder Module: {self.pri_decoder}")
        # print(f"ter_decoder Module: {self.ter_decoder}")
        # print(f"ga_encoder Module: {self.ga_encoder}")
        # print(f"gb_encoder Module: {self.gb_encoder}")
        print(f"Actor MLP: {self.actor_body}")
        # print(f"student MLP: {self.actor_student}")
        print(f"Critic MLP: {self.critic_body}")
        # print(f"student lstm: {self.memory_a}")

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self.latent_distribution = None
        self.count = 0
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    #! added by wz
    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
    
    def define_student_temp(self):
        num_obs = 45  #45
        loaded_dict = torch.load('/home/ysc/gym/ours/teacher_student/lite3_ts_isaac-main/logs/rough_lite3/Nov09_13-57-19_/model_25000.pt')
        
        
        

    def reset(self, dones=None):
        # self.memory_a.reset(dones)
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def adapt_bootstrap_probability(self, rewards):
        cv = torch.std(rewards) / torch.mean(rewards)
        bootstrap_prob = 1 - torch.tanh(cv)
        return bootstrap_prob.item()
    
    def update_distribution(self, observations, observations_history, privileged_obs):
        # latent_e = self.terrain_encoder(privileged_observations[:, 0:AC_Args.terrain_encoder_branch_input_dims[0]])
        # latent_p = self.privileged_encoder(privileged_observations[:, AC_Args.terrain_encoder_branch_input_dims[0]:
        #                                                         AC_Args.terrain_encoder_branch_input_dims[0]+AC_Args.privileged_encoder_branch_input_dims[0]])
        # latent = torch.cat((latent_e, latent_p), dim=-1)
        # latent = latent_e
        # latent_e = self.cenet_encoder(observations_history)#64
        # vel_var = self.vel_var(latent_e)
        # latent_var = self.latent_var(latent_e)
        # vel_mu = self.vel_mu(latent_e)
        # latent_mu = self.latent_mu(latent_e)
        # vel = self.reparameterize(vel_mu, vel_var)
        # latent = self.reparameterize(latent_mu, latent_var)
 
        # self.latent_distribution = Normal(self.latent_mu_, torch.exp(0.5 * self.latent_var_))

                
        self.latent_mu, self.latent_var, self.z = self.vae.cenet_forward(observations_history)

        #! add the belief encoder   
        l_t = self.vae.terrain_encoder(privileged_obs[:,:693])

        mean = self.actor_body(torch.cat((observations,self.z,self.latent_mu[:,:3], l_t), dim=-1))

        # mean_before = self.actor_body(torch.cat((observations,self.z,self.latent_mu[:,:3]), dim=-1))
        # mean_before2 = torch.where(torch.isfinite(mean_before.clone()),mean_before.clone(),torch.ones_like(mean_before.clone()))
        # mean = torch.clip(mean_before2,min=-100.,max=100.)
        # self.std = nn.Parameter(self.std).cuda(0)
        self.distribution = Normal(mean, mean * 0. + self.std)
    
    def act(self, observations, observations_history, privileged_obs, rew_buf, **kwargs):
        self.update_distribution(observations, observations_history, privileged_obs)

        # bootstrap_prob = self.adapt_bootstrap_probability(rew_buf)
        # if bootstrap_prob > self.bootstrap_threshold:
        #     return self.distribution.sample()
        # else:
        #     return self.distribution.mean
        return self.distribution.sample()
        # return self.action
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_expert(self, ob):
        return self.act_teacher(ob["obs"], ob["obs_history"], ob["privileged_obs"])

    def act_inference(self, ob):
        return self.act_student(ob["obs"], ob["privileged_obs"])

    def act_student(self, observations, observations_history, privileged_obs, lidar_latent, masks=None, hidden_states=None): 
        # observations = observations[:,:45]
        # h_t = self.memory_a(observations, masks, hidden_states)

        # height_map = privileged_obs[:,:693] #+ (2 * torch.rand_like(privileged_obs[:,:216]) - 1) * 0.2
        # l_t = self.terrain_encoder(height_map)

        # b_t1 = self.memory_a(torch.cat((observations[:,:6].clone(),observations[:,9:45].clone(),l_t),dim=-1))
        # alpha = self.ga_encoder(b_t1.squeeze(0))
        # b_t = self.gb_encoder(b_t1.squeeze(0)) + torch.mul(l_t , alpha)
        # self.latent_mu, self.latent_var, self.z, self.height_latent = self.vae.cenet_forward(observations_history, privileged_obs[..., 190:])
        latent_e = self.cenet_encoder(observations_history)#64
        # vel_var = self.vel_var(latent_e)
        # latent_var = self.latent_var(latent_e)
        # vel_mu = self.vel_mu(latent_e)
        latent_mu = self.latent_mu(latent_e)
        # height_latent = self.terrain_encoder(height_map)
        actions_mean = self.actor_student(torch.cat((observations,latent_mu[:,3:],latent_mu[:,:3], lidar_latent), dim=-1))
        
        # actions_mean = self.actor_student(torch.cat((observations[:,:45].clone(), b_t), dim=-1))

        # print(privileged_obs[0,:216])

        
        if self.count == 0:#need to delete while running 
            self.exporter = PolicyExporter(self.memory_a,self.terrain_encoder,self.actor_student,self.ga_encoder,self.gb_encoder)
            self.exporter.export(torch.cat((observations[:,:45],privileged_obs[:,:693]),dim=-1))
            loaded_dict = torch.load('/home/ysc/gym/ours/teacher_student/lite3_ts_isaac-main/velocity_estimator/policy_student_v3.4.pt')

            self.exporter.load_state_dict(loaded_dict.state_dict())
            self.count =self.count+1

            # for n,p in self.adaptation_encoder.named_parameters():
            #     print(n,':',p)
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # for n,p in exporter.adaptation_encoder.named_parameters():
            #     print(n,':',p)
        action_test = self.exporter(torch.cat((observations[:,:45],privileged_obs[:,:693]),dim=-1).cpu())
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        # print('actions_mean:',actions_mean)
        # print('action_test:',action_test)


        return  actions_mean

    def act_teacher(self, observations, observations_history, privileged_obs):
        # latent_e = self.terrain_encoder(privileged_observations[:, 0:AC_Args.terrain_encoder_branch_input_dims[0]])
        # latent = latent_e
        # latent = self.cenet_encoder(observations_history)#19



        latent_e = self.vae.cenet_encoder(observations_history)#64
        # vel = self.vel_mu(latent_e)
        latent = self.vae.latent_mu(latent_e)
        #! add the  belief encoder   
        l_t = self.vae.terrain_encoder(privileged_obs[:,:693])
        b_t1 = self.vae.memory_mlp(torch.cat((observations_history, l_t),dim=-1))
        # alpha = self.vae.ga_encoder(b_t1)
        b_t = b_t1 + torch.mul(l_t , b_t1)
        # height_latent = self.vae.terrain_encoder(privileged_obs[:, 0:693])
        actions_mean = self.actor_body(torch.cat((observations,latent[:,3:],latent[:,:3], b_t), dim=-1))
        obs_total = torch.cat((observations,observations_history), dim=-1)
        # if self.count == 0:#need to delete while running 
        #     self.exporter1 = TerrainEncoder(self.vae.terrain_encoder)
        #     self.exporter1.export(privileged_obs[..., 0:693])
        #     latent_lidar = self.exporter1(privileged_obs[..., 0:693].cpu())
        #     loaded_dict1 = torch.load('/media/ysc/C7FD28394D2A63C9/ysc/LearningCode/lite3_ts_isaac-main/policy/terrain_stair_H_v2.9.pt')
        #     self.exporter1.load_state_dict(loaded_dict1.state_dict())

        #     self.exporter = PolicyExporter_export(self.vae.cenet_encoder,self.actor_body,self.vae.latent_mu,self.vae.memory_mlp)
        #     self.exporter.export(obs_total,latent_lidar)
        #     loaded_dict = torch.load('/media/ysc/C7FD28394D2A63C9/ysc/LearningCode/lite3_ts_isaac-main/policy/policy_x30_H_v2.9.pt')
        #     self.exporter.load_state_dict(loaded_dict.state_dict())
        #     self.count =self.count+1

        # latent_lidar_test = self.exporter1(privileged_obs[..., 0:693].cpu())
        # action_test = self.exporter(obs_total.cpu(),latent_lidar_test)

        return actions_mean
    
    def evaluate(self, critic_observations, privileged_observations,base_vel, **kwargs):
        # latent_e = self.terrain_encoder(privileged_observations[:, 0:AC_Args.terrain_encoder_branch_input_dims[0]])
        # latent_p = self.privileged_encoder(privileged_observations[:, AC_Args.terrain_encoder_branch_input_dims[0]:
        #                                                         AC_Args.terrain_encoder_branch_input_dims[0]+AC_Args.privileged_encoder_branch_input_dims[0]])
        # latent = torch.cat((latent_e, privileged_observations[:, AC_Args.terrain_encoder_branch_input_dims[0]: AC_Args.terrain_encoder_branch_input_dims[0]+AC_Args.privileged_encoder_branch_input_dims[0]]), dim=-1)
        # latent = latent_e

        # latent_e = self.cenet_encoder(observations_history)
        # vel = self.vel_var(latent_e)
        vel = base_vel
        value = self.critic_body(torch.cat((critic_observations,vel,privileged_observations[:, 693:693+3],privileged_observations[:, 693+3:]), dim=-1))
        return value
    
    def act_privilege(self, privileged_observations):#no use
        latent_e = self.terrain_encoder(privileged_observations[:, 0:AC_Args.terrain_encoder_branch_input_dims[0]])
        # latent_p = self.privileged_encoder(privileged_observations[:, AC_Args.terrain_encoder_branch_input_dims[0]:
        #                                                         AC_Args.terrain_encoder_branch_input_dims[0]+AC_Args.privileged_encoder_branch_input_dims[0]])
        # latent = torch.cat((latent_e, latent_p), dim=-1)   
        latent = latent_e

        return latent     
    
    def get_hidden_states(self):
        return self.memory_a.hidden_states
    
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
    
class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
    
    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            if self.hidden_states is not None:
                self.hidden_states.detach_()
            
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
            # self.hidden_states[0].detach_()
            # self.hidden_states[1].detach_()
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        if self.hidden_states is not None:
            for hidden_state in self.hidden_states:
                hidden_state[..., dones, :] = 0.0

import os

class TerrainEncoder(torch.nn.Module):
    def __init__(self, terrain_encoder):
        super().__init__()
        self.terrain_encoder = copy.deepcopy(terrain_encoder)
        self.terrain_encoder.cpu()

    def forward(self, observations):
        latent_lidar = self.terrain_encoder(observations)
        return latent_lidar

    def export(self, observations):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        traced_script_module = torch.jit.script(self,observations.cpu())
        traced_script_module.save('/media/ysc/C7FD28394D2A63C9/ysc/LearningCode/lite3_ts_isaac-main/policy/terrain_stair_H_v2.9.pt')


class PolicyExporter_export(torch.nn.Module):
    def __init__(self, cenet_encoder, actor_body,latent_mu,memory_mlp): 
        super().__init__()

        self.cenet_encoder = copy.deepcopy(cenet_encoder)
        self.actor_body = copy.deepcopy(actor_body)
        self.latent_mu = copy.deepcopy(latent_mu)
        self.memory_mlp = copy.deepcopy(memory_mlp)
        self.cenet_encoder.cpu()
        self.actor_body.cpu()
        self.latent_mu.cpu()
        self.memory_mlp.cpu()

    def forward(self, observations_total,latent_lidar):

        latent_e = self.cenet_encoder(observations_total[:,45:])#64
        # vel = self.vel_mu(latent_e)
        latent = self.latent_mu(latent_e)
        b_t1 = self.memory_mlp(torch.cat((observations_total[:,45:], latent_lidar),dim=-1))
        b_t = b_t1 + torch.mul(latent_lidar , b_t1)
        actions_mean = self.actor_body(torch.cat((observations_total[:,0:45],latent[:,3:],latent[:,:3], b_t), dim=-1))

        return actions_mean


    def export(self, observations_total,latent_lidar):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        traced_script_module = torch.jit.script(self,torch.cat((observations_total.cpu(), latent_lidar.cpu()), dim=-1))
        traced_script_module.save('/media/ysc/C7FD28394D2A63C9/ysc/LearningCode/lite3_ts_isaac-main/policy/policy_x30_H_v2.9.pt')
