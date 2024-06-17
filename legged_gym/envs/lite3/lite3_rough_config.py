from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Lite3RoughCfg( LeggedRobotCfg ):
    #! added by wz
    class env( LeggedRobotCfg.env ):
        # num_envs = 4096
        num_envs = 2048
        num_observations = 45 # 45 + PMTG:13=58
        num_privileged_obs =  187 +3 +187 #212+384
        num_obs_history = 45*5
        num_observation_history = 5 # the length of history ,for lstm  =1
        debug_viz = False

        num_actions = 12
        play_commond  = False
        play_teacher = True
        num_lidar = 3200
        num_lidar_history = 5


    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.4] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # 'FL_HipX_joint': 0.1,   # [rad]
            # 'HL_HipX_joint': 0.1,   # [rad]
            # 'FR_HipX_joint': -0.1 ,  # [rad]
            # 'HR_HipX_joint': -0.1,   # [rad]
            #! changed by wz
            'FL_HipX_joint': 0.0,   # [rad]
            'HL_HipX_joint': 0.0,   # [rad]
            'FR_HipX_joint': -0.0 ,  # [rad]
            'HR_HipX_joint': -0.0,   # [rad]

            'FL_HipY_joint': -1.,     # [rad]
            'HL_HipY_joint': -1.,   # [rad]
            'FR_HipY_joint': -1.,     # [rad]
            'HR_HipY_joint': -1.,   # [rad]

            'FL_Knee_joint': 1.8,   # [rad]
            'HL_Knee_joint': 1.8,    # [rad]
            'FR_Knee_joint': 1.8,  # [rad]
            'HR_Knee_joint': 1.8,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 25.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/Lite3/urdf/Lite3.urdf'
        name = "Lite3"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "SHANK"]
        collision_state = ["TORSO","THIGH", "SHANK"]
        terminate_after_contacts_on = ["TORSO"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
    
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        # base_height_target = 0.32
        base_height_target = 0.34
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.000001
            dof_pos_limits = -10.0
            


            #######################################
            #! train first
            # feet_air_time = 0.0
            #! train after
            # feet_air_time = 1.0
            #######################################
            


            #######################################
            # #!step 2: unlock abilitity(seems impossible 0.23cm std:0.62 level:6.07) (wz_unbelievable)
            torques = -0.000001
            dof_pos_limits = -10.0
            dof_acc = -2.5e-7 / 10
            ang_vel_xy = - 0.05 / 10
            lin_vel_z = - 2.0 / 10
            smooth = -0.015 / 10
            feet_air_time = 1.0
             #######################################
            

            #######################################
            #!step 3: try from 0.23cm std:0.62 level:6.10 (wz_unbelievable_2 !!!has been tested)
            # torques = -0.000001
            # dof_pos_limits = -10.0
            # ang_vel_xy = - 0.05 / 10
            # lin_vel_z = - 2.0 / 10
            # smooth = -0.015 / 10
            # feet_air_time = 1.0
            #######################################


class Lite3RoughCfgPPO( LeggedRobotCfgPPO ):

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_lite3'
        max_iterations = 20000

        
        # load_run = '/home/wt/deeprobotics/dr_gym_vae/logs/rough_lite3/May21_12-43-51_'

        #! has been tested
        # load_run = '/home/wt/deeprobotics/dr_gym_vae/logs/rough_lite3/May28_10-20-14_'
        # load_run = '/home/wt/deeprobotics/dr_gym_vae/logs/rough_lite3/wz'
        # load_run = '/home/wt/deeprobotics/dr_gym_vae/logs/rough_lite3/May28_18-35-25_'
        # load_run = '/home/wt/deeprobotics/dr_gym_vae/logs/rough_lite3/May29_02-14-18_'

        #! added by wz
        resume =  True
        checkpoint = -1#2600#8000#'7600' # -1 = last saved model       

