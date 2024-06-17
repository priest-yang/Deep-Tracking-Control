from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class X30RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        # num_envs = 4096
        #! changed by wz
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
        pos = [0.0, 0.0, 0.51] # x,y,z [m]
        pos_z_range = [0.5, 0.51]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_HipX_joint': 0.1,   # [rad]0.1
            'HL_HipX_joint': 0.1,   # [rad]0.1
            'FR_HipX_joint': -0.1 ,  # [rad]-0.1
            'HR_HipX_joint': -0.1,   # [rad]-0.1

            'FL_HipY_joint': -0.715,     # [rad]
            'HL_HipY_joint': -0.715,   # [rad]
            'FR_HipY_joint': -0.715,     # [rad]
            'HR_HipY_joint': -0.715,   # [rad]

            'FL_Knee_joint': 1.43,   # [rad]
            'HL_Knee_joint': 1.43,    # [rad]
            'FR_Knee_joint': 1.43,  # [rad]
            'HR_Knee_joint': 1.43,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'HipX': 120. , 'HipY': 120. ,'Knee': 150. }  # [N*m/rad]
        damping = {'HipX': 3. , 'HipY': 3. ,'Knee': 3.5 }     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/X30/urdf/X30.urdf'
        name = "X30"
        foot_name = "FOOT"
        penalize_contacts_on = ["THIGH", "SHANK"]
        collision_state = ["TORSO","THIGH", "SHANK"]
        terminate_after_contacts_on = ["TORSO"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        base_height_target = 0.49
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.000001#-0.0002
            dof_pos_limits = -10.0

class X30RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_X30'
        max_iterations = 30000

        resume =  True
        load_run = '/home/wt/deeprobotics/dr_gym_vae/logs/rough_X30/May22_11-36-13_'
        checkpoint = -1#2600#8000#'7600' # -1 = last saved model        

   

  
