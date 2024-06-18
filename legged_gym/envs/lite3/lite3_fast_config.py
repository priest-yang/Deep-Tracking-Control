from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Lite3FastCfg( LeggedRobotCfg ):
    #! added by wz
    class env( LeggedRobotCfg.env ):
        # num_envs = 4096
        num_envs = 4096
        num_observations = 45 # 45 + PMTG:13=58
        num_privileged_obs =  693 +3 +693 #212+384
        num_obs_history = 45*5
        num_observation_history = 5 # the length of history ,for lstm  =1
        debug_viz = False

        num_actions = 12
        play_commond  = False
        play_teacher = True
        num_lidar = 3200
        num_lidar_history = 5

        #! added by wz
        # episode_length_s = 40

    #! changed by wz
    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]  
        border_size = 4 * 25
        
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 1 # number of terrain cols (types)

        terrain_length = 32.
        terrain_width = 32.

    #! changed by wz
    class commands( LeggedRobotCfg.commands ):
        resampling_time = 10. # time before command are changed[s]
        # resampling_time = 20. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges( LeggedRobotCfg.commands.ranges ):
            lin_vel_x = [-4.0, 4.0] # min max [m/s]
            lin_vel_y = [-4.0, 4.0]   # min max [m/s]
            ang_vel_yaw = [-2, 2]    # min max [rad/s]
            heading = [-3.14, 3.14]
     


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
        #! changed by wz
        tracking_sigma = 0.2
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.000001
            dof_pos_limits = -10.0
            #! changed by wz
            feet_air_time = 0.0
            tracking_lin_vel_2 = 0.2
            tracking_ang_vel_2 = 0.2

            base_height = -10.0

            #! changed by wz !!! important
            orientation = -10.0
            #! added by wz
            orientation_roll = -5.0


class Lite3FastCfgPPO( LeggedRobotCfgPPO ):

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'Fast_lite3'
        max_iterations = 20000

        
        # load_run = '/home/wt/deeprobotics/dr_gym_vae/logs/Fast_lite3/May21_12-43-51_'
        # load_run = '/home/wt/deeprobotics/dr_gym_vae/logs/Fast_lite3/May27_02-44-52_'
        #! has been tested
        # load_run = '/home/wt/deeprobotics/dr_gym_vae/logs/Fast_lite3/May27_03-19-35_'

        # load_run = '/home/wt/deeprobotics/dr_gym_vae/logs/Fast_lite3/tmp'

        # load_run = '/home/wt/deeprobotics/dr_gym_vae/logs/Fast_lite3/May27_19-27-20_'
        # load_run = '/home/wt/deeprobotics/dr_gym_vae/logs/Fast_lite3/stand_v2'    
    
        #! added by wz
        resume =  True
        checkpoint = -1#2600#8000#'7600' # -1 = last saved model       

