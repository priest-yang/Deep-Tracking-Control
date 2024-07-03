from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class X30DTCCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        # num_envs = 4096
        num_envs = 8# 4096 * 3
        num_observations = 45 + 8 # 45 + PMTG:13=58
        num_privileged_obs =  693 +3 +693 #212+384
        num_obs_history = (45 + 8)*5
        num_observation_history = 5 # the length of history ,for lstm  =1
        debug_viz = False

        num_actions = 12
        play_commond  = False
        play_teacher = True
        num_lidar = 3200
        num_lidar_history = 5

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        # mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # 0.05 # [m] ->0.05
        vertical_scale = 0.005 # [m]
        border_size = 20 # [m]
        #! changed by wz
        # border_size = 0
        
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        measure_heights = True
        #! DTC
        measured_points_x = [-0.8, -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        measured_points_y = [-0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        num_height_points = 33 * 21
        measured_x_dim = 33
        measured_y_dim = 21
        
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        #! train first
        max_init_terrain_level = 5 # starting curriculum state
        # #! train after
        # max_init_terrain_level = 9 # starting curriculum state
        # max_init_terrain_level = 0
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap_terrain, pit_terrain]
        #! changed by shaoze
        terrain_proportions = [0.1, 0.1, .2, .2, .2, .2, .0]
        
        # terrain_proportions = [0, 0, .5,.5, 0]  #stairs up/down
         
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

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
        
        def str_to_bit(s):
            return int(s.replace(' ', ''), 2)
        
        collision_filter = {
            'base': str_to_bit('1000 1000 1000 1000'),
            'fl_hip': str_to_bit('1100 0000 0000 0000'),
            'fl_thigh': str_to_bit('1110 0000 0000 0000'),
            'fl_shank': str_to_bit('1111 0000 0000 0000'),
            'fl_shank1': str_to_bit('1111 0000 0000 0000'),
            'fl_shank2': str_to_bit('1111 0000 0000 0000'),
            'fl_foot': str_to_bit('1111 0000 0000 0000'),

            'fr_hip': str_to_bit('1000 0100 0000 0000'),
            'fr_thigh': str_to_bit('1000 0110 0000 0000'),
            'fr_shank': str_to_bit('0000 1111 0000 0000'),
            'fr_shank1': str_to_bit('0000 1111 0000 0000'),
            'fr_shank2': str_to_bit('0000 1111 0000 0000'),
            'fr_foot': str_to_bit('0000 1111 0000 0000'),

            'hl_hip': str_to_bit('1000 0000 0100 0000'), 
            'hl_thigh': str_to_bit('1000 0000 0110 0000'),
            'hl_shank': str_to_bit('0000 0000 1111 0000'),
            'hl_shank1': str_to_bit('0000 0000 1111 0000'),
            'hl_shank2': str_to_bit('0000 0000 1111 0000'),
            'hl_foot': str_to_bit('0000 0000 1111 0000'),

            'hr_hip': str_to_bit('1000 0000 0000 0100'),
            'hr_thigh': str_to_bit('1000 0000 0000 0110'),
            'hr_shank': str_to_bit('0000 0000 0000 1111'),
            'hr_shank1': str_to_bit('0000 0000 0000 1111'),
            'hr_shank2': str_to_bit('0000 0000 0000 1111'),
            'hr_foot': str_to_bit('0000 0000 0000 1111'),
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
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        base_height_target = 0.49
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.000001#-0.0002
            dof_pos_limits = -10.0

            #######################################
            #!step 3: try from 0.23cm std:0.62 level:6.10 (wz_unbelievable_2 !!!has been tested)
            torques = -0.000001
            dof_pos_limits = -10.0
            ang_vel_xy = - 0.05 / 10
            lin_vel_z = - 2.0 / 10
            smooth = -0.015 / 10
            feet_air_time = 1.0
            #######################################
            tracking_optimal_footholds = 0.05

class X30DTCCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'X30_dtc'
        max_iterations = 30000

        resume =  True
        # load_run = '/home/wt/deeprobotics/dr_gym_vae/logs/rough_X30/May22_11-36-13_'
        checkpoint = -1 #2600#8000#'7600' # -1 = last saved model        

   

  
