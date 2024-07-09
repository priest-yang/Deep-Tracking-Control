from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Lite3DTCCfg( LeggedRobotCfg ):
    #! added by wz
    class env( LeggedRobotCfg.env ):
        # num_envs = 4096
        num_envs = 8
        num_observations = 45 + 8 # 45 + PMTG:13=58
        num_privileged_obs =  693 + 3 + 693  #212+384
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
        horizontal_scale = 0.05 # 0.05 # [m] ->0.05
        vertical_scale = 0.005 # [m]
        border_size = 20 # [m]
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
        
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 6 # number of terrain rows (levels)
        num_cols = 6 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap_terrain, pit_terrain]
        terrain_proportions = [0., 0., 0, 1, 0, 0, 0]
        
        # terrain_proportions = [0, 0, .5,.5, 0]  #stairs up/down
         
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        measure_foot_clearance = True



    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.4] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_HipX_joint': 0.1,   # [rad]
            'HL_HipX_joint': 0.1,   # [rad]
            'FR_HipX_joint': -0.1 ,  # [rad]
            'HR_HipX_joint': -0.1,   # [rad]

            'FL_HipY_joint': -1.,     # [rad]
            'HL_HipY_joint': -1.,   # [rad]
            'FR_HipY_joint': -1.,     # [rad]
            'HR_HipY_joint': -1.,   # [rad]

            'FL_Knee_joint': 1.8,   # [rad]
            'HL_Knee_joint': 1.8,    # [rad]
            'FR_Knee_joint': 1.8,  # [rad]
            'HR_Knee_joint': 1.8,    # [rad]
        }
        
        def str_to_bit(s):
            return int(s.replace(' ', ''), 2)
        
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
        
        penalize_contacts_on = ["TORSO", "THIGH", "SHANK"] #  ["THIGH", "SHANK"]
        collision_state = ["TORSO","THIGH", "SHANK"]
        terminate_after_contacts_on = ["TORSO"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            # lin_vel_x = [-1.0, 1.0] # min max [m/s]
            #! changed by wz
            lin_vel_x = [-.75, .75] # min max [m/s]
            lin_vel_y = [-.75, .75]   # min max [m/s]
            ang_vel_yaw = [-.5, .5]    # min max [rad/s]
            heading = [-3.14, 3.14]
    
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        # base_height_target = 0.32
        base_height_target = 0.34
        max_acc = 100.    
        class scales( LeggedRobotCfg.rewards.scales ):
            
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5
            orientation = -.5
            
            # #! train first
            # feet_air_time = 0.0
            #! train after
            feet_air_time = 1.0

            torques = -0.000001
            dof_pos_limits = -10.0
            dof_acc = -2.5e-7 / 10
            
            collision = -1.5
            termination = -0.1
            stand_still = -0.2
            base_height = 0.0
            body_higher_than_feet = 0.8
            
            #added
            action_rate = -0.01
            ang_vel_xy = - 0.05 / 5
            lin_vel_z = - 2.0 / 2
            foot_clearance = -0.01
            feet_slip = -0.05
            hip_pos = -0.4 / 10
            power = -6e-7
            powerchange =  -0.01 / 2
            pos_acc = -0.005
            foot_acc = -0.007
            smooth = -0.015 / 5 
            
            tracking_optimal_footholds = 1


class Lite3DTCCfgPPO( LeggedRobotCfgPPO ):

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.003
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'lite3_dtc'
        max_iterations = 20000

        resume =  True
        load_run = "legacy_v3"
        checkpoint = -1 # -1 = last saved model       
        

