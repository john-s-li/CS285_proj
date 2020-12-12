"""
    Environment file for Rex Training with
    Periodic Reward Composition for Quadruped Locomotion
"""

from gym import spaces
from rex_gym.envs import rex_gym_env
from rex_gym.model import rex_constants, motor
from rex_gym.model.gait_planner import GaitPlanner
from rex_gym.model.kinematics import Kinematics

import phase_constants
import numpy as np
import pybullet as p
import math
import time

OBSERVATION_EPS = 0.01

class rexPeriodicRewardEnv(rex_gym_env.RexGymEnv):
    """
        The gym environment for the rex.

        Adapted from Nicola Russo's rex_gym repository for the purpose
        of using the novel periodic reward composition idea from
        Jonah Siekmann 

    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 66}
    load_ui = True
    is_terminating = False

    def __init__(self,
                debug=False,
                urdf_version=None,
                control_time_step=0.005,
                action_repeat=5,
                control_latency=0,
                pd_latency=0,
                on_rack=False,
                motor_kp=1.0,
                motor_kd=0.02,
                render=False,
                num_steps_to_log=2000,
                env_randomizer=None,
                log_path=None,
                signal_type='ik',
                target_position=None,
                backwards=None,
                gait_type='trot',
                terrain_type='plane',
                terrain_id='plane',
                mark='base',
                ):
        """Initialize the rex alternating legs gym environment.

            Args:
            urdf_version: [DEFAULT_URDF_VERSION, DERPY_V0_URDF_VERSION] are allowable
                versions. If None, DEFAULT_URDF_VERSION is used. Refer to
                rex_gym_env for more details.
            control_time_step: The time step between two successive control signals.
            action_repeat: The number of simulation steps that an action is repeated.
            control_latency: The latency between get_observation() and the actual
                observation. See minituar.py for more details.
            pd_latency: The latency used to get motor angles/velocities used to
                compute PD controllers. See rex.py for more details.
            on_rack: Whether to place the rex on rack. This is only used to debug
                the walk gait. In this mode, the rex's base is hung midair so
                that its walk gait is clearer to visualize.
            motor_kp: The P gain of the motor.
            motor_kd: The D gain of the motor.
            remove_default_joint_damping: Whether to remove the default joint damping.
            render: Whether to render the simulation.
            num_steps_to_log: The max number of control steps in one episode. If the
                number of steps is over num_steps_to_log, the environment will still
                be running, but only first num_steps_to_log will be recorded in logging.
            env_randomizer: An instance (or a list) of EnvRanzomier(s) that can
                randomize the environment during when env.reset() is called and add
                perturbations when env.step() is called.
            log_path: The path to write out logs. For the details of logging, refer to
                rex_logging.proto.
        """
        self.phase = 0

        self._gait_type = gait_type        
        # for observation space bounding 
        self.max_speed = 1.0
        self.min_speed = 0.5 # change back to 0.2 for OLD TD3 model evaluation
        
        self.min_side_speed = 0.0
        self.max_side_speed = 0.0

        self.speed = np.random.uniform(self.min_speed, self.max_speed)
        self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)
        self.speed_des = [self.speed, self.side_speed]

        # Initialization variables for periodic reward sum composition
        self.theta_FL = phase_constants.PHASE_VALS[self._gait_type]['front_left']
        self.theta_FR = phase_constants.PHASE_VALS[self._gait_type]['front_right']
        self.theta_RL = phase_constants.PHASE_VALS[self._gait_type]['rear_left']
        self.theta_RR = phase_constants.PHASE_VALS[self._gait_type]['rear_right']

        self.min_swing_ratio = 0.6
        self.max_swing_ratio = 0.8
        self.ratio = np.random.uniform(self.min_swing_ratio, self.max_swing_ratio)

        super(rexPeriodicRewardEnv,
              self).__init__(urdf_version=urdf_version,
                             accurate_motor_model_enabled=True,
                             motor_overheat_protection=True,
                             motor_kp=motor_kp,
                             motor_kd=motor_kd,
                             remove_default_joint_damping=False,
                             control_latency=control_latency,
                             pd_latency=pd_latency,
                             on_rack=on_rack,
                             render=render,
                             num_steps_to_log=num_steps_to_log,
                             env_randomizer=env_randomizer,
                             log_path=log_path,
                             control_time_step=control_time_step,
                             action_repeat=action_repeat,
                             target_position=target_position,
                             signal_type=signal_type,
                             backwards=backwards,
                             debug=debug,
                             terrain_id=terrain_id,
                             terrain_type=terrain_type,
                             mark=mark,
                             ratio=self.ratio,
                             forward_reward_cap=5
                            )

        self.height_des = 0.206 # this is init standing height for rex

        self.cycle_complete = 0
        self.cycle_len = 1000 # this is L
        
        # vonmises variables
        self.kappa = phase_constants.VON_MISES_KAPPA

        rex_joints = p.getNumJoints(bodyUniqueId=self.rex.quadruped)
        link_name_to_ID = {}
        for i in range(rex_joints):
            name = p.getJointInfo(self.rex.quadruped, i)[12].decode('UTF-8')
            link_name_to_ID[name] = i

        self.link_name_to_ID = link_name_to_ID
        self.toe_pos_last = { 'front_left_toe_pos'  : p.getLinkState(self.rex.quadruped, self.link_name_to_ID['front_left_toe_link'])[0],
                              'front_right_toe_pos' : p.getLinkState(self.rex.quadruped, self.link_name_to_ID['front_right_toe_link'])[0],
                              'rear_left_toe_pos'   : p.getLinkState(self.rex.quadruped, self.link_name_to_ID['rear_left_toe_link'])[0],
                              'rear_right_toe_pos'  : p.getLinkState(self.rex.quadruped, self.link_name_to_ID['rear_right_toe_link'])[0]

        }  

        
    def step(self, action):
        """Step forward the simulation, given the action.

        Args:
          action: A list of desired motor angles for eight motors.
          --> COMMENT: why only 8 motors? There are 12 (shoulder motors not included??)

        Returns:
          observations: The angles, velocities and torques of all motors.
          reward: The reward for the current state-action pair.
          done: Whether the episode has ended.
          info: A dictionary that stores diagnostic information.

        Raises:
          ValueError: The action dimension is not the same as the number of motors.
          ValueError: The magnitude of actions is out of bounds.
        """
        self._last_base_position = self.rex.GetBasePosition()
        self._last_base_orientation = self.rex.GetBaseOrientation()
        if self._is_render:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self.control_time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            base_pos = self.rex.GetBasePosition()
            # Keep the previous orientation of the camera set by the user.
            [yaw, pitch, dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
            self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)

        for env_randomizer in self._env_randomizers:
            env_randomizer.randomize_step(self)

        # change up swing and stance ratio and desired speeds randomly for robustness
        if np.random.randint(300) == 0:
            self.ratio = np.random.uniform(self.min_swing_ratio, self.max_swing_ratio)

        if np.random.randint(300) == 0:
            self.speed = np.random.uniform(self.min_speed, self.max_speed)
            self.speed_des[0] = self.speed

        if np.random.randint(300) == 0:
            self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)
            self.speed_des[1] = self.side_speed

        self.base_vel_curr_trans, self.base_vel_curr_rot = self.get_base_velocity()
        action = self._transform_action_to_motor_command(action)
        self.rex.Step(action)
        self.base_vel_next_trans, self.base_vel_next_rot = self.get_base_velocity()
                            
        self._env_step_counter += 1
        self.phase += self._action_repeat # the cycle length is CYCLE_TIME/time_step so can add 
                                          # how many times an action was repeated

        if self.phase > self.cycle_len:
            self.phase = self.phase % self.cycle_len 
            self.cycle_complete += 1

        reward = self._reward()
        done = self._termination()

        if done:
            self.rex.Terminate()

        return np.array(self._get_observation_np()), reward, done, {'action': action}


    def reset(self):
        self.init_pose = rex_constants.INIT_POSES["stand"]
        self.phase = 0
        self.speed = np.random.uniform(self.min_speed, self.max_speed)
        self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)
        self.speed_des = [self.speed, self.side_speed]
        self.ratio = np.random.uniform(self.min_swing_ratio, self.max_swing_ratio)
        self.cycle_complete = 0
        
        return super(rexPeriodicRewardEnv, self).reset(initial_motor_angles=self.init_pose, reset_duration=0.5)
        

    def _reward(self):
        """
            Updated reward function with periodic reward composition
        """
        # Clock reward -----------------------------------------------------------------
        A, B = self.get_von_mises(0.0, self.ratio, self.kappa)
        phi = self.phase / self.cycle_len
        #print('Cycles completed = ', self.cycle_complete)

        #print('A, B = ', (A,B))

        phi_FL = self.wrap(phi + self.theta_FL)
        phi_FR = self.wrap(phi + self.theta_FR)
        phi_RL = self.wrap(phi + self.theta_RL)
        phi_RR = self.wrap(phi + self.theta_RR)

        #print(phi_FL)
        #print(phi_FR)
        #print(phi_RL)
        #print(phi_RR)

        FL_swing = self.in_swing(A, B, phi_FL)
        FR_swing = self.in_swing(A, B, phi_FR)
        RL_swing = self.in_swing(A, B, phi_RL)
        RR_swing = self.in_swing(A, B, phi_RR)

        #print('Time since reset = ', self.rex.GetTimeSinceReset())
        #print('phase phi = ', phi)
        #print('FL swing = ', FL_swing)
        #print('FR swing = ', FR_swing)
        #print('RL swing = ', RL_swing)
        #print('RR swing = ', RR_swing)

        if FL_swing:
            c_swing_frc_FL = 1
            c_swing_spd_FL = 0
        else:
            c_swing_frc_FL = 0
            c_swing_spd_FL = 1

        if FR_swing:
            c_swing_frc_FR = 1
            c_swing_spd_FR = 0
        else:
            c_swing_frc_FR = 0
            c_swing_spd_FR = 1

        if RL_swing:
            c_swing_frc_RL = 1
            c_swing_spd_RL = 0
        else:
            c_swing_frc_RL = 0
            c_swing_spd_RL = 1

        if RR_swing:
            c_swing_frc_RR = 1
            c_swing_spd_RR = 0
        else:
            c_swing_frc_RR = 0
            c_swing_spd_RR = 1

        FL_foot_force, FR_foot_force, RL_foot_force, RR_foot_force = self.get_contact_forces()
        FL_vel, FR_vel, RL_vel, RR_vel = self.get_foot_velocities()

        FL_penalty = c_swing_frc_FL*FL_foot_force + c_swing_spd_FL*FL_vel
        FR_penalty = c_swing_frc_FR*FR_foot_force + c_swing_spd_FR*FR_vel
        RL_penalty = c_swing_frc_RL*RL_foot_force + c_swing_spd_RL*RL_vel
        RR_penalty = c_swing_frc_RR*RR_foot_force + c_swing_spd_RR*RR_vel

        foot_penalties = FL_penalty + FR_penalty + RL_penalty + RR_penalty
  
        # Deviation Penalties ----------------------------------------------------------
        # Base height
        base_height = self.rex.GetBasePosition()[-1]
        height_err = np.abs(base_height - self.height_des)
    
        if height_err < 0.02:
            height_err = 0

        # Speed 
        vx, vy, _ = p.getBaseVelocity(bodyUniqueId=self.rex.quadruped)[0]
        vx = -vx # in rex, forward is the negative x direction
        x_vel_err = 4*np.abs(vx - self.speed) # higher emphasis on x velocity error
        y_vel_err = np.abs(vy - self.side_speed)

        # Orientation
        orient_curr = self.rex.GetBaseOrientation()
        orient_des = [0, 0, 0, 1] # not exact, but shouldn't be too far from this
        orient_err = 6 * (1 - np.inner(orient_curr, orient_des)**2 )

        shoulder_orient_des = [0, 0, 0, 1]
        FL_sh, FR_sh, RL_sh, RR_sh = self.get_shoulder_orientation()

        # quaternion similarity: 1 - <q1, q2>**2 == 0 when 100% similar
        # good when error < 0.01 (individually)
        # put HUGE penalty on this
        shoulder_err = 10 * ((1 - np.inner(shoulder_orient_des, FL_sh)**2) + 
                             (1 - np.inner(shoulder_orient_des, FR_sh)**2) +
                             (1 - np.inner(shoulder_orient_des, RL_sh)**2) + 
                             (1 - np.inner(shoulder_orient_des, RR_sh)**2))

        # Energy Penalties --------------------------------------------------------------
        energy_penalty = np.abs(np.dot(self.rex.GetMotorTorques(),
                                       self.rex.GetMotorVelocities())) * self._time_step

        # Acceleration
        a_trans, a_rot = self.get_base_accelerations()
        accel_penalty = 0.15 * np.abs(a_trans.sum() + a_rot.sum())

        # need to encourage exploration: current issue --> Rex is stuck at origin
        # because positive rewards all the time
        # need lim error --> 0, reward > 0 

        beta = -0.6

        reward = beta + \
                 0.200 * np.exp(-orient_err - shoulder_err) +  \
                 0.275 * np.exp(-foot_penalties) +  \
                 0.075 * np.exp(-height_err)     +  \
                 0.250 * np.exp(-x_vel_err)      +  \
                 0.100 * np.exp(-y_vel_err)      +  \
                 0.075 * np.exp(-accel_penalty)  +  \
                 0.025 * np.exp(-energy_penalty)


        return reward

    def get_shoulder_orientation(self):
        """
            returns the orientation of shoulder links in the local inertial frame
        """
        FL_shoulder_orientation = p.getLinkState(bodyUniqueId=self.rex.quadruped, 
                                                 linkIndex=self.link_name_to_ID['front_left_shoulder_link'])[-1]
        FR_shoulder_orientation = p.getLinkState(bodyUniqueId=self.rex.quadruped, 
                                                 linkIndex=self.link_name_to_ID['front_right_shoulder_link'])[-1]
        RL_shoulder_orientation = p.getLinkState(bodyUniqueId=self.rex.quadruped, 
                                                 linkIndex=self.link_name_to_ID['rear_left_shoulder_link'])[-1]
        RR_shoulder_orientation = p.getLinkState(bodyUniqueId=self.rex.quadruped, 
                                                 linkIndex=self.link_name_to_ID['rear_right_shoulder_link'])[-1]

        return [FL_shoulder_orientation, FR_shoulder_orientation, RL_shoulder_orientation, RR_shoulder_orientation]


    def get_base_velocity(self):
        trans_v, rot_v = p.getBaseVelocity(bodyUniqueId=self.rex.quadruped)
        return trans_v, rot_v

    def get_base_accelerations(self):
        trans_a = (np.asarray(self.base_vel_next_trans) - np.asarray(self.base_vel_curr_trans)) / (self._time_step * self._action_repeat)
        rot_a = (np.asarray(self.base_vel_next_rot) - np.asarray(self.base_vel_curr_rot)) / (self._time_step * self._action_repeat)

        return trans_a, rot_a

    def get_contact_forces(self):
        contact_foot_forces = { 'front_left_toe_link' : { 'pos': None, 'force': 0.0},
                                'front_right_toe_link': { 'pos': None, 'force': 0.0},
                                'rear_left_toe_link'  : { 'pos': None, 'force': 0.0},
                                'rear_right_toe_link' : { 'pos': None, 'force': 0.0}
                        }
    
        contact_points = p.getContactPoints(bodyA=self.ground_id, bodyB=self.rex.quadruped)
        # for some odd reason, multiple contact forces can appear...meaning there's more than just a point contact
        # I assume simulation is realistic enough that there could be a semi-soft body collision

        for contact in contact_points: # deal with a PyBullet bug
            if contact[9] == 0.0:
                continue

            link_index = contact[4]
            if link_index > 0:
                link_name = p.getJointInfo(self.rex.quadruped, link_index)[12].decode('UTF-8')
                if link_name in contact_foot_forces.keys():
                    if contact_foot_forces[link_name]['pos'] == None:
                        contact_foot_forces[link_name]['pos'] = contact[6]
                        contact_foot_forces[link_name]['force'] = contact[9]

                    else: # multiple contact forces for a unique link
                        pos1 = np.asarray(contact_foot_forces[link_name]['pos'])
                        pos2 = np.asarray(contact[6])

                        if np.linalg.norm((pos1-pos2), ord=2) < 0.01: # if contacts are close together
                            contact1 = contact_foot_forces[link_name]['force']
                            contact2 = contact[9]

                            new_contact = np.linalg.norm(np.array([contact1, contact2]), ord=2)
                            contact_foot_forces[link_name]['force'] = new_contact

        return [contact_foot_forces['front_left_toe_link']['force'],
                contact_foot_forces['front_right_toe_link']['force'],
                contact_foot_forces['rear_left_toe_link']['force'],
                contact_foot_forces['rear_right_toe_link']['force']
        ]


    def get_foot_velocities(self):
        # Get the current position after env step
        FL_pos_curr = np.asarray( p.getLinkState(self.rex.quadruped, self.link_name_to_ID['front_left_toe_link'])[0] )
        FR_pos_curr = np.asarray( p.getLinkState(self.rex.quadruped, self.link_name_to_ID['front_right_toe_link'])[0] )
        RL_pos_curr = np.asarray( p.getLinkState(self.rex.quadruped, self.link_name_to_ID['rear_left_toe_link'])[0] )
        RR_pos_curr = np.asarray( p.getLinkState(self.rex.quadruped, self.link_name_to_ID['rear_right_toe_link'])[0] )

        # get past positions from dictionary
        FL_pos_last = np.asarray( self.toe_pos_last['front_left_toe_pos'] )
        FR_pos_last = np.asarray( self.toe_pos_last['front_right_toe_pos'] )
        RL_pos_last = np.asarray( self.toe_pos_last['rear_left_toe_pos'] )
        RR_pos_last = np.asarray( self.toe_pos_last['rear_right_toe_pos'] )

        # Do velocity calculations
        FL_toe_vel = np.linalg.norm(( (FL_pos_curr - FL_pos_last)/(self._time_step * self._action_repeat) ), ord=2)
        FR_toe_vel = np.linalg.norm(( (FR_pos_curr - FR_pos_last)/(self._time_step * self._action_repeat) ), ord=2)
        RL_toe_vel = np.linalg.norm(( (RL_pos_curr - RL_pos_last)/(self._time_step * self._action_repeat) ), ord=2)
        RR_toe_vel = np.linalg.norm(( (RR_pos_curr - RR_pos_last)/(self._time_step * self._action_repeat) ), ord=2)

        # If any velocities are super tiny, zero them out
        eps = 0.01
        FL_toe_vel = 0.0 if FL_toe_vel < eps else FL_toe_vel
        FR_toe_vel = 0.0 if FR_toe_vel < eps else FR_toe_vel
        RL_toe_vel = 0.0 if RL_toe_vel < eps else RL_toe_vel
        RR_toe_vel = 0.0 if RR_toe_vel < eps else RR_toe_vel

        # Update the current positions
        self.toe_pos_last['front_left_toe_pos'] = FL_pos_curr 
        self.toe_pos_last['front_right_toe_pos'] = FR_pos_curr 
        self.toe_pos_last['rear_left_toe_pos'] = RL_pos_curr 
        self.toe_pos_last['rear_right_toe_pos'] = RR_pos_curr 

        return [FL_toe_vel, FR_toe_vel, RL_toe_vel, RR_toe_vel]

   
    def get_clock(self):
        """
            returns the clock input info for policy conditioning for 4 legs
        """
        return [np.sin(2*np.pi*(self.phase/self.cycle_len + self.theta_FL)/self.cycle_len),
                np.sin(2*np.pi*(self.phase/self.cycle_len + self.theta_FR)/self.cycle_len),
                np.sin(2*np.pi*(self.phase/self.cycle_len + self.theta_RL)/self.cycle_len),
                np.sin(2*np.pi*(self.phase/self.cycle_len + self.theta_RR)/self.cycle_len)
        ]


    def in_swing(self, A, B, phi):
        """
            returns wheter a leg should be in stance or swing based 
            on vonmises distribution
        """
        if phi >= A and phi <= B: return True

        return False


    def get_von_mises(self, a, b, kappa):
        """
            returns A_i and B_i sampled from a von mises distribution with 
            a common variance param kappa
        """
        A = np.random.vonmises(a, kappa)
        B = np.random.vonmises(b, kappa)

        if A < 0: A = 0

        return (A,B)

    def wrap(self, x):
        if x < 0:
            x = 1 + x
        elif x > 1:
            x = x - 1
        return x

    def _get_observation_np(self) -> np.ndarray: # need this for baselines
        """Get observation of this environment, including noise and latency.

        rex class maintains a history of true observations. Based on the
        latency, this function will find the observation at the right time,
        interpolate if necessary. Then Gaussian noise is added to this observation
        based on self.observation_noise_stdev.

        Returns:
          The noisy observation with latency.
        """
        observation = []
        observation.extend(self.rex.GetMotorAngles().tolist())
        observation.extend(self.rex.GetMotorVelocities().tolist())
        observation.extend(self.rex.GetMotorTorques().tolist())
        observation.extend(list(self.rex.GetBaseOrientation()))

        # in addition to state, will need ratio, clock_variables, and desired speed
        observation.extend([self.ratio]) # only 1
        observation.extend(self.get_clock()) # 4 variables (1 per leg)
        observation.extend(self.speed_des) # [vx_des, vy_des]
        self._observation = observation
        return np.array(self._observation)


    def _get_observation_upper_bound(self):
        """Get the upper bound of the observation.

        Returns:
          The upper bound of an observation. See GetObservation() for the details
            of each element of an observation.
        """
        upper_bound = np.zeros(self._get_observation_dimension())
        num_motors = self.rex.num_motors
        upper_bound[0:num_motors] = math.pi  # Joint angle.
        upper_bound[num_motors:2 * num_motors] = motor.MOTOR_SPEED_LIMIT  # Joint velocity.
        upper_bound[2 * num_motors:3 * num_motors] = motor.OBSERVED_TORQUE_LIMIT  # Joint torque.
        upper_bound[3 * num_motors:-7] = 1.0  # Quaternion of base orientation.
        upper_bound[-7] = 1.0 # ratio in [0,1]
        upper_bound[-6:-2] = [1.0, 1.0, 1.0, 1.0] # sin in [-1, 1]
        upper_bound[-2:] = [self.max_speed, self.max_side_speed]

        return upper_bound

    def _get_observation_lower_bound(self):
        """Get the lower bound of the observation."""
        lower_bound = -self._get_observation_upper_bound()
        lower_bound[-7] = 0.0
        lower_bound[-2:] = [self.min_speed, self.min_side_speed]
        return lower_bound

    def _get_observation_dimension(self):
        """Get the length of the observation list.

        Returns:
          The length of the observation list.
        """
        return len(self._get_observation_np())
        

        

    