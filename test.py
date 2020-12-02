# Test file things needed 

import gym
import numpy as np

from rex_gym.envs.rex_gym_env import RexGymEnv
from rex_gym.envs.gym.standup_env import RexStandupEnv

env = RexStandupEnv(terrain_id='plane', render=True) 
p = env.rex._pybullet_client
model_id = env.rex.quadruped
print('model ID: ', model_id)
rex_joints = p.getNumJoints(bodyUniqueId=model_id)
print('Number of Rex Joints = ', rex_joints)

def print_joint_state(joint_dict):
    print('Rex Joints States')
    for i in range(rex_joints):
        joint_state_info = p.getJointState(bodyUniqueId=model_id, jointIndex=i)
        print(joint_dict[i] + ' = ', joint_state_info)


def print_link_state(link_dict):
    print('Rex Link States')
    for i in range(rex_joints):
        link_state_info = p.getLinkState(bodyUniqueId=model_id, linkIndex=i)
        if 'toe' in link_dict[i]:
            print(link_dict[i] + ' = ', link_state_info)


def print_contact_info():
    """Gets all contact points and forces
        
        Prints:
        list -- list of entries (link_name, position in m, force in N)
    """
    contact_foot_forces = { 'front_left_toe_link' : { 'pos': None, 'force': 0.0},
                            'front_right_toe_link': { 'pos': None, 'force': 0.0},
                            'rear_left_toe_link'  : { 'pos': None, 'force': 0.0},
                            'rear_right_toe_link' : { 'pos': None, 'force': 0.0}
                        }
    
    contact_points = p.getContactPoints(bodyA=env._ground_id, bodyB=env.rex.quadruped)
    # for some odd reason, multiple contact forces can appear...meaning there's more than just a point contact
    # I assume simulation is realistic enough that there could be a semi-soft body collision

    for contact in contact_points: # deal with a PyBullet bug
        if contact[9] == 0.0:
            continue

        link_index = contact[4]
        if link_index > 0:
            link_name = p.getJointInfo(model_id, link_index)[12].decode('UTF-8')
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

    print(contact_foot_forces)

def print_toe_velocities(toe_pos_last):
    """
        Computes an Euler Backward Approximation of Rex's toe velocities
    """
    # Get the current position after env step
    FL_pos_curr = np.asarray( p.getLinkState(model_id, link_name_to_ID['front_left_toe_link'])[0] )
    FR_pos_curr = np.asarray( p.getLinkState(model_id, link_name_to_ID['front_right_toe_link'])[0] )
    RL_pos_curr = np.asarray( p.getLinkState(model_id, link_name_to_ID['rear_left_toe_link'])[0] )
    RR_pos_curr = np.asarray( p.getLinkState(model_id, link_name_to_ID['rear_right_toe_link'])[0] )

    # get past positions from dictionary
    FL_pos_last = np.asarray( toe_pos_last['front_left_toe_pos'] )
    FR_pos_last = np.asarray( toe_pos_last['front_right_toe_pos'] )
    RL_pos_last = np.asarray( toe_pos_last['rear_left_toe_pos'] )
    RR_pos_last = np.asarray( toe_pos_last['rear_right_toe_pos'] )

    # Do velocity calculations
    FL_toe_vel = np.linalg.norm(( (FL_pos_curr - FL_pos_last)/env._time_step ), ord=2)
    FR_toe_vel = np.linalg.norm(( (FR_pos_curr - FR_pos_last)/env._time_step ), ord=2)
    RL_toe_vel = np.linalg.norm(( (RL_pos_curr - RL_pos_last)/env._time_step ), ord=2)
    RR_toe_vel = np.linalg.norm(( (RR_pos_curr - RR_pos_last)/env._time_step ), ord=2)

    # If any velocities are super tiny, zero them out
    eps = 0.01
    FL_toe_vel = 0.0 if FL_toe_vel < eps else FL_toe_vel
    FR_toe_vel = 0.0 if FR_toe_vel < eps else FR_toe_vel
    RL_toe_vel = 0.0 if RL_toe_vel < eps else RL_toe_vel
    RR_toe_vel = 0.0 if RR_toe_vel < eps else RR_toe_vel

    # Update the current positions
    toe_pos_last['front_left_toe_pos'] = FL_pos_curr 
    toe_pos_last['front_right_toe_pos'] = FR_pos_curr 
    toe_pos_last['rear_left_toe_pos'] = RL_pos_curr 
    toe_pos_last['rear_right_toe_pos'] = RR_pos_curr 

    vel_list = [FL_toe_vel, FR_toe_vel, RL_toe_vel, RR_toe_vel]
    print('Velocities (FL, FR, RL, RR): ', vel_list)
    
# Build joint name dictionary
joint_dict = {}
for i in range(rex_joints):
    joint_info = p.getJointInfo(bodyUniqueId=model_id, jointIndex=i)

    if 'toe' in str(joint_info[1]) or True:
        joint_dict[i] = joint_info[1].decode('UTF-8')

# Build link name dictionary (in PyBullet, link id == joint id)
link_name_to_ID = {}
for i in range(rex_joints):
	name = p.getJointInfo(model_id, i)[12].decode('UTF-8')
	link_name_to_ID[name] = i

# initialize toe_position logging (in world frame)
toe_pos = { 'front_left_toe_pos'  : p.getLinkState(model_id, link_name_to_ID['front_left_toe_link'])[0],
            'front_right_toe_pos' : p.getLinkState(model_id, link_name_to_ID['front_right_toe_link'])[0],
            'rear_left_toe_pos'   : p.getLinkState(model_id, link_name_to_ID['rear_left_toe_link'])[0],
            'rear_right_toe_pos'  : p.getLinkState(model_id, link_name_to_ID['rear_right_toe_link'])[0]

        }       

time = 0
action = env.action_space.sample()
for _ in range(1000):
    ob, re, done, ac = env.step([0]) # take a random action
    time += env._time_step
    print('Sim time t = ', time)
    print_contact_info()
    print_toe_velocities(toe_pos)

env.close()