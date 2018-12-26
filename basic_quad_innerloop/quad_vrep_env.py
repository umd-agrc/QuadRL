import numpy as np
from abc import abstractmethod
from gym import Env
from gym.spaces import Box
import random
import time
from collections import namedtuple

import vrep

class QuadVrepEnv(Env):
    def __init__(
            self,
            episode_len=None
    ):

        self.clientID = None
        # Start V-REP connection
        try:
            vrep.simxFinish(-1)
            print("Connecting to simulator...")
            self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
            if self.clientID == -1:
                print("Failed to connect to remote API Server")
                self.vrep_exit()
        except KeyboardInterrupt:
            self.vrep_exit()
            return

        self.episode_len = episode_len
        self.timestep = 0
        self.dt = .001
        self.propellers = ['rotor1thrust',
                           'rotor2thrust',
                           'rotor3thrust',
                           'rotor4thrust']
        self.quad_name = 'Quadricopter'
        self.scene_name = 'quad_innerloop.ttt'
        self.setpoint_delta = np.array([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        self.quad_handle = None

        self.pos_start = np.zeros(3)
        self.euler_start = np.zeros(3)
        self.vel_start = np.zeros(3)
        self.angvel_start = np.zeros(3)
        self.pos_old = self.pos_start
        self.euler_old = self.euler_start
        self.vel_old = self.vel_start
        self.angvel_old = self.angvel_start
        self.pos_new = self.pos_old
        self.euler_new = self.euler_old
        self.vel_new = self.vel_old
        self.angvel_new = self.angvel_old

        ob_high = np.array(
                [10., 10., 10., 20., 20., 20., 21., 21., 21., 21., 21., 21.,
                 10., 10., 10., 20., 20., 20., 21., 21., 21., 21., 21., 21.])
        ob_low = -ob_high
        self.observation_space = Box(low=ob_low, high=ob_high, dtype=np.float32)
        ac_high = np.array([20., 20., 20., 20.])
        ac_low = np.zeros(4)
        self.action_space = Box(low=ac_low, high=ac_high, dtype=np.float32)

        print("Setting simulator to async mode...")
        vrep.simxSynchronous(self.clientID, True)
        vrep.simxSetFloatingParameter(self.clientID,
                                      vrep.sim_floatparam_simulation_time_step,
                                      self.dt,  # specify a simulation time step
                                      vrep.simx_opmode_oneshot)

        # Load V-REP scene
        print("Loading scene...")
        vrep.simxLoadScene(
                self.clientID, self.scene_name, 0xFF, vrep.simx_opmode_blocking)

        # Get quadrotor handle
        err, self.quad_handle = vrep.simxGetObjectHandle(
                self.clientID, self.quad_name, vrep.simx_opmode_blocking)

        # Initialize quadrotor position and orientation
        vrep.simxGetObjectPosition(
                self.clientID, self.quad_handle, -1, vrep.simx_opmode_streaming)
        vrep.simxGetObjectOrientation(
                self.clientID, self.quad_handle, -1, vrep.simx_opmode_streaming)
        vrep.simxGetObjectVelocity(
                self.clientID, self.quad_handle, vrep.simx_opmode_streaming)

    def reset(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)
        time.sleep(0.1)

        # Setup V-REP simulation
        print("Setting simulator to async mode...")
        vrep.simxSynchronous(self.clientID, True)
        vrep.simxSetFloatingParameter(self.clientID,
                                      vrep.sim_floatparam_simulation_time_step,
                                      self.dt,  # specify a simulation time step
                                      vrep.simx_opmode_oneshot)

        # Load V-REP scene
        print("Loading scene...")
        vrep.simxLoadScene(
                self.clientID, self.scene_name, 0xFF,vrep.simx_opmode_blocking)

        # Get quadrotor handle
        err, self.quad_handle = vrep.simxGetObjectHandle(
                self.clientID, self.quad_name, vrep.simx_opmode_blocking)

        # Initialize quadrotor position and orientation
        vrep.simxGetObjectPosition(
                self.clientID, self.quad_handle, -1, vrep.simx_opmode_streaming)
        vrep.simxGetObjectOrientation(
                self.clientID, self.quad_handle, -1, vrep.simx_opmode_streaming)
        vrep.simxGetObjectVelocity(
                self.clientID, self.quad_handle, vrep.simx_opmode_streaming)

        # Start simulation
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

        # Initialize rotors
        print("Initializing propellers...")
        for i in range(len(self.propellers)):
            vrep.simxClearFloatSignal(
                    self.clientID, self.propellers[i], vrep.simx_opmode_oneshot)

        # Get quadrotor initial position and orientation
        err, self.pos_start = vrep.simxGetObjectPosition(
                self.clientID, self.quad_handle, -1, vrep.simx_opmode_buffer)
        err, self.euler_start = vrep.simxGetObjectOrientation(
                self.clientID, self.quad_handle, -1, vrep.simx_opmode_buffer)
        err, self.vel_start, self.angvel_start = vrep.simxGetObjectVelocity(
                self.clientID, self.quad_handle, vrep.simx_opmode_buffer)

        self.pos_start = np.asarray(self.pos_start)
        self.euler_start = np.asarray(self.euler_start)*10.
        self.vel_start = np.asarray(self.vel_start)
        self.angvel_start = np.asarray(self.angvel_start)

        self.pos_old = self.pos_start
        self.euler_old = self.euler_start
        self.vel_old = self.vel_start
        self.angvel_old = self.angvel_start
        self.pos_new = self.pos_old
        self.euler_new = self.euler_old
        self.vel_new = self.vel_old
        self.angvel_new = self.angvel_old

        self.pos_error = (self.pos_start + self.setpoint_delta[0:3]) \
                         - self.pos_new
        self.euler_error = (self.euler_start + self.setpoint_delta[3:6]) \
                           - self.euler_new
        self.vel_error = (self.vel_start + self.setpoint_delta[6:9]) \
                         - self.vel_new
        self.angvel_error = (self.angvel_start + self.setpoint_delta[9:12]) \
                            - self.angvel_new

        self.pos_error_all = self.pos_error
        self.euler_error_all = self.euler_error

        self.init_f=5.8

        self.propeller_vels = [self.init_f, self.init_f, self.init_f, self.init_f]

        self.timestep = 1

        return self.get_state()

    def step(self, actions):
        # Set propeller thrusts
        print("Setting propeller thrusts...")
        # Only PD control bc can't find api function for getting simulation time
        self.propeller_vels = pid(
                self.pos_error,self.euler_new,self.euler_error[2],
                self.vel_error,self.angvel_error)
        self.propeller_vels += actions

        # Send propeller thrusts
        print("Sending propeller thrusts...")
        [vrep.simxSetFloatSignal(
            self.clientID, prop, vels, vrep.simx_opmode_oneshot) for prop, vels in
         zip(self.propellers, self.propeller_vels)]

        # Trigger simulator step
        print("Stepping simulator...")
        vrep.simxSynchronousTrigger(self.clientID)

        # Get quadrotor initial position and orientation
        err, self.pos_new = vrep.simxGetObjectPosition(
                self.clientID, self.quad_handle, -1, vrep.simx_opmode_buffer)
        err, self.euler_new = vrep.simxGetObjectOrientation(
                self.clientID, self.quad_handle, -1, vrep.simx_opmode_buffer)
        err, self.vel_new, self.angvel_new = vrep.simxGetObjectVelocity(
                self.clientID, self.quad_handle, vrep.simx_opmode_buffer)

        self.pos_new = np.asarray(self.pos_new)
        self.euler_new = np.asarray(self.euler_new)*10
        self.vel_new = np.asarray(self.vel_new)
        self.angvel_new = np.asarray(self.angvel_new)

        self.pos_old = self.pos_new
        self.euler_old = self.euler_new
        self.vel_old = self.vel_new
        self.angvel_old = self.angvel_new

        self.pos_error = (self.pos_start + self.setpoint_delta[0:3]) \
                         - self.pos_new
        self.euler_error = (self.euler_start + self.setpoint_delta[3:6]) \
                           - self.euler_new
        self.euler_error %= 2*np.pi
        for i in range(len(self.euler_error)):
            if self.euler_error[i] > np.pi:
                self.euler_error[i] -= 2*np.pi
        self.vel_error = (self.vel_start + self.setpoint_delta[6:9]) \
                         - self.vel_new
        self.angvel_error = (self.angvel_start + self.setpoint_delta[9:12]) \
                            - self.angvel_new

        valid = self.is_valid_state()

        rew = self.get_reward()
        self.timestep += 1
        done = False
        if self.timestep > self.episode_len or not valid:
            done = True

        return self.get_state(), rew, done, {}

    def get_state(self):
        self.state = np.concatenate(
                (self.pos_error,
                 self.euler_error,
                 self.vel_error,
                 self.angvel_error,
                 self.pos_error,
                 self.euler_error,
                 self.vel_error,
                 self.angvel_error))
        return self.state

    def get_reward(self):
        if self.is_valid_state():
            return 2*(1-(self.state-self.observation_space.low)\
                        /(self.observation_space.high-self.observation_space.low))-1
        else:
            return -1*np.ones(len(self.state))

    def check_quad_flipped(self):
        if abs(self.euler_new[0]) > 20. or abs(self.euler_new[1]) > 20.:
            print("Quad flipped")
            return True

    def is_valid_state(self):
        valid = True

        if self.pos_new[2] < 2.0:
            valid = False
        diff = np.fabs(self.pos_new - self.pos_start)
        if np.amax(diff[:3]) > 10.:
            valid = False
        if self.check_quad_flipped():
            valid = False
        return valid

    def vrep_exit(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_oneshot_wait)
        vrep.simxFinish(self.clientID)
        exit(0)



def motor_ctl(ctl):
    base_thrust = 5.8
    return [base_thrust + ctl['thrust'] + ctl['pitch'] - ctl['roll'] + ctl['yaw'],
            base_thrust + ctl['thrust'] - ctl['pitch'] - ctl['roll'] - ctl['yaw'],
            base_thrust + ctl['thrust'] - ctl['pitch'] + ctl['roll'] + ctl['yaw'],
            base_thrust + ctl['thrust'] + ctl['pitch'] + ctl['roll'] - ctl['yaw']]

R_psi = lambda x: np.array([[np.cos(x),np.sin(x),0],
                            [-np.sin(x),np.cos(x),0],
                            [0,0,1]])
R_theta = lambda x: np.array([[np.cos(x),0,-np.sin(x)],
                              [0,1,0],
                              [np.sin(x),0,np.cos(x)]])
R_phi = lambda x: np.array([[1,0,0],
                            [0,np.cos(x),np.sin(x)],
                            [0,-np.sin(x),np.cos(x)]])
R_euler = lambda phi,theta,psi: np.matmul(R_phi(phi),
                                          np.matmul(R_theta(theta),R_psi(psi)))

# rotation matrix from absolute vel in NED frame to body axis vel uvw
R_vel_body = lambda phi,theta,psi: np.linalg.inv(
        [[np.cos(theta)*np.cos(psi),
          np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi),
          np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi)],
         [np.cos(theta)*np.sin(psi),
          np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi),
          np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi)],
         [-np.sin(theta),
          np.sin(phi)*np.cos(theta),
          np.cos(phi)*np.cos(theta)]])

R_angvel_body = lambda phi,theta,psi: np.array(
        [[1, 0, -np.sin(theta)],
         [0, np.cos(phi), np.sin(phi)*np.cos(theta)],
         [0, -np.sin(phi), np.cos(phi)*np.cos(theta)]])

def pid(pos_error,euler_absolute,yaw_error,vel_error,angvel_error):
    kp = {'roll' : -3, 'pitch' : -3, 'yaw' : -1,
          'x' : -2, 'y' : 2, 'z' : 4.5}
    kd = {'roll' : -0.5, 'pitch' : -0.5, 'yaw' : -0.2,
          'x' : -0.4, 'y' : 0.4, 'z' : 1.2}
    ki = {'roll' : 1, 'pitch' : 1, 'yaw' : 1,
          'x' : 1, 'y' : 1, 'z' : 1}
    R = R_euler(*euler_absolute)
    pos_error = np.matmul(R,pos_error)
    R_vel = R_vel_body(*euler_absolute)
    vel_error = np.matmul(R_vel,vel_error)
    R_angvel = R_angvel_body(*euler_absolute)
    angvel_error = np.matmul(R_angvel,angvel_error)
    trpy_ctl = {'thrust' : 0, 'roll' : 0, 'pitch' : 0, 'yaw' : 0}
    trpy_ctl['thrust'] = (1/(np.cos(euler_absolute[0])*np.cos(euler_absolute[1]))) \
            * (pos_error[2] * kp['z'] + vel_error[2] * kd['z'])
    #roll_des = kp['y'] * pos_error[0] + kd['y'] * vel_error[0]
    #pitch_des = kp['x'] * pos_error[1] + kd['x'] * vel_error[1]
    pitch_des = kp['y'] * pos_error[0] + kd['y'] * vel_error[0]
    roll_des = kp['x'] * pos_error[1] + kd['x'] * vel_error[1]

    roll_error = roll_des - euler_absolute[0]
    pitch_error = pitch_des - euler_absolute[1]

    trpy_ctl['roll'] = kp['roll'] * roll_error + kd['roll'] * angvel_error[0]
    trpy_ctl['pitch'] = kp['pitch'] * pitch_error + kd['pitch'] * angvel_error[1]
    trpy_ctl['yaw'] = kp['yaw'] * yaw_error + kd['yaw'] * angvel_error[2]
    return motor_ctl(trpy_ctl)



