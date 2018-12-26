import random
import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import vrep


ROLLOUT_LEN = 5000


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

scene_name = 'quad_innerloop.ttt'
quad_name = 'Quadricopter'
propellers = ['rotor1thrust', 'rotor2thrust', 'rotor3thrust', 'rotor4thrust']

setpoint_delta = np.array([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.])

# Static vars decorator
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, input, hidden, output=1):
        super(DQN, self).__init__()

        self.linear1 = nn.Linear(input, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        #self.linear3 = nn.Linear(hidden, hidden)

        self.bn1 = nn.BatchNorm1d(input)

        self.bn2 = nn.BatchNorm1d(hidden)
        #self.bn3 = nn.BatchNorm1d(hidden)

        self.linear_final = nn.Linear(hidden, output)

    def forward(self, x):
        x = self.bn1(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = F.relu(x)


        #x = self.linear3(x)
        #x = self.bn3(x)
        #x = F.relu(x)

        x = self.linear_final(x)
        return x


def is_valid_state(pos_start, pos_current, euler_current):
    valid = True

    if pos_current[2] < 2.0:
        valid = False
    diff = np.fabs(pos_current - pos_start)
    if np.amax(diff[:3]) > 10.:
        valid = False
    if check_quad_flipped(euler_current):
        valid = False
    return valid

def generate_forces(model, state,learning_rate):
    state=Variable(state, requires_grad=True)
    model.eval()
    V=model(state)
    V.backward()

    return list( np.sign(state.grad.data[0,-5:-1].numpy()) *1.)

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

# Only PD control bc can't find api function for getting simulation time
# wilselby.com/research/arducopter/controller-design/
@static_vars(xi=0,yi=0,zi=0,rolli=0,pitchi=0)
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
    ##TODO angle/angle-rate controller
    return motor_ctl(trpy_ctl)

def apply_forces(forces, delta_forces):
    for i in range(4):
        forces[i]+=delta_forces[i]
        if forces[i]>20:
            forces[i]=20
        if forces[i]<0.:
            forces[i]=0.
    return forces

def reset(clientID):
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
    time.sleep(0.1)
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)


def DQN_update2(model, memory, batch_size, GAMMA, optimizer):
    model.train()

    if len(memory) < 10:
        return
    elif len(memory) < batch_size:
        batch_size = len(memory)
    else:
        batch_size = batch_size
    transitions = memory.sample(batch_size)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken

    state_action_values = model(state_batch).view(-1,1)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(batch_size, 1))

    next_state_values[non_final_mask] = model(non_final_next_states).view(-1,1)

    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-.1, .1)
    optimizer.step()


def vrep_exit(clientID):
    vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(clientID)
    exit(0)


def check_quad_flipped(euler):

    if abs(euler[0]) > 20. or abs(euler[1]) > 20.:
        print("Quad flipped")
        return True


def main():
    # Start V-REP connection
    try:
        vrep.simxFinish(-1)
        print("Connecting to simulator...")
        clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)
        if clientID == -1:
            print("Failed to connect to remote API Server")
            vrep_exit(clientID)
    except KeyboardInterrupt:
        vrep_exit(clientID)
        return

    # Setup V-REP simulation
    print("Setting simulator to async mode...")
    vrep.simxSynchronous(clientID, True)
    dt = .001
    vrep.simxSetFloatingParameter(clientID,
                                  vrep.sim_floatparam_simulation_time_step,
                                  dt,  # specify a simulation time step
                                  vrep.simx_opmode_oneshot)

    # Load V-REP scene
    print("Loading scene...")
    vrep.simxLoadScene(clientID, scene_name, 0xFF, vrep.simx_opmode_blocking)

    # Get quadrotor handle
    err, quad_handle = vrep.simxGetObjectHandle(clientID, quad_name, vrep.simx_opmode_blocking)
    print(err,quad_handle)

    # Initialize quadrotor position and orientation
    vrep.simxGetObjectPosition(clientID, quad_handle, -1, vrep.simx_opmode_streaming)
    vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_streaming)
    vrep.simxGetObjectVelocity(clientID, quad_handle, vrep.simx_opmode_streaming)

    # Start simulation
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)

    # Initialize rotors
    print("Initializing propellers...")
    for i in range(len(propellers)):
        vrep.simxClearFloatSignal(clientID, propellers[i], vrep.simx_opmode_oneshot)

    # Get quadrotor initial position and orientation
    err, pos_start = vrep.simxGetObjectPosition(clientID, quad_handle, -1, vrep.simx_opmode_buffer)
    err, euler_start = vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_buffer)
    err, vel_start, angvel_start = vrep.simxGetObjectVelocity(clientID,
            quad_handle, vrep.simx_opmode_buffer)

    pos_start = np.asarray(pos_start)
    euler_start = np.asarray(euler_start)*10.
    vel_start = np.asarray(vel_start)
    angvel_start = np.asarray(angvel_start)

    pos_old = pos_start
    euler_old = euler_start
    vel_old = vel_start
    angvel_old = angvel_start
    pos_new = pos_old
    euler_new = euler_old
    vel_new = vel_old
    angvel_new = angvel_old

    pos_error = (pos_start + setpoint_delta[0:3]) - pos_new
    euler_error = (euler_start + setpoint_delta[3:6]) - euler_new
    vel_error = (vel_start + setpoint_delta[6:9]) - vel_new
    angvel_error = (angvel_start + setpoint_delta[9:12]) - angvel_new

    pos_error_all = pos_error
    euler_error_all = euler_error

    n_input = 6
    n_forces=4
    init_f=7.

    state = [0 for i in range(n_input)]
    state = torch.from_numpy(np.asarray(state, dtype=np.float32)).view(-1, n_input)

    propeller_vels = [init_f, init_f, init_f, init_f]
    delta_forces = [0., 0., 0., 0.]

    extended_state=torch.cat((state,torch.from_numpy(np.asarray([propeller_vels], dtype=np.float32))),1)
    memory = ReplayMemory(ROLLOUT_LEN)
    test_num = 1
    timestep = 1
    while (vrep.simxGetConnectionId(clientID) != -1):

        # Set propeller thrusts
        print("Setting propeller thrusts...")
        # Only PD control bc can't find api function for getting simulation time
        propeller_vels = pid(pos_error,euler_new,euler_error[2],vel_error,angvel_error)
        #propeller_vels = apply_forces(propeller_vels, delta_forces) # for dqn

        # Send propeller thrusts
        print("Sending propeller thrusts...")
        [vrep.simxSetFloatSignal(clientID, prop, vels, vrep.simx_opmode_oneshot) for prop, vels in
         zip(propellers, propeller_vels)]

        # Trigger simulator step
        print("Stepping simulator...")
        vrep.simxSynchronousTrigger(clientID)

        # Get quadrotor initial position and orientation
        err, pos_new = vrep.simxGetObjectPosition(clientID, quad_handle, -1, vrep.simx_opmode_buffer)
        err, euler_new = vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_buffer)
        err, vel_new, angvel_new = vrep.simxGetObjectVelocity(clientID,
                quad_handle, vrep.simx_opmode_buffer)

        pos_new = np.asarray(pos_new)
        euler_new = np.asarray(euler_new)*10
        vel_new = np.asarray(vel_new)
        angvel_new = np.asarray(angvel_new)
        #euler_new[2]/=100

        valid = is_valid_state(pos_start, pos_new, euler_new)

        if valid:
            next_state = torch.FloatTensor(np.concatenate([euler_new, pos_new - pos_old]))
            #next_state = torch.FloatTensor(euler_new )

            next_extended_state=torch.FloatTensor([np.concatenate([next_state,np.asarray(propeller_vels)])])
        else:
            next_state = None
            next_extended_state = None

        reward=np.float32(0)

        memory.push(extended_state, torch.from_numpy(np.asarray([delta_forces],dtype=np.float32)), next_extended_state,
                                torch.from_numpy(np.asarray([[reward]])))
        state = next_state
        extended_state=next_extended_state
        pos_old = pos_new
        euler_old = euler_new
        vel_old = vel_new
        angvel_old = angvel_new
        print("Propeller Velocities:")
        print(propeller_vels)
        print("\n")

        pos_error = (pos_start + setpoint_delta[0:3]) - pos_new
        euler_error = (euler_start + setpoint_delta[3:6]) - euler_new
        euler_error %= 2*np.pi
        for i in range(len(euler_error)):
            if euler_error[i] > np.pi:
                euler_error[i] -= 2*np.pi
        vel_error = (vel_start + setpoint_delta[6:9]) - vel_new
        angvel_error = (angvel_start + setpoint_delta[9:12]) - angvel_new

        pos_error_all = np.vstack([pos_error_all,pos_error])
        euler_error_all = np.vstack([euler_error_all,euler_error])

        print("Errors (pos,ang):")
        print(pos_error,euler_error)
        print("\n")

        timestep += 1
        if not valid or timestep > ROLLOUT_LEN:
            np.savetxt('csv/pos_error{0}.csv'.format(test_num),
                       pos_error_all,
                       delimiter=',',
                       fmt='%8.4f')
            np.savetxt('csv/euler_error{0}.csv'.format(test_num),
                       euler_error_all,
                       delimiter=',',
                       fmt='%8.4f')

            print('RESET')
            # reset
            vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)
            time.sleep(0.1)

            # Setup V-REP simulation
            print("Setting simulator to async mode...")
            vrep.simxSynchronous(clientID, True)
            dt = .001
            vrep.simxSetFloatingParameter(clientID,
                                          vrep.sim_floatparam_simulation_time_step,
                                          dt,  # specify a simulation time step
                                          vrep.simx_opmode_oneshot)

            print("Loading scene...")
            vrep.simxLoadScene(clientID, scene_name, 0xFF, vrep.simx_opmode_blocking)

            # Get quadrotor handle
            err, quad_handle = vrep.simxGetObjectHandle(clientID, quad_name, vrep.simx_opmode_blocking)

            # Initialize quadrotor position and orientation
            vrep.simxGetObjectPosition(clientID, quad_handle, -1, vrep.simx_opmode_streaming)
            vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_streaming)
            vrep.simxGetObjectVelocity(clientID, quad_handle, vrep.simx_opmode_streaming)

            # Start simulation
            vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)

            for i in range(len(propellers)):
                vrep.simxClearFloatSignal(clientID, propellers[i], vrep.simx_opmode_oneshot)

            err, pos_start = vrep.simxGetObjectPosition(clientID, quad_handle, -1, vrep.simx_opmode_buffer)
            err, euler_start = vrep.simxGetObjectOrientation(clientID, quad_handle, -1, vrep.simx_opmode_buffer)
            err, vel_start, angvel_start = vrep.simxGetObjectVelocity(clientID,
                    quad_handle, vrep.simx_opmode_buffer)

            pos_start = np.asarray(pos_start)
            euler_start = np.asarray(euler_start)*10.
            vel_start = np.asarray(vel_start)
            angvel_start = np.asarray(angvel_start)*10.

            pos_old = pos_start
            euler_old = euler_start
            vel_old = vel_start
            angvel_old = angvel_start
            pos_new = pos_old
            euler_new = euler_old
            vel_new = vel_old
            angvel_new = angvel_old

            pos_error = (pos_start + setpoint_delta[0:3]) - pos_new
            euler_error = (euler_start + setpoint_delta[3:6]) - euler_new
            vel_error = (vel_start + setpoint_delta[6:9]) - vel_new
            angvel_error = (angvel_start + setpoint_delta[9:12]) - angvel_new

            pos_error_all = np.vstack([pos_error_all,pos_error])
            euler_error_all = np.vstack([euler_error_all,euler_error])

            state = [0 for i in range(n_input)]
            state = torch.FloatTensor(np.asarray(state)).view(-1, n_input)

            propeller_vels = [init_f, init_f, init_f, init_f]

            extended_state = torch.cat((state, torch.FloatTensor(np.asarray([propeller_vels]))), 1)
            print('duration: ',len(memory))
            memory = ReplayMemory(ROLLOUT_LEN)
            test_num += 1
            timestep = 1

if __name__ == '__main__':
    main()
