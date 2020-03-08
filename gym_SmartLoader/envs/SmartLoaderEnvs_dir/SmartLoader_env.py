# !/usr/bin/env python3
# building custom gym environment:
# # https://medium.com/analytics-vidhya/building-custom-gym-environments-for-reinforcement-learning-24fa7530cbb5
# # for testing
import gym

# class PickUpEnv(gym.Env):
#     def __init__(self):
#         print("Environment initialized")
#     def step(self):
#         print("Step successful!")
#     def reset(self):
#         print("Environment reset")

import sys
import time
from src.EpisodeManager import *
from src.Unity2RealWorld import *
import gym
from gym import spaces
import numpy as np
from math import pi as pi
import rospy
from std_msgs.msg import Header
from std_msgs.msg import Int32, Bool
from std_msgs.msg import String
from sensor_msgs.msg import Joy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, TwistStamped


class BaseEnv(gym.Env):

    def VehiclePositionCB(self,stamped_pose):
        x = stamped_pose.pose.position.x
        y = stamped_pose.pose.position.y
        z = stamped_pose.pose.position.z
        self.world_state['VehiclePos'] = np.array([x,y,z])

        qx = stamped_pose.pose.orientation.x
        qy = stamped_pose.pose.orientation.y
        qz = stamped_pose.pose.orientation.z
        qw = stamped_pose.pose.orientation.w
        self.world_state['VehicleOrien'] = np.array([qx,qy,qz,qw])

        # rospy.loginfo('position is:' + str(stamped_pose.pose))

    def VehicleVelocityCB(self, stamped_twist):
        vx = stamped_twist.twist.linear.x
        vy = stamped_twist.twist.linear.y
        vz = stamped_twist.twist.linear.z
        self.world_state['VehicleLinearVel'] = np.array([vx,vy,vz])

        wx = stamped_twist.twist.angular.x
        wy = stamped_twist.twist.angular.y
        wz = stamped_twist.twist.angular.z
        self.world_state['VehicleAngularVel'] = np.array([wx,wy,wz])

        # rospy.loginfo('velocity is:' + str(stamped_twist.twist))

    def ArmHeightCB(self, data):
        height = data.data
        self.world_state['ArmHeight'] = np.array([height])

        # rospy.loginfo('arm height is:' + str(height))

    def BladeImuCB(self, imu):
        qx = imu.orientation.x
        qy = imu.orientation.y
        qz = imu.orientation.z
        qw = imu.orientation.w
        self.world_state['BladeOrien'] = np.array([qx,qy,qz,qw])

        wx = imu.angular_velocity.x
        wy = imu.angular_velocity.y
        wz = imu.angular_velocity.z
        self.world_state['BladeAngularVel'] = np.array([wx,wy,wz])

        ax = imu.linear_acceleration.x
        ay = imu.linear_acceleration.y
        az = imu.linear_acceleration.z
        self.world_state['BladeLinearAcc'] = np.array([ax,ay,az])

        # rospy.loginfo('blade imu is:' + str(imu))

    def VehicleImuCB(self, imu):
        qx = imu.orientation.x
        qy = imu.orientation.y
        qz = imu.orientation.z
        qw = imu.orientation.w
        self.world_state['VehicleOrien'] = np.array([qx,qy,qz,qw])

        wx = imu.angular_velocity.x
        wy = imu.angular_velocity.y
        wz = imu.angular_velocity.z
        self.world_state['VehicleAngularVel'] = np.array([wx,wy,wz])

        ax = imu.linear_acceleration.x
        ay = imu.linear_acceleration.y
        az = imu.linear_acceleration.z
        self.world_state['VehicleLinearAcc'] = np.array([ax,ay,az])

        # rospy.loginfo('vehicle imu is:' + str(imu))

    def StonePositionCB(self, data, arg):
        position = data.pose.position
        stone = arg

        x = position.x
        y = position.y
        z = position.z
        self.stones['StonePos' + str(stone)] = np.array([x,y,z])

        # rospy.loginfo('stone ' + str(stone) + ' position is:' + str(position))

    def StoneIsLoadedCB(self, data, arg):
        question = data.data
        stone = arg
        self.stones['StoneIsLoaded' + str(stone)] = question

        # rospy.loginfo('Is stone ' + str(stone) + ' loaded? ' + str(question))

    def do_action(self, action):
        # TODO
        rospy.logdebug(self.joyactions)
        joymessage = Joy()
        # self.setDebugAction(action) # DEBUG
        self.setAction(action)
        joymessage.axes = [self.joyactions["0"], 0., self.joyactions["2"], self.joyactions["3"], self.joyactions["4"], self.joyactions["5"], 0., 0.]

        # while not rospy.is_shutdown():
        self.pubjoy.publish(joymessage)
        rospy.logdebug(joymessage)
            # self.rate.sleep()

    def setDebugAction(self, values): # DEBUG
        self.joyactions["0"] = values["0"]
        self.joyactions["2"] = values["2"]
        self.joyactions["3"] = values["3"]
        self.joyactions["4"] = values["4"]
        self.joyactions["5"] = values["5"]

    def debugAction(self):
        actionValues = {"0": 0., "2": 1., "3": 0., "4": 0., "5": -1.}  # drive forwards
        return actionValues

    def setAction(self, values):
        # translate chosen action (array) to joystick action (dict)
        self.joyactions["0"] = values[0]
        self.joyactions["2"] = values[1]
        self.joyactions["3"] = values[2]
        self.joyactions["4"] = values[3]
        self.joyactions["5"] = values[4]



    def __init__(self,numStones=1):
        super(BaseEnv, self).__init__()

        print('environment created!')

        self.world_state = {}
        self.stones = {}
        self.joyactions = {}
        self.simOn = False
        self.numStones = numStones

        ## ROS messages
        rospy.init_node('slagent', anonymous=False)
        self.rate = rospy.Rate(10)  # 10hz

        # Define Subscribers
        self.vehiclePositionSub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.VehiclePositionCB)
        self.vehicleVelocitySub = rospy.Subscriber('mavros/local_position/velocity', TwistStamped, self.VehicleVelocityCB)
        self.heightSub = rospy.Subscriber('arm/height', Int32, self.ArmHeightCB)
        self.bladeImuSub = rospy.Subscriber('arm/blade/Imu', Imu, self.BladeImuCB)
        self.vehicleImuSub = rospy.Subscriber('mavros/imu/data', Imu, self.VehicleImuCB)

        self.stonePoseSubList = []
        self.stoneIsLoadedSubList = []

        for i in range(1, self.numStones+1):
            topicName = 'stone/' + str(i) + '/Pose'
            self.stonePoseSubList.append(rospy.Subscriber(topicName, PoseStamped, self.StonePositionCB, i))
            topicName = 'stone/' + str(i) + '/IsLoaded'
            self.stoneIsLoadedSubList.append(rospy.Subscriber(topicName, Bool, self.StoneIsLoadedCB, i))

        # Define Publishers
        self.joyactions["0"] = 0.                            # steer
        self.joyactions["2"] = 1.                            # speed backwards
        self.joyactions["3"] = 0.                            # blade pitch
        self.joyactions["4"] = 0.                            # arm height
        self.joyactions["5"] = 1.                            # speed forwards

        self.pubjoy = rospy.Publisher("joy", Joy, queue_size=10)

        ## Define gym space
        # action = [steering, armHeight, throttle, armPitch]
        # min_action = np.array([-100,   0, -100, -pi/2])
        # max_action = np.array([ 100, 100,  100, pi/2])
        # TODO: limits normalized as joystick action [-1,1]?
        min_action = np.array([-1., -1., -1., -1., -1.])
        max_action = np.array([ 1.,  1.,  1.,  1.,  1.])

        self.action_space = spaces.Box(low=min_action,high=max_action)
        self.observation_space = self.obs_space_init()

        # self.init_env()


    def obs_space_init(self):
        # obs = [local_pose:(x,y,z), local_orien_quat:(x,y,z,w)
        #        velocity: linear:(vx,vy,vz), angular:(wx,wy,wz)
        #        arm_height: h
        #        arm_imu: orein_quat:(x,y,z,w), vel:(wx,wy,wz)
        #        stone<id>: pose:(x,y,z), isLoaded:bool]
        # TODO: update all limits


        min_pos = np.array(3*[0.])
        max_pos = np.array(3*[500.]) # size of ground in Unity - TODO: update to room size
        min_quat = np.array(4*[-1.])
        max_quat = np.array(4*[ 1.])
        min_lin_vel = np.array(3*[-5.])
        max_lin_vel = np.array(3*[ 5.])
        min_ang_vel = np.array(3*[-pi/4])
        max_ang_vel = np.array(3*[ pi/4])
        min_arm_height = np.array([0.])
        max_arm_height = np.array([100.])
        # SPACES DICT:
        # obsSpace = spaces.Dict({"VehiclePos": spaces.Box(low=self.min_pos, high=self.max_pos),
        #                         "VehicleOrien": spaces.Box(low=min_quat, high=max_quat),
        #                         "VehicleLinearVel": spaces.Box(low=min_lin_vel, high=max_lin_vel),
        #                         "VehicleAngularVel": spaces.Box(low=min_ang_vel, high=max_ang_vel),
        #                         "ArmHeight": spaces.Box(low=np.array([0.]), high=np.array([100.])),
        #                         "BladeOrien": spaces.Box(low=min_quat, high=max_quat),
        #                         "BladeAngularVel": spaces.Box(low=min_ang_vel, high=max_ang_vel),
        #                         "Stones": spaces.Dict(self.obs_stones())})

        # SPACES BOX - WITHOUT IS LOADED
        low  = np.concatenate((min_pos,min_quat,min_lin_vel,min_ang_vel,min_arm_height,min_quat,min_ang_vel), axis=None)
        high = np.concatenate((max_pos,max_quat,max_lin_vel,max_ang_vel,max_arm_height,max_quat,max_ang_vel), axis=None)
        for ind in range(1, self.numStones + 1):
            low  = np.concatenate((low,min_pos), axis=None)
            high = np.concatenate((high,max_pos), axis=None)
        obsSpace = spaces.Box(low=low, high=high)

        return obsSpace


    # def obs_stones(self):
    #     stone_dict = {}
    #     for ind in range(1, self.numStones+1):
    #         stone_dict['StonePos' + str(ind)] = spaces.Box(low=self.min_pos, high=self.max_pos)
    #         stone_dict['StoneIsLoaded' + str(ind)] = spaces.Discrete(2)
    #
    #     return stone_dict


    def current_obs(self):
        # rospy.loginfo(self.world_state)

        # SPACES DICT - fit current obs data to obs_space.Dict structure
        # obs = {}
        # keys = {'VehiclePos','VehicleOrien','VehicleLinearVel','VehicleAngularVel','ArmHeight','BladeOrien','BladeAngularVel'}
        # obs = {k:v for (k,v) in self.world_state.items() if k in keys}
        # obs['Stones'] = {}
        # for ind in range(1, self.numStones+1):
        #     obs['Stones']['StonePos' + str(ind)] = self.stones['StonePos' + str(ind)]
        #     if 'StoneIsLoaded' + str(ind) in self.stones:
        #         obs['Stones']['StoneIsLoaded' + str(ind)] = self.stones['StoneIsLoaded' + str(ind)]

        # SPACES BOX - fit current obs data to obs_space.Box structure
        obs = np.array([])
        keys = {'VehiclePos', 'VehicleOrien', 'VehicleLinearVel', 'VehicleAngularVel', 'ArmHeight', 'BladeOrien', 'BladeAngularVel'}
        while True: # wait for all topics to arrive
            if all(key in self.world_state for key in keys):
                break
        for key in keys:
            obs = np.concatenate((obs, self.world_state[key]), axis=None)
        for ind in range(1, self.numStones + 1):
            obs = np.concatenate((obs, self.stones['StonePos'+str(ind)]), axis=None)

        return obs


    def init_env(self):
        if self.simOn:
            self.episode.killSimulation()

        self.episode = EpisodeManager()
        self.episode.generateAndRunWholeEpisode(typeOfRand="verybasic")
        self.simOn = True


    def reset(self):
        # what happens when episode is done
        # rospy.loginfo('reset func called')

        # clear all
        self.world_state = {}
        self.stones = {}
        self.steps = 0
        self.total_reward = 0

        # initial state depends on environment (mission)
        self.init_env()

        # wait for simulation to set up
        while True: # wait for all topics to arrive
            # change to 2*numStones when IsLoaded is fixed
            if bool(self.world_state) and bool(self.stones) and len(self.stones) >= self.numStones:
                break

        # wait for simulation to stabilize, stones stop moving
        time.sleep(5)
        # self.last_stone_pos = {}
        # static = 0
        # while True:
        #     self.current_stone_pos = self.stones
        #     if bool(self.last_stone_pos):  # don't enter first time when last_stone_height is empty
        #         # print('current pos = ', self.current_stone_pos, 'last pos =', self.last_stone_pos)
        #         if self.current_stone_pos == self.last_stone_pos:
        #             static += 1
        #     self.last_stone_pos = self.current_stone_pos
        #     if static > 5000:
        #         break


        # get observation from simulation
        obs = self.current_obs()
        # rospy.loginfo(obs)

        return obs


    def step(self, action):
        # rospy.loginfo('step func called')

        # send action to simulation
        self.do_action(action)

        # get observation from simulation
        obs = self.current_obs()

        # calc step reward and add to total
        r_t = self.reward_func()

        # check if done
        done, final_reward, reset = self.end_of_episode()

        step_reward = r_t + final_reward
        self.total_reward = self.total_reward + step_reward
        print('total reward = ', self.total_reward)

        info = {"state": obs, "action": action, "reward": self.total_reward, "step": self.steps, "reset reason": reset}

        return obs, step_reward, done, info


    def reward_func(self):
        raise NotImplementedError

    def end_of_episode(self):
        raise NotImplementedError

    def render(self, mode='human'):
        pass


    def run(self):
        # DEBUG
        obs = self.reset()
        rospy.loginfo(obs)
        done = False
        while not done:
            action = self.debugAction()
            obs, _, done, _ = self.step(action)
            rospy.loginfo(obs)
            self.rate.sleep()


class PickUpEnv(BaseEnv):
    def __init__(self, numStones=1): #### Number of stones ####
        BaseEnv.__init__(self, numStones)
        # initial state depends on environment (mission)
        # self.stones_on_ground = self.numStones*[True]
        self.current_stone_height = {}
        self.last_stone_height = {}

    def reward_func(self):
        # reward per step
        reward = -0.1

        # IS LOADED
        # SINGLE_STONE_IN_BLADE = 1000
        # for stone in range(self.numStones+1):
        #     if self.stones_on_ground[stone]:
        #         if 'StoneIsLoaded' + str(stone) in self.stones:
        #             if self.stones['StoneIsLoaded' + str(stone)]:
        #                 reward += SINGLE_STONE_IN_BLADE
        #                 self.stones_on_ground[stone] = False

        # Stone height
        STONE_UP = 10
        for stone in range(1, self.numStones + 1):
            self.current_stone_height['stoneHeight'+str(stone)] = self.stones['StonePos'+str(stone)][2]

            if bool(self.last_stone_height): # don't enter first time when last_stone_height is empty
                if self.current_stone_height['stoneHeight'+str(stone)] > self.last_stone_height['stoneHeight'+str(stone)]:
                    reward += STONE_UP
                    rospy.loginfo('---------------- positive reward! ----------------')

            self.last_stone_height['stoneHeight' + str(stone)] = self.current_stone_height['stoneHeight' + str(stone)]

        return reward

    def end_of_episode(self):
        done = False
        reset = 'No'
        final_reward = 0

        MAX_STEPS = 6000 # 10 = 1 second
        if self.steps > MAX_STEPS:
            done = True
            reset = 'limit time steps'
            print('----------------', reset ,'----------------')
            self.episode.killSimulation()
            self.simOn = False

        # IS LOADED
        # if not all(self.stones_on_ground):
        #     done = True
        #     reset = 'sim success'
        #     print('----------------', reset, '----------------')
        #     self.episode.killSimulation()

        # Stone height
        HEIGHT_LIMIT = 100
        if all(height >= HEIGHT_LIMIT for height in self.current_stone_height.values()):
            done = True
            reset = 'sim success'
            print('----------------', reset, '----------------')
            self.episode.killSimulation()
            self.simOn = False

        self.steps += 1

        return done, final_reward, reset


class PutDownEnv(BaseEnv):
    def __init__(self, numStones=1):
        BaseEnv.__init__(self, numStones)
        self.desired_stone_pose = [250, 250]
        # initial state depends on environment (mission)
        # send reset to simulation with initial state
        self.stones_on_ground = self.numStones*[False]

    def reward_func(self):
        # reward per step
        reward = -0.1

        return reward

    def end_of_episode(self):
        done = False
        reset = 'No'
        final_reward = 0

        MAX_STEPS = 6000
        if self.steps > MAX_STEPS:
            done = True
            reset = 'limit time steps'
            print('----------------', reset ,'----------------')

        if all(self.stones_on_ground):
            done = True
            reset = 'sim success'
            print('----------------', reset, '----------------')
            final_reward = self.succ_reward()

        self.steps += 1

        return done, final_reward, reset

    def succ_reward(self):
        # end of episode reward depending on distance of stones from desired location
        reward = 1000

        for ind in range(1, self.numStones + 1):
            curret_pos = self.stones['StonePos' + str(ind)][0:2]
            dis = np.linalg.norm(curret_pos - self.desired_stone_pose)
            reward -= dis

        return reward


class MoveWithStonesEnv(BaseEnv):
    def __init__(self, numStones=1):
        BaseEnv.__init__(self, numStones)
        self.desired_vehicle_pose = [250,250]
        # initial state depends on environment (mission)
        # send reset to simulation with initial state
        self.stones_on_ground = self.numStones*[False]

    def reward_func(self):
        # reward per step
        reward = -0.1

        SINGLE_STONE_FALL = 1000
        for stone in range(1, self.numStones + 1):
            if not self.stones_on_ground[stone]:
                if not self.stones['StoneIsLoaded' + str(stone)]:
                    reward -= SINGLE_STONE_FALL
                    self.stones_on_ground[stone] = True

        return reward

    def end_of_episode(self):
        done = False
        reset = 'No'
        final_reward = 0

        MAX_STEPS = 6000
        SUCC_REWARD = 1000
        if self.steps > MAX_STEPS:
            done = True
            reset = 'limit time steps'
            print('----------------', reset, '----------------')

        if self.got_to_desired_pose():
            done = True
            reset = 'sim success'
            print('----------------', reset, '----------------')
            final_reward = SUCC_REWARD

        self.steps += 1

        return done, final_reward, reset

    def got_to_desired_pose(self):
        # check if vehicle got within tolerance of desired position
        success = False

        current_pos = self.world_state['VehiclePos'][0:2]
        dis = np.linalg.norm(current_pos - self.desired_vehicle_pose)
        TOLERANCE = 0.1
        if dis < TOLERANCE:
            success = True

        return success


class PushStonesEnv(BaseEnv):
    def __init__(self, numStones=1):
        BaseEnv.__init__(self, numStones)
        self.desired_stone_pose = np.random.uniform(0, 500, 2) # [250, 250]
        # initial state depends on environment (mission)
        # send reset to simulation with initial state
        self.current_dis = {}
        self.last_dis = {}

    def reward_func(self):
        # reward per step
        reward = -0.1

        # K = 0.001
        # dis = self.dis_from_desired_pose()
        # reward += -K*np.mean(dis)

        STONE_CLOSER = 10
        self.current_dis = self.dis_from_desired_pose()
        if bool(self.last_dis): # don't enter first time when last_stone_height is empty
            if any(True for curr, last in zip(self.current_dis, self.last_dis) if curr < last):
                reward += STONE_CLOSER
                rospy.loginfo('---------------- positive reward! ----------------')

        self.last_dis = self.current_dis

        return reward

    def end_of_episode(self):
        done = False
        reset = 'No'
        final_reward = 0

        MAX_STEPS = 6000
        SUCC_REWARD = 1000
        if self.steps > MAX_STEPS:
            done = True
            reset = 'limit time steps'
            print('----------------', reset, '----------------')

        if self.got_to_desired_pose():
            done = True
            reset = 'sim success'
            print('----------------', reset, '----------------')
            final_reward = SUCC_REWARD

        self.steps += 1

        return done, final_reward, reset

    def dis_from_desired_pose(self):
        # list of stones distances from desired pose

        dis = []
        for stone in range(1, self.numStones + 1):
            current_pos = self.stones['StonePos' + str(stone)][0:2]
            dis.append(np.linalg.norm(current_pos - self.desired_stone_pose))

        return dis

    def got_to_desired_pose(self):
        # check if all stones within tolerance from desired pose

        success = False
        dis = self.dis_from_desired_pose()

        TOLERANCE = 0.1
        if all(item < TOLERANCE for item in dis):
            success = True

        return success



# DEBUG
# if __name__ == '__main__':
#     from stable_baselines.common.env_checker import check_env
#
#     env = PickUpEnv()
#     # It will check your custom environment and output additional warnings if needed
#     check_env(env)

#     node = PickUpEnv(2)
#     node.run()
