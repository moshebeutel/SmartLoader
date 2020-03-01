# # building custom gym environment:
# # https://medium.com/analytics-vidhya/building-custom-gym-environments-for-reinforcement-learning-24fa7530cbb5
# # for testing
# import gym

# class MoveRockEnv(gym.Env):
#     def __init__(self):
#         print("Environment initialized")
#     def step(self):
#         print("Step successful!")
#     def reset(self):
#         print("Environment reset")

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

        rospy.loginfo('position is:' + str(stamped_pose.pose))

    def VehicleVelocityCB(self, stamped_twist):
        vx = stamped_twist.twist.linear.x
        vy = stamped_twist.twist.linear.y
        vz = stamped_twist.twist.linear.z
        self.world_state['VehicleLinearVel'] = np.array([vx,vy,vz])

        wx = stamped_twist.twist.angular.x
        wy = stamped_twist.twist.angular.y
        wz = stamped_twist.twist.angular.z
        self.world_state['VehicleAngularVel'] = np.array([wx,wy,wz])        

        rospy.logdebug('velocity is:' + str(stamped_twist.twist))

    def ArmHeightCB(self, data):
        height = data.data
        self.world_state['ArmHeight'] = np.array([height])

        rospy.logdebug('arm height is:' + str(height))

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

        rospy.logdebug('blade imu is:' + str(imu))

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

        rospy.logdebug('vehicle imu is:' + str(imu))

    def StonePositionCB(self, data, arg):
        position = data.pose
        stone = arg
        
        x = position.x
        y = position.y
        z = position.z
        self.stones['StonePos' + str(stone)] = np.array([x,y,z])

        rospy.logdebug('stone ' + str(stone) + ' position is:' + str(position))

    def StoneIsLoadedCB(self, data, arg):
        question = data.data
        stone = arg
        self.stones['StoneIsLoaded' + str(stone)] = question

        rospy.logdebug('Is stone ' + str(stone) + ' loaded? ' + str(question))

    def do_action(self, action):
        # TODO: get meaning of actions from michel/yaniv
        rospy.logdebug(self.joyactions)
        joymessage = Joy()
        joymessage.axes = [self.joyactions["0"], self.joyactions["1"], self.joyactions["2"], self.joyactions["3"], self.joyactions["4"], self.joyactions["5"]]

        self.pubjoy.publish(joymessage)
        rospy.logdebug("Here also I am: ")
        rospy.logdebug(joymessage)

    def setDebugAction(self): # DEBUG
        actionValues = {"0": 0.0, "1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4, "5": 0.5}
        self.SetActionValues(actionValues)

        return actionValues

    def SetActionValues(self, values):
        # TODO: translate chosen network's action (1x4 array) to joystick action 
        self.joyactions["0"] = values["0"]
        self.joyactions["1"] = values["1"]
        self.joyactions["2"] = values["2"]
        self.joyactions["3"] = values["3"]
        self.joyactions["4"] = values["4"]
        self.joyactions["5"] = values["5"]


    def __init__(self,numStones):
        
        self.world_state = {}
        self.stones = {}
        self.joyactions = {}
        self.numStones = numStones

        ## ROS messages
        rospy.init_node('slagent', anonymous=False)

        # Define Subscribers
        self.vehiclePositionSub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.VehiclePositionCB)
        self.vehicleVelocitySub = rospy.Subscriber('mavros/local_position/velocity', TwistStamped, self.VehicleVelocityCB)
        self.heightSub = rospy.Subscriber('arm/height', Int32, self.ArmHeightCB)
        self.bladeImuSub = rospy.Subscriber('arm/blade/Imu', Imu, self.BladeImuCB)
        self.vehicleImuSub = rospy.Subscriber('mavros/imu/data', Imu, self.VehicleImuCB)

        self.stonePoseSubList = []
        self.stoneIsLoadedSubList = []
        
        for i in range(self.numStones):
            topicName = 'stone/' + str(i) + '/Pose'
            self.stonePoseSubList.append(rospy.Subscriber(topicName, PoseStamped, self.StonePositionCB, i))
            topicName = 'stone/' + str(i) + '/IsLoaded'
            self.stoneIsLoadedSubList.append(rospy.Subscriber(topicName, Bool, self.StoneIsLoadedCB, i))

        # Define Publishers
        self.joyactions["0"] = self.joyactions["1"] = self.joyactions["2"] = 0
        self.joyactions["3"] = self.joyactions["4"] = self.joyactions["5"] = 0

        self.pubjoy = rospy.Publisher("joy", Joy, queue_size=10)

        ## Define gym space
        # action = [steering, armHeight, throttle, armPitch]
        min_action = np.array([-100,   0, -100, -pi/2])
        max_action = np.array([ 100, 100,  100, pi/2])

        self.action_space = spaces.Box(low=min_action,high=max_action)
        self.observation_space = self.obs_space_init()

        self.init_env()


    def obs_space_init(self):
        # obs = [local_pose:(x,y,z), local_orien_quat:(x,y,z,w)
        #        velocity: linear:(vx,vy,vz), angular:(wx,wy,wz)
        #        arm_height: h
        #        arm_imu: orein_quat:(x,y,z,w), vel:(wx,wy,wz)
        #        stone<id>: pose:(x,y,z), isLoaded:bool]
        # TODO: update all limits
        
        self.min_pos = np.array([  0.,  0.,  0.])
        self.max_pos = np.array([500.,500.,500.]) # size of ground in Unity
        min_quat = np.array([-1.,-1.,-1.,-1.])
        max_quat = np.array([ 1., 1., 1., 1.])
        min_lin_vel, max_lin_vel = -5., 5.
        min_ang_vel, max_ang_vel =  -pi/4, pi/4
        obsSpace = spaces.Dict({"VehiclePos": spaces.Box(low=self.min_pos, high=self.max_pos),
                                "VehicleOrien": spaces.Box(low=min_quat, high=max_quat),
                                "VehicleLinearVel": spaces.Box(low=min_lin_vel, high=max_lin_vel),
                                "VehicleAngularVel": spaces.Box(low=min_ang_vel, high=max_ang_vel),
                                "ArmHeight": spaces.Box(low=0, high=100),
                                "BladeOrien": spaces.Box(low=min_quat, high=max_quat),
                                "BladeAngularVel": spaces.Box(low=min_ang_vel, high=max_ang_vel),
                                "Stones": spaces.Dict(self.obs_stones())})
        return obsSpace


    def obs_stones(self):
        stone_dict = {}
        for ind in range(self.numStones):
            stone_dict['StonePos' + str(ind)] = spaces.Box(low=self.min_pos, high=self.max_pos)
            stone_dict['StoneIsLoaded' + str(ind)] = spaces.Discrete(2)

        return stone_dict


    def current_obs(self):
        # fit current obs data to obs_space.Dict structure
        obs = {}
        obs = {k:v for (k,v) in self.world_state.items() if self.observation_space.contains(k)}
        obs['Stones'] = {k:v for (k,v) in self.stones.items() if self.observation_space.contains(k)}
        
        return obs


    def init_env(self):
        pass


    def reset(self):
        # what happens when episode is done
        self.steps = 0
        self.total_reward = 0

        # initial state depends on environment (mission)
        # TODO: send reset to simulation with initial state
        self.init_world_state()

        # wait for vehicle to arrive to desired position

        # get observation from simulation
        obs = self.current_obs()

        return obs


    def step(self, action):
        # send action to simulation
        self.do_action(action)

        # get observation from simulation
        obs = self.current_obs()

        # calc step reward and add to total
        r_t = self.reward_func()
        self.total_reward = self.total_reward + r_t

        # check if done
        done, reset = self.end_of_episode()

        info = {"state" : obs, "action": action, "reward": self.total_reward, "step": self.steps, "reset reason": reset}

        return obs, self.total_reward, done, info


    def init_world_state(self):
        raise NotImplementedError

    def reward_func(self):
        raise NotImplementedError

    def end_of_episode(self):
        raise NotImplementedError

    def render(self):
        pass

    def run(self):
        # DEBUG
        _ = self.reset()
        action = self.setDebugAction()
        done = False
        while not done:
            _, _, done, _ = self.step(action)



class PickUpEnv(BaseEnv):
    def __init__(self, numStones):
        BaseEnv.__init__(self, numStones)
        
    def init_world_state(self):
        # initial state depends on environment (mission)
        # send reset to simulation with initial state 
        self.stones_on_ground = self.numStones*[True]
    
    def reward_func(self):
        # reward per step
        reward = -0.1

        SINGLE_STONE_IN_BLADE = 1000
        for stone in range(self.numStones):
            if self.stones_on_ground[stone]:
                if self.stones['StoneIsLoaded' + str(stone)]:
                    reward += SINGLE_STONE_IN_BLADE
                    self.stones_on_ground[stone] = False

        return reward

    def end_of_episode(self):
        done = False
        reset = 'No'

        MAX_STEPS = 5000
        SUCC_REWARD = 10000
        if self.steps > MAX_STEPS:
            done = True
            reset = 'limit time steps'
            print('----------------', reset ,'----------------')
        
        if all(self.stones_on_ground == False):
            done = True
            reset = 'sim success'
            print('----------------', reset, '----------------')
            self.total_reward += SUCC_REWARD  
        
        self.steps += 1

        return done, reset


class PutDownEnv(BaseEnv):
    def __init__(self, numStones):
        BaseEnv.__init__(self, numStones)
        self.desired_stone_pose = [250,250]

    def init_world_state(self):
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

        MAX_STEPS = 5000
        if self.steps > MAX_STEPS:
            done = True
            reset = 'limit time steps'
            print('----------------', reset ,'----------------')
        
        if all(self.stones_on_ground == True):
            done = True
            reset = 'sim success'
            print('----------------', reset, '----------------')
            self.total_reward += self.succ_reward()  
        
        self.steps += 1

        return done, reset

    def succ_reward(self):
        # end of episode reward depending on distance of stones from desired location
        reward = 10000

        for ind in range(self.numStones):
            curret_pos = self.stones['StonePos' + str(ind)][0:2]
            dis = np.linalg.norm(curret_pos - self.desired_stone_pose)
            reward -= dis

        return reward


class MoveWithStonesEnv(BaseEnv):
    def __init__(self, numStones):
        BaseEnv.__init__(self, numStones)
        self.desired_vehicle_pose = [250,250]
        
    def init_world_state(self):
        # initial state depends on environment (mission)
        # send reset to simulation with initial state        
        self.stones_on_ground = self.numStones*[False]

    def reward_func(self):
        # reward per step
        reward = -0.1

        SINGLE_STONE_FALL = 10000
        for stone in range(self.numStones):
            if not self.stones_on_ground[stone]:
                if not self.stones['StoneIsLoaded' + str(stone)]:
                    reward -= SINGLE_STONE_FALL
                    self.stones_on_ground[stone] = True

        return reward

    def end_of_episode(self):
        done = False
        reset = 'No'

        MAX_STEPS = 5000
        SUCC_REWARD = 10000
        if self.steps > MAX_STEPS:
            done = True
            reset = 'limit time steps'
            print('----------------', reset ,'----------------')
        
        if self.got_to_desired_pose():
            done = True
            reset = 'sim success'
            print('----------------', reset, '----------------')
            self.total_reward += SUCC_REWARD  
        
        self.steps += 1

        return done, reset

    def got_to_desired_pose(self):
        # check if vehicle got within tolrance of desired position
        success = False

        curret_pos = self.world_state['VehiclePos'][0:2]
        dis = np.linalg.norm(curret_pos - self.desired_vehicle_pose)
        TOLERANCE = 0.1
        if dis < TOLERANCE:
            success = True
    
        return success


if __name__ == '__main__':
    node = PickUpEnv(1)
    node.run()
        