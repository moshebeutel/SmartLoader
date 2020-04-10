#!/usr/bin/env python3
import argparse
import os
from stable_baselines.sac.policies import MlpPolicy as sac_MlpPolicy

from stable_baselines.ddpg.policies import MlpPolicy as ddpg_MlpPolicy
from stable_baselines.common.policies import MlpPolicy as Common_MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.gail import ExpertDataset
from stable_baselines import TRPO, A2C, DDPG, PPO1, PPO2, SAC, ACER, ACKTR, GAIL, DQN, HER, TD3, logger
from os import system
import gym
import gym_SmartLoader.envs
import time
import numpy as np
from typing import Dict
from tempfile import TemporaryFile
import csv

# for custom callbacks stable-baselines should be upgraded using -
# pip3 install stable-baselines[mpi] --upgrade
from stable_baselines.common.callbacks import BaseCallback

ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'her': HER,
    'sac': SAC,
    'ppo1': PPO1,
    'ppo2': PPO2,
    'trpo': TRPO,
    'td3': TD3,
    'gail': GAIL
}

def expert_dataset(name):
    # Benny's recordings to dict
    path = os.getcwd() + '/' + name
    numpy_dict = {
        'actions': np.load(path + '/act.npy'),
        'obs': np.load(path + '/obs.npy'),
        'rewards': np.load(path + '/rew.npy'),
        'episode_returns': np.load(path + '/ep_ret.npy'),
        'episode_starts': np.load(path + '/ep_str.npy')
    }  # type: Dict[str, np.ndarray]

    # for key, val in numpy_dict.items():
    #     print(key, val.shape)

    # dataset = TemporaryFile()
    save_path = os.getcwd() + '/dataset'
    os.makedirs(save_path)
    np.savez(save_path, **numpy_dict)

class ExpertDatasetLoader:
    dataset = None
    def __call__(self, force_load=False):
        if dataset is None or force_load:
            print('loading expert dataset')
            dataset = ExpertDataset(expert_path=(os.getcwd() + '/dataset.npz'), traj_limitation=-1)
        return dataset

# The following template is taken from stable-baselines tutorial

# class CustomCallback(BaseCallback):
#     """
#     A custom callback that derives from ``BaseCallback``.
#
#     :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
#     """
#     def __init__(self, verbose=0):
#         super(CustomCallback, self).__init__(verbose)
#         # Those variables will be accessible in the callback
#         # (they are defined in the base class)
#         # The RL model
#         # self.model = None  # type: BaseRLModel
#         # An alias for self.model.get_env(), the environment used for training
#         # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
#         # Number of time the callback was called
#         # self.n_calls = 0  # type: int
#         # self.num_timesteps = 0  # type: int
#         # local and global variables
#         # self.locals = None  # type: Dict[str, Any]
#         # self.globals = None  # type: Dict[str, Any]
#         # The logger object, used to report things in the terminal
#         # self.logger = None  # type: logger.Logger
#         # # Sometimes, for event callback, it is useful
#         # # to have access to the parent object
#         # self.parent = None  # type: Optional[BaseCallback]
#
#     def _on_training_start(self) -> None:
#         """
#         This method is called before the first rollout starts.
#         """
#         pass
#
#     def _on_rollout_start(self) -> None:
#         """
#         A rollout is the collection of environment interaction
#         using the current policy.
#         This event is triggered before collecting new samples.
#         """
#         pass
#
#     def _on_step(self) -> bool:
#         """
#         This method will be called by the model after each call to `env.step()`.
#
#         For child callback (of an `EventCallback`), this will be called
#         when the event is triggered.
#
#         :return: (bool) If the callback returns False, training is aborted early.
#         """
#         return True
#
#     def _on_rollout_end(self) -> None:
#         """
#         This event is triggered before updating the policy.
#         """
#         pass
#
#     def _on_training_end(self) -> None:
#         """
#         This event is triggered before exiting the `learn()` method.
#         """
#         pass

class CheckEvalCallback(BaseCallback):
    """
    A custom callback that checks agent's evaluation every predefined number of steps.
    :param model_dir: (str) directory path for model save
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    :param save_interval: (int) Number of timestamps between best mean model saves
    """
    def __init__(self, model_dir, verbose=0,  save_interval = 2000):
        super(CheckEvalCallback, self).__init__(verbose)
        self._best_model_path = model_dir
        self._last_model_path = model_dir
        self._best_mean_reward = -np.inf
        self._save_interval = save_interval

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        print('_on_training_start')

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        print('_on_rollout_start')

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        print("_on_step", self.num_timesteps)

        if (self.num_timesteps + 1) % self._save_interval == 0:
            # Evaluate policy training performance
            mean_reward = round(float(np.mean(self.locals['episode_rewards'][-101:-1])), 1)
            print(self.num_timesteps + 1, 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward: {:.2f}".format(self._best_mean_reward, mean_reward))
            # New best model, save the agent
            if mean_reward > self._best_mean_reward:
                self._best_mean_reward = mean_reward
                print("Saving new best model")
                self.model.save(self._best_model_path + '_rew_' + str(np.round(self._best_mean_reward, 2)))
            path = self._last_model_path + '_' + str(time.localtime().tm_mday) + '_' + str(time.localtime().tm_hour) + '_' + str(time.localtime().tm_min)
            self.model.save(path)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        print('_on_rollout_end')
        print('locals',self.locals)
        print('globals', self.globals)
        print('n_calls', self.n_calls)
        print('num_timesteps', self.num_timesteps)
        print('training_env', self.training_env)

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print('_on_training_end')


def data_saver(obs, act, rew, dones, ep_rew):
    np.save('/home/graphics/git/SmartLoader/saved_ep/obs', obs)
    np.save('/home/graphics/git/SmartLoader/saved_ep/act', act)
    np.save('/home/graphics/git/SmartLoader/saved_ep/rew', rew)

    ep_str = [False] * len(dones)
    ep_str[0] = True

    for i in range(len(dones) - 1):
        if dones[i]:
            ep_str[i + 1] = True

    np.save('/home/graphics/git/SmartLoader/saved_ep/ep_str', ep_str)
    np.save('/home/graphics/git/SmartLoader/saved_ep/ep_ret', ep_rew)

def build_model(algo, env, env_name, log_dir, expert_dataset=None):
    '''
    Initialize model according to algorithm, architecture and hyperparameters
    :param algo: (str) Name of rl algorithm - 'sac', 'ppo2' etc.
    :param env:gym.env
    :param env_name:(str)
    :param log_dir:(str)
    :param expert_dataset:(ExpertDataset)
    :return:
    '''
    model = None
    if algo == 'sac':
        model = SAC(sac_MlpPolicy, env, gamma=0.99, learning_rate=1e-4, buffer_size=500000,
                    learning_starts=0, train_freq=100, batch_size=64,
                    tau=0.01, ent_coef='auto', target_update_interval=1,
                    gradient_steps=1, target_entropy='auto', action_noise=None,
                    random_exploration=0.0, verbose=2, tensorboard_log=log_dir,
                    _init_setup_model=True, full_tensorboard_log=True,
                    seed=None, n_cpu_tf_sess=None)
    elif algo == 'ppo1':
        model = PPO1(Common_MlpPolicy, env, gamma=0.99, timesteps_per_actorbatch=256, clip_param=0.2, entcoeff=0.01,
                     optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, adam_epsilon=1e-5,
                     schedule='linear', verbose=0, tensorboard_log=None, _init_setup_model=True,
                     policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)
    elif algo == 'trpo':
        model = TRPO('MlpPolicy', env, timesteps_per_batch=4096, tensorboard_log=log_dir, verbose=1)
    elif algo == 'gail':
        assert expert_dataset is not None
        model = GAIL('MlpPolicy', env_name, expert_dataset, tensorboard_log=log_dir ,verbose=1)
    assert model is not None
    return model

def pretrain_model(dataset, model):
    # load dataset only once
    # expert_dataset('3_rocks_40_episodes')
    assert (dataset in locals() or dataset in globals()) and dataset is not None
    print('pretrain')
    model.pretrain(dataset, n_epochs=2000)

def record():
    # mission = 'PushStonesRecorder'  # Change according to algorithm
    mission = 'PushStonesEnv'
    env = gym.make(mission + '-v0').unwrapped
    num_episodes = 10
    obs = []
    actions = []
    rewards = []
    dones = []
    episode_rewards = []
    for episode in range(num_episodes):

        ob = env.reset()
        done = False
        print('Episode number ', episode)
        episode_reward = 0

        while not done:
            act = "recording"
            new_ob, reward, done, action = env.step(act)

            # ind = [0, 1, 2, 18, 21, 24]
            ind = [0, 1, 2]
            # print(ob)

            obs.append(ob)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            episode_reward = episode_reward + reward

            ob = new_ob

        episode_rewards.append(episode_reward)
    data_saver(obs, actions, rewards, dones, episode_rewards)

def play(dir, env):
    global model
    # env = gym.make('PickUpEnv-v0')
    model = SAC.load(dir + '/model_dir/sac/test_25_25_14_15', env=env,
                     custom_objects=dict(learning_starts=0))  ### ADD NUM
    for _ in range(2):

        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            # print('state: ', obs[0:3], 'action: ', action)

def train(args, dir, env, env_name):
    # create new folder
    try:
        tests = os.listdir(dir + '/model_dir/sac')
        indexes = []
        for item in tests:
            indexes.append(int(item.split('_')[1]))
        if not bool(indexes):
            k = 0
        else:
            k = max(indexes) + 1
    except FileNotFoundError:
        os.makedirs(dir + '/log_dir/sac')
        k = 0
    model_dir = os.getcwd() + '/' + dir + '/model_dir/sac/test_{}'.format(str(k))
    log_dir = dir + '/log_dir/sac/test_{}'.format(str(k))
    logger.configure(folder=log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    print('log directory created', log_dir)
    policy_kwargs = dict(layers=[64, 64, 64])
    dataset = ExpertDatasetLoader() if args.pretrain or args.algo == 'gail' else None
    model = build_model(algo=args.algo, env=env, env_name=env_name, log_dir=log_dir, expert_dataset=dataset)
    if args.pretrain:
        pretrain_model(dataset, model)
    # Load best model and continue learning
    # models = os.listdir(dir + '/model_dir/sac')
    # models_rew = (model for model in models if 'rew' in model)
    # ind, reward = [], []
    # for model in models_rew:
    #     ind.append(model.split('_')[1])
    #     reward.append(model.split('_')[3])
    # best_reward = max(reward)
    # best_model_ind = reward.index(best_reward)
    # k = ind[best_model_ind]
    # model = SAC.load(dir + '/model_dir/sac/test_' + k + '_rew_' + best_reward, env=env,
    #                  custom_objects=dict(learning_starts=0))
    # Load last saved model and continue learning
    # models = os.listdir(dir + '/model_dir/sac')
    # models_time = (model for model in models if 'rew' not in model)
    # ind, hour, min = [], [], []
    # for model in models_time:
    #     ind.append(model.split('_')[1])
    #     hour.append(model.split('_')[3])
    #     min.append(model.split('_')[4])
    # date = models_time[0].split('_')[2]
    # latest_hour = max(hour)
    # latest_hour_ind = [i for i, n in enumerate(hour) if n == latest_hour]
    # latest_min = max(min[latest_hour_ind])
    # latest_min_ind = min(latest_min)
    # k = ind[latest_min_ind]
    # model = SAC.load(dir + '/model_dir/sac/test_' + k + '_' + date + '_' + latest_hour[0] + '_' + latest_min + 'zip',
    #                  env=env, custom_objects=dict(learning_starts=0))
    # model = SAC.load(dir + '/model_dir/sac/test_53_rew_24383.0',
    #                  env=env, tensorboard_log=log_dir,
    #                  custom_objects=dict(learning_starts=0, learning_rate=2e-4,
    #                                      train_freq=8, gradient_steps=4, target_update_interval=4))
    # #                                              # batch_size=32))
    # Test the pre-trained model
    # env = model.get_env()
    # obs = env.reset()
    #
    # reward_sum = 0.0
    # for _ in range(1000):
    #     action, _ = model.predict(obs)
    #     obs, reward, done, _ = env.step(action)
    #     reward_sum += reward
    #     if done:
    #         print(reward_sum)
    #         reward_sum = 0.0
    #         obs = env.reset()
    #
    # env.close()

    # learn
    print("learning model type", type(model))
    eval_callback = CheckEvalCallback(model_dir)
    model.learn(total_timesteps=args.n_timesteps, callback=eval_callback)
    model.save(env_name)

def main(args):
    mission = 'PushStonesEnv'  # Change according to algorithm
    # mission = 'PushMultipleStones'
    env_name = mission + '-v0'
    env = gym.make(env_name).unwrapped
    print('gym env created', env_name, env)
    # Create log and model dir
    dir = 'stable_bl/' + mission
    os.makedirs(dir + '/model_dir/sac', exist_ok=True)

    jobs = ['train', 'record', 'BC_agent', 'play']
    job = jobs[0]

    if job == 'train':
        train(args, dir, env, env_name)
    elif job == 'record':
        record()
    elif job == 'play':
        play(dir, env)

def add_arguments(parser):
    parser.add_argument('--env', type=str, default="PushStonesEnv", help='environment ID')
    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='log_dir', type=str)
    parser.add_argument('-mdl', '--model-dir', help='model directory', default='model_dir', type=str)
    parser.add_argument('--algo', help='RL Algorithm', default='sac', type=str, required=False,
                        choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=int(1e6), type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1, type=int)
    parser.add_argument('--eval-freq', help='Evaluate the agent every n steps (if negative, no evaluation)',
                        default=10000, type=int)
    parser.add_argument('--eval-episodes', help='Number of episodes to use for evaluation', default=5, type=int)
    parser.add_argument('--save-freq', help='Save the model every n steps (if negative, no checkpoint)', default=-1,
                        type=int)
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1, type=int)
    parser.add_argument('--pretrain', help='Evaluate pretrain phase', default=False, type=bool)
    parser.add_argument('--load-expert-dataset', help='Load Expert Dataset', default=False, type=bool)
    # parser.add_argument('-params', '--hyperparams', type=str, nargs='+', action=StoreDict,
    #                     help='Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)')
    # parser.add_argument('-uuid', '--uuid', action='store_true', default=False,
    #                     help='Ensure that the run has a unique ID')
    # parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict,
    #                     help='Optional keyword argument to pass to the env constructor')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)
