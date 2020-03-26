#!/usr/bin/env python3

import os
from stable_baselines.sac.policies import MlpPolicy as sac_MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as ddpg_MlpPolicy
from stable_baselines.common.policies import MlpPolicy as Common_MlpPolicy

from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import TRPO
from stable_baselines import DDPG
from stable_baselines import PPO1
from stable_baselines import SAC
from stable_baselines import logger
from os import system
import gym
import gym_SmartLoader.envs
import time
import numpy as np

n_steps = 0
save_interval = 2000
best_mean_reward = -np.inf

def save_fn(_locals, _globals):
    global model, n_steps, best_mean_reward, best_model_path, last_model_path
    if (n_steps + 1) % save_interval == 0:

        # Evaluate policy training performance
        mean_reward = round(float(np.mean(_locals['episode_rewards'][-101:-1])), 1)
        print(n_steps + 1, 'timesteps')
        print("Best mean reward: {:.2f} - Last mean reward: {:.2f}".format(best_mean_reward, mean_reward))
        # New best model, save the agent
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print("Saving new best model")
            model.save(best_model_path + '_rew_' + str(np.round(best_mean_reward, 2)))
        model.save(
            last_model_path + '_' + str(time.localtime().tm_mday) + '_' + str(time.localtime().tm_hour) + '_' + str(time.localtime().tm_min))
    n_steps += 1
    pass


def main():
    global model, best_model_path, last_model_path
    mission = 'PushStonesEnv' # Change according to algorithm
    env = gym.make(mission + '-v0').unwrapped
    train_model = False

    # Create log and model dir
    dir = 'stable_bl/' + mission
    os.makedirs(dir + '/model_dir/sac', exist_ok=True)

    if train_model:
        # create new folder
        # try:
        tests = os.listdir(dir + '/model_dir/sac')
        indexes = []
        for item in tests:
            indexes.append(int(item.split('_')[1]))
        k = max(indexes) + 1

        # except FileNotFoundError:
        #     os.makedirs(dir + '/log_dir/sac')
        #     k = 0

        model_dir = os.getcwd() + '/' + dir + '/model_dir/sac/test_{}'.format(str(k))

        best_model_path = model_dir
        last_model_path = model_dir

        num_timesteps = int(2e5)

        policy_kwargs = dict(layers=[64, 64, 64])

        log_dir = dir + '/log_dir/sac/test_{}'.format(str(k))
        logger.configure(folder=log_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])

        # SAC - start learning from scratch
        # model = SAC(sac_MlpPolicy, env, gamma=0.99, learning_rate=1e-4, buffer_size=500000,
        #      learning_starts=3000, train_freq=1, batch_size=64,
        #      tau=0.01, ent_coef='auto', target_update_interval=1,
        #      gradient_steps=1, target_entropy='auto', action_noise=None,
        #      random_exploration=0.0, verbose=2, tensorboard_log=log_dir,
        #      _init_setup_model=True, full_tensorboard_log=True,
        #      seed=None, n_cpu_tf_sess=None)

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

        model = SAC.load(dir + '/model_dir/sac/test_30_26_16_24.zip',
                         env=env, tensorboard_log=log_dir,
                         custom_objects=dict(learning_starts=0, learning_rate=4e-4,
                                             buffer_size=100000))
                                             # train_freq=8, gradient_steps=2,
                                             # batch_size=32))

        # learn
        model.learn(total_timesteps=num_timesteps, callback=save_fn)

        # PPO1
        # model = PPO1(Common_MlpPolicy, env, gamma=0.99, timesteps_per_actorbatch=256, clip_param=0.2, entcoeff=0.01,
        #      optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, adam_epsilon=1e-5,
        #      schedule='linear', verbose=0, tensorboard_log=None, _init_setup_model=True,
        #      policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)

        # TRPO
        # model = TRPO(MlpPolicy, env, timesteps_per_batch=4096, tensorboard_log=log_dir, verbose=1)
        # model.learn(total_timesteps=500000)
        # model.save(log_dir)

    else:
        # env = gym.make('PickUpEnv-v0')
        model = SAC.load(dir + '/model_dir/sac/test_31_26_17_43.zip', env=env, custom_objects=dict(learning_starts=0)) ### ADD NUM

        for _ in range(2):

            obs = env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                # print('state: ', obs[0:3], 'action: ', action)


if __name__ == '__main__':
    main()