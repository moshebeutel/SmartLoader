import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines import DDPG
from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy
# from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    return True

# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
# env = gym.make('MountainCarContinuous-v0')
env = gym.make('MoveRockEnv-v0')
env = Monitor(env, log_dir, allow_early_resets=True)
# Automatically normalize the input features
# env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

# # the noise objects for DDPG
# n_actions = env.action_space.shape[-1]
# param_noise = None
# action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
model = DDPG(MlpPolicy, env, verbose=2, observation_range=(-5.0,5.0)) # param_noise=param_noise, action_noise=action_noise)
# TODO: check what is the desired observation_range (for normalization)
# OR
# Add some param noise for exploration
# param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
# model = DDPG(LnMlpPolicy, env, verbose=2, param_noise=param_noise)

# train
time_steps = 1e5
model.learn(total_timesteps=int(time_steps), callback=callback)
# plot results
results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "DDPG MoveRock")
plt.show()
# save
model.save(log_dir + "ddpg_MoveRock")
# Don't forget to save the VecNormalize statistics when saving the agent
# env.save(os.path.join(log_dir, "vec_normalize.pkl"))
# Evaluate the agent
mean_reward, n_steps = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# display trained model
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()