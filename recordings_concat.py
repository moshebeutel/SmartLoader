import numpy as np

act1 = np.load('/home/graphics/git/SmartLoader/saved_ep/40_1/actions.npy')
obs1 = np.load('/home/graphics/git/SmartLoader/saved_ep/40_1/obs.npy')
rew1 = np.load('/home/graphics/git/SmartLoader/saved_ep/40_1/rewards.npy')
ep_ret1 = np.load('/home/graphics/git/SmartLoader/saved_ep/40_1/episode_returns.npy')
ep_str1 = np.load('/home/graphics/git/SmartLoader/saved_ep/40_1/episode_starts.npy')

act2 = np.load('/home/graphics/git/SmartLoader/saved_ep/40_2/actions.npy')
obs2 = np.load('/home/graphics/git/SmartLoader/saved_ep/40_2/obs.npy')
rew2 = np.load('/home/graphics/git/SmartLoader/saved_ep/40_2/rewards.npy')
ep_ret2 = np.load('/home/graphics/git/SmartLoader/saved_ep/40_2/episode_returns.npy')
ep_str2 = np.load('/home/graphics/git/SmartLoader/saved_ep/40_2/episode_starts.npy')

act = np.concatenate((act1, act2))
obs = np.concatenate((obs1, obs2))
rew = np.concatenate((rew1,rew2))
ep_ret = np.concatenate((ep_ret1,ep_ret2))
ep_str = np.concatenate((ep_str1,ep_str2))

np.save('/home/graphics/git/SmartLoader/saved_ep/act', act)
np.save('/home/graphics/git/SmartLoader/saved_ep/obs', obs)
np.save('/home/graphics/git/SmartLoader/saved_ep/rew', rew)
np.save('/home/graphics/git/SmartLoader/saved_ep/ep_ret', ep_ret)
np.save('/home/graphics/git/SmartLoader/saved_ep/ep_str', ep_str)