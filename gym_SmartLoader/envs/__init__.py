from gym.envs.registration import register
# add all custom envs here

register(
    id='PickUpEnv-v0',
    entry_point='envs.SmartLoaderEnvs_dir:PickUpEnv',
)

register(
    id='PutDownEnv-v0',
        entry_point='envs:SmartLoaderEnvs_dir:PutDownEnv',
)

register(
    id='MoveWithStonesEnv-v0',
        entry_point='envs:SmartLoaderEnvs_dir:MoveWithStonesEnv',
)