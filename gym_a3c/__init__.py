from gym.envs.registration import register

###### Tic Tac Toe environment
register(
    id='Connect4-A3C-v0',
    entry_point='gym_a3c.Connect4_env:Connect4Env',
    max_episode_steps=100
)
