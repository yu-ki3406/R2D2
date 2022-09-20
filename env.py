from gfootball.env.wrappers import SMMWrapper
import gfootball.env as football_env

def env_make():
    env = football_env.create_environment(
        env_name='11_vs_11_easy_stochastic',
        representation='raw',
        stacked=False,
        logdir=None,
        rewards='scoring,checkpoints',
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
        dump_frequency=0
    )
    env = SMMWrapper(env)

    return env
