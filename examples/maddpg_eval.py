import os

import numpy as np
import torch

from flock import FlockEnv
from flock.maddpg import SHMADDPG


if __name__ == "__main__":
    n_agents = 2
    width, height = (300, 300)
    
    env = FlockEnv(n_agents, width, height, num_obstacles=0)

    agent = SHMADDPG(
        n_agents=n_agents,
        local_obs_dim=11,
        global_obs_dim=6,
        agent_act_dim=2,
        gamma=0.99,
        tau=0.005,
        buffer_size=int(1e6),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # only load actor
    agent.actor.load_state_dict(torch.load(os.path.join("./models", "shmaddpg_actor.pth"), weights_only=True))
    
    obs = env.reset()

    info = {}
    total_reward = 0
    done = False

    while not done:
        actions = agent.select_actions(obs[:-agent.global_obs_dim], obs[-agent.global_obs_dim:], explore=False)
        obs, reward, done, info = env.step(actions)
        total_reward += reward

        env.render()

        if hasattr(env, "user_exit") and env.user_exit:
            print("User requested exit. Stopping simulation.")
            break

        print(
            f"Step: {env.current_step}, Reward: {reward:.2f}, "
            f"Distance: {info['distance_to_target']:.2f}"
        )

    print(f"Episode done. Total reward: {total_reward:.2f}")
    print(f"Success: {info['success']}")

    env.close()
        
