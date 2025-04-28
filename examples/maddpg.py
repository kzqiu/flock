import torch

from flock import FlockEnv
from flock.maddpg import SHMADDPG, train_SHMADDPG


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

    episodes = 250
    max_ep_len = 750

    print("Starting training...")
    train_SHMADDPG(
        env,
        agent,
        episodes,
        max_ep_len,
        batch_size=1024,
        noise_stddev_start=0.6,
        noise_stddev_end=0.05,
        noise_decay_steps=episodes * max_ep_len,
        update_freq=100,
    )
    print("Training finished...")

    # save model weights!
    agent.save_models("./models")

    # sample simulation
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
