import torch

from flock import FlockEnv
from flock.maddpg import SHMADDPG, train_SHMADDPG


if __name__ == "__main__":
    n_agents = 3
    env = FlockEnv(n_agents, width=400, height=300, num_obstacles=0)

    agent = SHMADDPG(
        n_agents=n_agents,
        local_obs_dim=11,
        global_obs_dim=6,
        agent_act_dim=2,
        gamma=0.95,
        tau=0.01,
        buffer_size=int(1e6),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Starting training...")
    train_SHMADDPG(
        env,
        agent,
        num_episodes=100,
        max_ep_len=500,
        batch_size=512,
        noise_stddev_start=0.3,
        noise_stddev_end=0.05,
        noise_decay_steps=50000,
        update_freq=100,
    )
    print("Training finished...")

    # TODO: save model weights!
    agent.save_models()

    # Example of selecting actions without exploration after training
    obs_test = env.reset()
    actions_test = agent.select_actions(obs_test[:-agent.global_obs_dim], obs_test[-agent.global_obs_dim:], explore=False)
    print("\nExample actions selected without exploration:")
    print(actions_test)
