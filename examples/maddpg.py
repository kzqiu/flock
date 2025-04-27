import torch

from flock import FlockEnv
from flock.maddpg import SHMADDPG, train_SHMADDPG


if __name__ == "__main__":
    n_agents = 5
    env = FlockEnv(n_agents, width=400, height=300, num_obstacles=0)

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

    print("Starting training...")
    train_SHMADDPG(
        env,
        agent,
        num_episodes=1,
        max_ep_len=500,
        batch_size=2,
        noise_stddev_start=0.3,
        noise_stddev_end=0.05,
        noise_decay_steps=50000,
    )
    print("Training finished...")

    # TODO: save model weights!

    # Example of selecting actions without exploration after training
    # obs_test = env.reset()
    # actions_test = agent.select_actions(obs_test, explore=False)
    # print("\nExample actions selected without exploration:")
    # print(actions_test)
