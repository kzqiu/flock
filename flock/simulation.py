import numpy as np
from environment.flock_env import FlockEnv


def main():
    env = FlockEnv(num_agents=50, num_obstacles=15)

    # reset the environment to get initial observation
    observation = env.reset()

    done = False
    total_reward = 0

    while not done:
        # sample random actions for all agents
        actions = np.random.uniform(-1, 1, size=(env.num_agents, 2))

        observation, reward, done, info = env.step(actions)

        total_reward += reward

        env.render()

        # check if user requested exit
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


if __name__ == "__main__":
    main()
