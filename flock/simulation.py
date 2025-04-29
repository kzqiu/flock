import numpy as np
from environment.flock_env import FlockEnv
from deterministic_controller import DeterministicController

def main():
    env = FlockEnv(num_agents=25, num_obstacles=15)
    controller = DeterministicController(num_agents=env.num_agents)
    
    # reset the environment to get initial observation
    observation = env.reset()

    done = False
    total_reward = 0

    while not done:
        # get actions from controller
        actions = controller.get_actions(observation, env)
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