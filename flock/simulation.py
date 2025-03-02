import numpy as np
from environment.environment import Environment
from environment.agent import Agent

def main():
    # create environment
    env = Environment(width=800, height=600)
    
    # create agents in a circle around the transport object
    num_agents = 30
    for i in range(num_agents):
        angle = i * (2 * np.pi / num_agents)
        # position around the transport object
        x = env.width/2 + np.cos(angle) * 100
        y = env.height/2 + np.sin(angle) * 100
        agent = Agent(position=(x, y))
        env.add_agent(agent)
    
    # run the simulation
    env.run()

if __name__ == "__main__":
    main()