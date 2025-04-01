import os
import base64
from IPython import display
import numpy as np
import pygame
import gym
from gym import spaces
from .transport_object import TransportObject
from .agent import Agent
from .obstacle import create_random_obstacle


class FlockEnv(gym.Env):
    """
    Flock environment following the OpenAI Gym interface.

    Agents must cooperatively transport an object to a target location.
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, num_agents=50, width=800, height=600, num_obstacles=5):
        super(FlockEnv, self).__init__()

        # environment dimensions
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.num_obstacles = num_obstacles

        # pygame setup (for rendering)
        self.screen = None
        self.clock = None
        self.FPS = 60
        self.isopen = False

        # flag for user-initiated exit
        self.user_exit = False

        # define action and observation space
        # actions: force vector for each agent [fx, fy]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_agents, 2), dtype=np.float32
        )

        # observations: state of environment
        # for each agent: [pos_x, pos_y, vel_x, vel_y,
        #                 object_rel_x, object_rel_y, target_rel_x, target_rel_y,
        #                 nearest_obstacle_dist, nearest_obstacle_dir_x, nearest_obstacle_dir_y]
        # plus global information: [object_pos_x, object_pos_y, object_vel_x, object_vel_y,
        #                          target_pos_x, target_pos_y]
        agent_obs_dim = 11  # 11 values per agent
        global_obs_dim = 6  # 6 global values

        self.observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(num_agents * agent_obs_dim + global_obs_dim,),
            dtype=np.float32,
        )

        # initialize environment components
        self.agents = []
        self.obstacles = []
        self.transport_object = None
        self.target_pos = None
        self.target_radius = 30

        # metrics
        self.elapsed_time = 0
        self.success = False
        self.previous_distance = None  # for reward calculation

        # episode settings
        self.max_episode_steps = 500
        self.current_step = 0

    def seed(self, seed=None):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        return [seed]

    def reset(self):
        """
        Reset the environment to an initial state.

        Returns:
            observation: The initial observation.
        """
        self.elapsed_time = 0
        self.current_step = 0
        self.success = False

        # create target location
        self.target_pos = np.array([self.width * 0.75, self.height * 0.75])

        # create transport object
        self.transport_object = TransportObject(
            position=np.array([self.width / 2, self.height / 2]), width=60, height=60
        )

        # reset previous distance for reward calculation
        self.previous_distance = np.linalg.norm(
            self.transport_object.position - self.target_pos
        )

        # create agents in a circle around the transport object
        self.agents = []
        for i in range(self.num_agents):
            angle = i * (2 * np.pi / self.num_agents)
            # position around the transport object
            x = self.width / 2 + np.cos(angle) * 100
            y = self.height / 2 + np.sin(angle) * 100

            agent = Agent(position=(x, y))
            agent.environment = self  # give agent reference to environment
            self.agents.append(agent)

        # create obstacles
        self.obstacles = []
        self.create_random_obstacles(self.num_obstacles)

        # get initial observation
        observation = self._get_observation()

        return observation

    def step(self, actions):
        """
        Take a step in the environment using the given actions.

        Args:
            actions: Array of shape (num_agents, 2) containing force vectors for each agent.

        Returns:
            observation: The new observation.
            reward: The reward signal.
            done: Whether the episode has ended.
            info: Additional information.
        """
        self.current_step += 1
        dt = 1.0 / self.FPS  # fixed timestep for stability

        # scale actions from [-1, 1] range to actual force range
        max_force = 100.0
        scaled_actions = actions * max_force

        # apply actions (forces) to agents
        for i, agent in enumerate(self.agents):
            if i < len(scaled_actions):  # safety check
                force = scaled_actions[i]
                agent.apply_force(force)

        for agent in self.agents:
            agent.update(dt)

        self._handle_collisions(dt)

        self.transport_object.update(dt, self.width, self.height)

        self.elapsed_time += dt

        # check success condition
        distance_to_target = np.linalg.norm(
            self.transport_object.position - self.target_pos
        )
        self.success = distance_to_target < (
            self.target_radius + self.transport_object.width / 2
        )

        observation = self._get_observation()

        reward = self._compute_reward(distance_to_target)

        # update previous distance for next reward calculation
        self.previous_distance = distance_to_target

        # check if episode is done
        done = self.success or self.current_step >= self.max_episode_steps

        info = {
            "distance_to_target": distance_to_target,
            "success": self.success,
            "elapsed_time": self.elapsed_time,
        }

        return observation, reward, done, info

    def render(self, mode="human"):
        """
        Render the environment.

        Args:
            mode: 'human' for window display, 'rgb_array' for array return.
        """
        try:
            # initialize pygame if not already initialized
            if not pygame.get_init():
                pygame.init()

            if self.screen is None and mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Flock Environment")
                self.clock = pygame.time.Clock()
                self.isopen = True

            # process any pending events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None
                # check for key presses to exit simulation
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        print("Exiting simulation by keyboard command...")
                        self.user_exit = True
                        self.close()
                        return None

            if self.screen is not None:
                # fill background with white
                self.screen.fill((255, 255, 255))

                self._draw_grid()

                # draw target
                pygame.draw.circle(
                    self.screen,
                    (0, 200, 0),
                    self.target_pos.astype(int),
                    self.target_radius,
                )
                pygame.draw.circle(
                    self.screen,
                    (0, 100, 0),
                    self.target_pos.astype(int),
                    self.target_radius - 5,
                )

                for obstacle in self.obstacles:
                    obstacle.render(self.screen)

                self.transport_object.render(self.screen)

                for agent in self.agents:
                    agent.render(self.screen)

                self._display_info()

                if mode == "human":
                    pygame.display.flip()
                    if self.clock:
                        self.clock.tick(self.FPS)
                    return None
                elif mode == "rgb_array":
                    # convert screen to array
                    return np.transpose(
                        np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
                    )

        except Exception as e:
            print(f"Error in render method: {e}")
            import traceback

            traceback.print_exc()
            return None

    def close(self):
        """Close the environment"""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def _get_observation(self):
        """
        Construct observation from current environment state.

        Returns:
            observation: Flattened observation array.
        """
        observations = []

        # collect per-agent observations
        for agent in self.agents:
            # basic agent info (position and velocity)
            agent_obs = [
                agent.position[0] / self.width,  # normalize positions
                agent.position[1] / self.height,
                agent.velocity[0] / agent.max_speed,  # normalize velocities
                agent.velocity[1] / agent.max_speed,
            ]

            # relative vectors to object and target
            obj_rel = (self.transport_object.position - agent.position) / self.width
            target_rel = (self.target_pos - agent.position) / self.width

            agent_obs.extend([obj_rel[0], obj_rel[1], target_rel[0], target_rel[1]])

            # nearest obstacle information
            nearest_obstacle = None
            nearest_dist = float("inf")

            for obstacle in self.obstacles:
                # approximate distance check
                dist = np.linalg.norm(obstacle.position - agent.position)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_obstacle = obstacle

            if nearest_obstacle:
                # normalize distance by sensor range
                normalized_dist = min(1.0, nearest_dist / agent.sensor_range)
                dir_from_obstacle = agent.position - nearest_obstacle.position
                if np.linalg.norm(dir_from_obstacle) > 0:
                    dir_from_obstacle = dir_from_obstacle / np.linalg.norm(
                        dir_from_obstacle
                    )

                agent_obs.extend(
                    [normalized_dist, dir_from_obstacle[0], dir_from_obstacle[1]]
                )
            else:
                agent_obs.extend([1.0, 0.0, 0.0])  # no obstacle detected

            observations.extend(agent_obs)

        # add global information
        global_obs = [
            self.transport_object.position[0] / self.width,
            self.transport_object.position[1] / self.height,
            self.transport_object.velocity[0] / self.transport_object.max_speed,
            self.transport_object.velocity[1] / self.transport_object.max_speed,
            self.target_pos[0] / self.width,
            self.target_pos[1] / self.height,
        ]

        observations.extend(global_obs)

        return np.array(observations, dtype=np.float32)

    def _compute_reward(self, current_distance):
        """
        Compute the reward based on the current state.

        Args:
            current_distance: Current distance from object to target.

        Returns:
            reward: The computed reward value.
        """
        # reward for moving closer to the target
        distance_improvement = self.previous_distance - current_distance
        distance_reward = distance_improvement * 10.0  # scale to make it meaningful

        # success reward
        success_reward = 100.0 if self.success else 0.0

        # small time penalty to encourage efficiency
        time_penalty = -0.01

        # combine rewards
        total_reward = distance_reward + success_reward + time_penalty

        return total_reward

    def _handle_collisions(self, dt):
        """Handle all collisions in the environment"""
        # agent-agent collisions
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                if self.agents[i].check_collision(self.agents[j]):
                    self.agents[i].resolve_collision(self.agents[j])

        # agent-obstacle collisions
        for agent in self.agents:
            for obstacle in self.obstacles:
                if obstacle.check_collision_with_agent(agent):
                    obstacle.resolve_collision_with_agent(agent)

        # agent-object collisions
        for agent in self.agents:
            if self.transport_object.check_collision_with_agent(agent):
                self.transport_object.resolve_collision_with_agent(agent)

    def create_random_obstacles(self, count):
        """Create random obstacles in the environment"""
        self.obstacles = []

        # define regions to avoid when placing obstacles
        source_region = {
            "x": self.width / 2,
            "y": self.height / 2,
            "radius": 120,  # keep clear area around starting point
        }

        target_region = {
            "x": self.target_pos[0],
            "y": self.target_pos[1],
            "radius": 50,  # keep clear area around target
        }

        # try to create the specified number of obstacles
        attempts = 0
        max_attempts = 200

        while len(self.obstacles) < count and attempts < max_attempts:
            attempts += 1

            # create a random obstacle
            obstacle = create_random_obstacle(0, self.width, 0, self.height)

            # check if it overlaps with source region
            dist_to_source = np.linalg.norm(
                np.array(
                    [
                        obstacle.position[0] - source_region["x"],
                        obstacle.position[1] - source_region["y"],
                    ]
                )
            )
            if dist_to_source < source_region["radius"]:
                continue  # skip this obstacle

            # check if it overlaps with target region
            dist_to_target = np.linalg.norm(
                np.array(
                    [
                        obstacle.position[0] - target_region["x"],
                        obstacle.position[1] - target_region["y"],
                    ]
                )
            )
            if dist_to_target < target_region["radius"]:
                continue  # skip this obstacle

            # check if it overlaps with other obstacles
            overlaps = False
            for existing in self.obstacles:
                # simple distance check which can prob be improved for different shapes
                dist = np.linalg.norm(obstacle.position - existing.position)
                if dist < 60:  # arbitrary minimum separation
                    overlaps = True
                    break

            if not overlaps:
                self.obstacles.append(obstacle)

    def _draw_grid(self):
        """Draw a reference grid on the background"""
        if not self.screen:
            return

        grid_spacing = 50
        color = (200, 200, 200)  # light gray

        for x in range(0, self.width, grid_spacing):
            pygame.draw.line(self.screen, color, (x, 0), (x, self.height))
        for y in range(0, self.height, grid_spacing):
            pygame.draw.line(self.screen, color, (0, y), (self.width, y))

    def _display_info(self):
        """Display environment information on screen"""
        if not self.screen:
            return

        font = pygame.font.SysFont(None, 24)

        # basic simulation info
        info_text = [
            f"Time: {self.elapsed_time:.1f}s",
            f"Agents: {self.num_agents}",
            f"Step: {self.current_step}/{self.max_episode_steps}",
            f"Status: {'SUCCESS' if self.success else 'Running'}",
        ]

        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 20))

    def _init_pygame_headless(self):
        """Initialize Pygame in headless mode for Google Colab"""
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.isopen = True

    def render_for_colab(self, mode="rgb_array"):
        """Render and display an image in Google Colab"""
        if not pygame.get_init():
            self._init_pygame_headless()

        img = self.render(mode="rgb_array")
        if img is not None:
            from PIL import Image

            img = Image.fromarray(img)

            img_path = "/tmp/flock_render.png"
            img.save(img_path)

            with open(img_path, "rb") as f:
                image_data = f.read()
            image = base64.b64encode(image_data).decode("utf-8")
            display.display(display.HTML(f'<img src="data:image/png;base64,{image}"/>'))

            return img
        return None
