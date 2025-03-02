import pygame
import numpy as np
from .transport_object import TransportObject

class Environment:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Swarm Robotics Simulation")
        
        # set up clock for controlling frame rate
        self.clock = pygame.time.Clock()
        self.FPS = 60
        
        # colors
        self.WHITE = (255, 255, 255)
        self.GRAY = (200, 200, 200)
        self.GREEN = (0, 200, 0)
        
        # environment properties
        self.running = True
        self.paused = False
        self.elapsed_time = 0
        self.success = False
        self.debug_mode = True  # default debug mode on to help diagnose issues
        self.max_dt = 0.016  # maximum physics timestep (about 60 fps)
        
        # set up font for displaying information
        self.font = pygame.font.SysFont(None, 24)
        
        # create target location
        self.target_pos = np.array([width * 0.75, height * 0.75])
        self.target_radius = 30
        
        # create transport object
        self.transport_object = TransportObject(
            position=np.array([width/2, height/2]),
            width=60,
            height=60
        )
        
        # agents will be added later from main.py
        self.agents = []
        
        # for tracking object paths and debugging
        self.object_positions = []
        self.max_path_length = 30  # Keep last 30 positions
        
        # metrics
        self.frame_times = []
        self.avg_fps = 0
    
    def add_agent(self, agent):
        """Add an agent to the environment"""
        self.agents.append(agent)
        # let agent know about its environment
        agent.environment = self
    
    def handle_events(self):
        """Handle user input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.reset()
                elif event.key == pygame.K_d:
                    # Toggle debug mode
                    self.debug_mode = not self.debug_mode
    
    def reset(self):
        """Reset the simulation to its initial state"""
        # reset transport object
        self.transport_object.position = np.array([self.width/2, self.height/2])
        self.transport_object.velocity = np.array([0.0, 0.0])
        
        # reset agents in a circle around the transport object
        num_agents = len(self.agents)
        for i, agent in enumerate(self.agents):
            angle = i * (2 * np.pi / num_agents)
            x = self.width/2 + np.cos(angle) * 100
            y = self.height/2 + np.sin(angle) * 100
            agent.position = np.array([x, y])
            agent.last_position = agent.position.copy()
            agent.velocity = np.array([0.0, 0.0])
        
        # reset simulation state
        self.paused = False
        self.success = False
        self.elapsed_time = 0
        self.object_positions = []
    
    def handle_agent_collisions(self, dt):
        """Check and resolve collisions between agents using standard and swept detection"""
        # first handle standard collisions (agents already overlapping)
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                agent_a = self.agents[i]
                agent_b = self.agents[j]
                
                # check if agents currently collide
                if agent_a.check_collision(agent_b):
                    # resolve the collision
                    agent_a.resolve_collision(agent_b)
        
        # check for swept collisions (continuous detection for fast moving agents)
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                agent_a = self.agents[i]
                agent_b = self.agents[j]
                
                # skip if agents are already in contact (already handled)
                if agent_a.check_collision(agent_b):
                    continue
                
                # only check swept collision if either agent is moving quickly
                speed_a = np.linalg.norm(agent_a.velocity)
                speed_b = np.linalg.norm(agent_b.velocity)
                
                if speed_a > 50 or speed_b > 50:
                    # check for swept collision
                    if agent_a.check_swept_collision(agent_b, dt):
                        # if swept collision detected, resolve it
                        agent_a.resolve_collision(agent_b)
    
    def handle_transport_object_collisions(self, dt):
        """Check and resolve collisions between agents and transport object using swept collision detection"""
        for agent in self.agents:
            # first check for standard collision
            if self.transport_object.check_collision_with_agent(agent):
                self.transport_object.resolve_collision_with_agent(agent)
            # then check for swept collision (continuous detection) if agent is moving fast
            elif np.linalg.norm(agent.velocity) > 50:  # only for fast-moving agents
                if self.transport_object.check_swept_collision_with_agent(agent, dt):
                    self.transport_object.resolve_collision_with_agent(agent)
    
    def update(self, dt):
        """Update the simulation state"""
        if self.paused:
            return
        
        # cap dt to prevent physics instability
        dt = min(dt, self.max_dt)
        
        # update elapsed time
        self.elapsed_time += dt
        
        # store transport object position for path visualization
        if len(self.object_positions) > self.max_path_length:
            self.object_positions.pop(0)
        self.object_positions.append(self.transport_object.position.copy())
        
        # update all agents
        for agent in self.agents:
            agent.update(dt)
        
        # handle collisions between agents with improved detection
        self.handle_agent_collisions(dt)
        
        # handle collisions between agents and transport object
        self.handle_transport_object_collisions(dt)
        
        # apply forces from agents to transport object (pushing behavior)
        self.handle_agent_object_interactions()
        
        # update transport object with explicit boundary dimensions
        self.transport_object.update(dt, self.width, self.height)
        
        # check for success condition
        self.check_success()
    
    def handle_agent_object_interactions(self):
        """Handle interactions between agents and the transport object (pushing behavior)"""
        for agent in self.agents:
            # calculate vector from agent to object
            agent_to_object = self.transport_object.position - agent.position
            distance = np.linalg.norm(agent_to_object)
            
            # if agent is close to the object edge
            contact_threshold = agent.radius + self.transport_object.width / 2
            if abs(distance - contact_threshold) < 2.0:
                # normalize direction vector
                if distance > 0:  # avoid division by zero
                    direction = agent_to_object / distance
                else:
                    direction = np.array([1.0, 0.0])
                
                # only apply pushing force if agent is moving toward object
                vel_toward_obj = np.dot(agent.velocity, direction)
                if vel_toward_obj > 0:
                    # scale force by velocity component toward object
                    push_force = direction * vel_toward_obj * 0.6
                    self.transport_object.apply_force(push_force)
    
    def check_success(self):
        """Check if the transport object has reached the target"""
        distance = np.linalg.norm(self.transport_object.position - self.target_pos)
        if distance < self.target_radius + self.transport_object.width/2:
            self.success = True
            self.paused = True
    
    def render(self):
        """Render the environment and all objects in it"""
        # fill background
        self.screen.fill(self.WHITE)
        
        # draw a grid pattern for visual reference
        self.draw_grid()
        
        # draw target location
        pygame.draw.circle(self.screen, self.GREEN, self.target_pos.astype(int), self.target_radius)
        pygame.draw.circle(self.screen, (0, 100, 0), self.target_pos.astype(int), self.target_radius - 5)
        
        # draw transport object path if in debug mode
        if self.debug_mode and len(self.object_positions) > 1:
            for i in range(len(self.object_positions) - 1):
                pygame.draw.line(self.screen, (200, 100, 100), 
                                self.object_positions[i].astype(int), 
                                self.object_positions[i+1].astype(int), 1)
        
        # draw transport object
        self.transport_object.render(self.screen)
        
        # draw all agents
        for agent in self.agents:
            agent.render(self.screen)
        
        # display simulation information
        self.render_info()
        
        # update the display
        pygame.display.flip()
    
    def draw_grid(self):
        """Draw a grid pattern on the background"""
        grid_spacing = 50
        for x in range(0, self.width, grid_spacing):
            pygame.draw.line(self.screen, self.GRAY, (x, 0), (x, self.height))
        for y in range(0, self.height, grid_spacing):
            pygame.draw.line(self.screen, self.GRAY, (0, y), (self.width, y))
    
    def render_info(self):
        """Display simulation information as text"""
        # show elapsed time
        time_text = f"Time: {self.elapsed_time:.1f}s"
        time_surface = self.font.render(time_text, True, (0, 0, 0))
        self.screen.blit(time_surface, (10, 10))
        
        # show number of agents
        agents_text = f"Agents: {len(self.agents)}"
        agents_surface = self.font.render(agents_text, True, (0, 0, 0))
        self.screen.blit(agents_surface, (10, 30))
        
        # show status (running/paused)
        status = "PAUSED" if self.paused else "RUNNING"
        status_surface = self.font.render(status, True, (0, 0, 0))
        self.screen.blit(status_surface, (10, 50))
        
        # show current date/time as specified
        date_text = "2025-03-02 19:43:02"
        date_surface = self.font.render(f"UTC: {date_text}", True, (0, 0, 0))
        self.screen.blit(date_surface, (10, 70))
        
        # show object info if in debug mode
        if self.debug_mode:
            # position and velocity
            pos = self.transport_object.position
            vel = self.transport_object.velocity
            pos_text = f"Box: pos=({pos[0]:.1f}, {pos[1]:.1f})"
            vel_text = f"vel=({vel[0]:.1f}, {vel[1]:.1f})"
            
            pos_surface = self.font.render(pos_text, True, (200, 0, 0))
            vel_surface = self.font.render(vel_text, True, (200, 0, 0))
            
            self.screen.blit(pos_surface, (10, 90))
            self.screen.blit(vel_surface, (10, 110))
            
            # fps info
            fps_text = f"FPS: {self.avg_fps:.1f}"
            fps_surface = self.font.render(fps_text, True, (0, 0, 0))
            self.screen.blit(fps_surface, (10, 130))
        
        # show success message if target reached
        if self.success:
            success_text = "TARGET REACHED!"
            success_surface = self.font.render(success_text, True, (0, 200, 0))
            text_rect = success_surface.get_rect(center=(self.width/2, 50))
            self.screen.blit(success_surface, text_rect)
        
        # display controls info at bottom
        controls = "Controls: Space=Pause, R=Reset, D=Debug, ESC=Quit"
        controls_surface = self.font.render(controls, True, (0, 0, 0))
        self.screen.blit(controls_surface, (10, self.height - 30))
    
    def run(self):
        """Main simulation loop"""
        # for fps calculation
        frame_count = 0
        fps_update_time = 0
        
        while self.running:
            # handle events
            self.handle_events()
            
            # calculate delta time
            dt = self.clock.tick(self.FPS) / 1000.0
            
            # update FPS calculation
            frame_count += 1
            fps_update_time += dt
            if fps_update_time >= 1.0:  # update FPS every second
                self.avg_fps = frame_count / fps_update_time
                frame_count = 0
                fps_update_time = 0
            
            # update simulation
            self.update(dt)
            
            # Render
            self.render()
        
        # clean up pygame
        pygame.quit()