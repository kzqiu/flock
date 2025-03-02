import pygame
import numpy as np

class Agent:
    def __init__(self, position, radius=5):
        self.position = np.array(position, dtype=float)
        self.last_position = np.array(position, dtype=float)  # for teleport detection
        self.velocity = np.array([0.0, 0.0])
        self.radius = radius
        self.color = (255, 0, 0)  # red
        self.sensor_range = 100.0
        self.mass = 1.0
        self.max_speed = 200.0
        self.max_force = 100.0
        
        # collision properties
        self.restitution = 0.3  # reduced bounciness for stability
        self.separation_buffer = 1.0  # extra buffer to prevent clipping
        
        # reference to the environment (set by environment when agent is added)
        self.environment = None
    
    def apply_force(self, force):
        """Apply a force to the agent"""
        # limit the maximum force
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > self.max_force * 2:
            force = (force / force_magnitude) * (self.max_force * 2)
            
        acceleration = force / self.mass
        self.velocity += acceleration
    
    def sense_environment(self):
        """Gather information about the environment from agent's perspective"""
        if self.environment is None:
            return None
            
        # get direction to transport object
        transport_obj = self.environment.transport_object
        obj_pos = transport_obj.position
        dir_to_object = obj_pos - self.position
        dist_to_object = np.linalg.norm(dir_to_object)
        if dist_to_object > 0:
            dir_to_object = dir_to_object / dist_to_object  # normalize
        
        # get direction to target
        target_pos = self.environment.target_pos
        dir_to_target = target_pos - self.position
        dist_to_target = np.linalg.norm(dir_to_target)
        if dist_to_target > 0:
            dir_to_target = dir_to_target / dist_to_target  # normalize
        
        # get information about nearby agents (mean field)
        nearby_agents = [a for a in self.environment.agents if a is not self and
                         np.linalg.norm(a.position - self.position) < self.sensor_range]
        
        avg_neighbor_velocity = np.array([0.0, 0.0])
        if len(nearby_agents) > 0:
            for agent in nearby_agents:
                avg_neighbor_velocity += agent.velocity
            avg_neighbor_velocity = avg_neighbor_velocity / len(nearby_agents)
            
        #return all sensed information
        return {
            'dir_to_object': dir_to_object,
            'dist_to_object': dist_to_object,
            'dir_to_target': dir_to_target,
            'dist_to_target': dist_to_target,
            'nearby_agents_count': len(nearby_agents),
            'avg_neighbor_velocity': avg_neighbor_velocity
        }
    
    def simple_control_policy(self, sensed_data):
        """A simple rule-based control policy"""
        if sensed_data is None:
            return np.array([0.0, 0.0])
            
        force = np.array([0.0, 0.0])
        
        # if far from object, move towards it
        if sensed_data['dist_to_object'] > self.radius * 2:
            force = sensed_data['dir_to_object'] * self.max_force
        
        # if close to object, push towards target
        else:
            force = sensed_data['dir_to_target'] * self.max_force
        
        # avoid overcrowding - if too many nearby agents, move slightly away
        if sensed_data['nearby_agents_count'] > 3:
            # get a force component in the opposite direction of average neighbor velocity
            avoidance = -sensed_data['avg_neighbor_velocity'] * 0.5  # increased avoidance
            force += avoidance
        
        # add repulsion from nearby agents (stronger)
        if self.environment:
            nearby_agents = [a for a in self.environment.agents 
                           if a is not self and np.linalg.norm(a.position - self.position) < self.radius * 3]
            
            if nearby_agents:
                repulsion = np.array([0.0, 0.0])
                for agent in nearby_agents:
                    direction = self.position - agent.position
                    dist = np.linalg.norm(direction)
                    if dist > 0:
                        # stronger repulsion as they get closer
                        strength = (self.radius * 3 - dist) / (self.radius * 3)
                        repulsion += direction / dist * strength * self.max_force * 0.5
                force += repulsion
            
        # return the resulting force
        return force
    
    def check_collision(self, other_agent):
        """Check if this agent collides with another agent"""
        distance = np.linalg.norm(self.position - other_agent.position)
        return distance < (self.radius + other_agent.radius)
    
    def check_swept_collision(self, other_agent, dt):
        """Check for collision using swept circles for continuous collision detection"""
        # get movement vectors for this frame
        movement1 = self.velocity * dt
        movement2 = other_agent.velocity * dt
        
        # calculate relative movement
        relative_movement = movement1 - movement2
        
        # start positions
        start_pos1 = self.position
        start_pos2 = other_agent.position
        
        # vector from start_pos2 to start_pos1
        start_to_start = start_pos1 - start_pos2
        
        # sum of radii with buffer
        sum_radii = self.radius + other_agent.radius + self.separation_buffer
        
        # calculate quadratic equation components
        a = np.dot(relative_movement, relative_movement)
        b = 2 * np.dot(relative_movement, start_to_start)
        c = np.dot(start_to_start, start_to_start) - sum_radii * sum_radii
        
        # if agents are already overlapping, return True
        if c <= 0:
            return True
        
        # if relative movement is negligible, no collision will occur
        if a < 0.0001:
            return False
        
        # calculate discriminant
        discriminant = b * b - 4 * a * c
        
        # if discriminant is negative, no collision occurs during this time step
        if discriminant < 0:
            return False
        
        # calculate collision time
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        
        # check if collision happens during this time step
        return 0 <= t <= 1
    
    def resolve_collision(self, other_agent):
        """Resolve collision between this agent and another agent"""
        # vector from other agent to this agent
        delta = self.position - other_agent.position
        distance = np.linalg.norm(delta)
        
        # avoid division by zero
        if distance < 0.0001:
            # if agents are exactly at the same position, move one slightly
            self.position += np.array([1.0, 0.0])
            return
            
        # normalize collision vector
        collision_vector = delta / distance
        
        # calculate target distance with buffer to prevent clipping
        target_distance = (self.radius + other_agent.radius) + self.separation_buffer
        
        # calculate overlap
        overlap = target_distance - distance
        
        # separate the agents to resolve overlap
        # weight the movement based on mass ratio (heavier agents move less)
        mass_sum = self.mass + other_agent.mass
        self_ratio = other_agent.mass / mass_sum
        other_ratio = self.mass / mass_sum
        
        # apply position correction with mass weighting
        self.position += collision_vector * overlap * self_ratio
        other_agent.position -= collision_vector * overlap * other_ratio
        
        # calculate relative velocity
        relative_velocity = self.velocity - other_agent.velocity
        
        # calculate relative velocity along the collision vector
        velocity_along_normal = np.dot(relative_velocity, collision_vector)
        
        # if objects are moving away from each other, don't apply impulse
        if velocity_along_normal > 0:
            return
            
        # calculate impulse scalar using coefficient of restitution (bounciness)
        j = -(1 + self.restitution) * velocity_along_normal
        j /= 1/self.mass + 1/other_agent.mass
        
        # limit the maximum impulse to prevent instability
        j = np.clip(j, -80, 80)
        
        # apply impulse
        impulse = j * collision_vector
        self.velocity += impulse / self.mass
        other_agent.velocity -= impulse / other_agent.mass
    
    def update(self, dt):
        """Update the agent's state"""
        # store last position
        self.last_position = self.position.copy()
        
        # sense the environment
        sensed_data = self.sense_environment()
        
        # determine force to apply using control policy
        force = self.simple_control_policy(sensed_data)
        
        # apply the force
        self.apply_force(force)
        
        # cap the velocity magnitude
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
        
        # update position based on velocity
        # use substeps for fast-moving agents
        if speed > 50:
            steps = max(1, int(speed / 50))
            small_dt = dt / steps
            for _ in range(steps):
                self.position += self.velocity * small_dt
        else:
            self.position += self.velocity * dt
        
        # apply friction/damping
        self.velocity *= 0.95
    
    def render(self, screen):
        """Render the agent"""
        # draw the agent as a circle
        pygame.draw.circle(screen, self.color, self.position.astype(int), self.radius)
        
        # draw a small line showing the agent's velocity direction
        if np.linalg.norm(self.velocity) > 0:
            vel_dir = self.velocity / np.linalg.norm(self.velocity)
            end_pos = self.position + vel_dir * self.radius * 1.2
            pygame.draw.line(screen, (0, 0, 0), self.position.astype(int), end_pos.astype(int), 2)