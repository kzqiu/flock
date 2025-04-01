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

        # reference to the environment
        self.environment = None

    def apply_force(self, force):
        """Apply a force to the agent"""
        # convert to numpy array if needed
        if not isinstance(force, np.ndarray):
            force = np.array(force, dtype=float)

        # clip force to prevent extreme values
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > self.max_force:
            force = (force / force_magnitude) * self.max_force

        # F = ma, so a = F/m
        acceleration = force / self.mass
        self.velocity += acceleration

    def update(self, dt):
        """Update the agent's position based on velocity"""
        # store last position
        self.last_position = self.position.copy()

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
                # enforce boundaries after each substep
                self.enforce_boundaries()
        else:
            self.position += self.velocity * dt
            # enforce boundaries
            self.enforce_boundaries()

        # apply friction/damping
        self.velocity *= 0.98

    def enforce_boundaries(self):
        """Keep the agent within the environment boundaries"""
        if self.environment:
            width = self.environment.width
            height = self.environment.height

            # X boundaries
            if self.position[0] - self.radius < 0:
                self.position[0] = self.radius
                self.velocity[0] = 0  # stop movement into boundary
            elif self.position[0] + self.radius > width:
                self.position[0] = width - self.radius
                self.velocity[0] = 0  # stop movement into boundary

            # Y boundaries
            if self.position[1] - self.radius < 0:
                self.position[1] = self.radius
                self.velocity[1] = 0  # stop movement into boundary
            elif self.position[1] + self.radius > height:
                self.position[1] = height - self.radius
                self.velocity[1] = 0  # stop movement into boundary

    def check_collision(self, other_agent):
        """Check if this agent collides with another agent"""
        distance = np.linalg.norm(self.position - other_agent.position)
        return distance < (self.radius + other_agent.radius)

    def resolve_collision(self, other_agent):
        """Resolve collision between this agent and another agent"""
        # vector from other agent to this agent
        delta = self.position - other_agent.position
        distance = np.linalg.norm(delta)

        # avoid division by zero
        if distance < 0.0001:
            # if agents are exactly at the same position, move one slightly
            self.position += np.array([0.1, 0.1])
            return

        # normalize collision vector
        collision_vector = delta / distance

        # calculate overlap
        overlap = (self.radius + other_agent.radius) - distance + self.separation_buffer

        # separate the agents to resolve overlap based on mass
        mass_sum = self.mass + other_agent.mass
        self_ratio = other_agent.mass / mass_sum
        other_ratio = self.mass / mass_sum

        self.position += collision_vector * overlap * self_ratio
        other_agent.position -= collision_vector * overlap * other_ratio

        # calculate relative velocity along collision normal
        rel_velocity = self.velocity - other_agent.velocity
        vel_along_normal = np.dot(rel_velocity, collision_vector)

        # if objects are moving away from each other, don't apply impulse
        if vel_along_normal > 0:
            return

        # calculate impulse scalar
        j = -(1 + self.restitution) * vel_along_normal
        j /= 1 / self.mass + 1 / other_agent.mass

        # apply impulse
        impulse = j * collision_vector
        self.velocity += impulse / self.mass
        other_agent.velocity -= impulse / other_agent.mass

    def render(self, screen):
        """Render the agent on the screen"""
        pygame.draw.circle(screen, self.color, self.position.astype(int), self.radius)

        # draw direction indicator
        if np.linalg.norm(self.velocity) > 0:
            vel_normalized = self.velocity / np.linalg.norm(self.velocity)
            line_end = self.position + vel_normalized * (self.radius + 3)
            pygame.draw.line(
                screen, (0, 0, 0), self.position.astype(int), line_end.astype(int), 2
            )
