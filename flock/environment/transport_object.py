import pygame
import numpy as np


class TransportObject:
    def __init__(self, position, width=50, height=50):
        self.position = np.array(position, dtype=float)
        self.last_position = self.position.copy()  # track previous position
        self.velocity = np.array([0.0, 0.0])
        self.width = width
        self.height = height
        self.color = (0, 0, 255)  # blue
        self.mass = 250.0  # heavier than agents
        self.restitution = 0.3  # reduced bounciness for stability
        self.max_speed = 100.0  # reduced maximum speed to prevent glitching
        self.teleport_threshold = 50.0  # used to detect teleporting

        # for continuous collision detection
        self.swept_collision_buffer = 2.0  # extra buffer for swept collision detection

    def apply_force(self, force, source=None):
        """
        Apply a force to the transport object

        Args:
            force: The force vector
            source: Optional source of the force (e.g. an agent)
        """
        # limit the maximum force that can be applied at once
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > 200:
            force = (force / force_magnitude) * 200

        # F = ma, so a = F/m
        acceleration = force / self.mass
        self.velocity += acceleration

        # cap velocity to prevent extreme speeds
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed

    def apply_directed_force(self, force, source):
        """
        Apply a directed force from a specific source

        Args:
            force: The force vector
            source: The agent applying the force
        """
        # for transport object, this works like a regular force
        # but could be extended for special behaviors
        self.apply_force(force)

    def update(self, dt, width, height):
        """Update the object's position based on its velocity"""
        # store last position for teleport detection
        self.last_position = self.position.copy()

        # cap the dt to prevent large jumps
        capped_dt = min(dt, 0.02)

        # use multiple smaller steps for high velocities
        num_steps = max(1, int(np.linalg.norm(self.velocity) / 50))
        small_dt = capped_dt / num_steps

        for _ in range(num_steps):
            # update position with small step
            new_position = self.position + self.velocity * small_dt

            # check if the new position would be valid
            half_width = self.width / 2
            half_height = self.height / 2

            # clamp X position
            if new_position[0] - half_width < 0:
                new_position[0] = half_width
                self.velocity[0] = 0  # stop horizontal movement
            elif new_position[0] + half_width > width:
                new_position[0] = width - half_width
                self.velocity[0] = 0  # stop horizontal movement

            # clamp Y position
            if new_position[1] - half_height < 0:
                new_position[1] = half_height
                self.velocity[1] = 0  # stop vertical movement
            elif new_position[1] + half_height > height:
                new_position[1] = height - half_height
                self.velocity[1] = 0  # stop vertical movement

            # apply the valid position
            self.position = new_position

        # detect teleporting (sudden large position change)
        movement = np.linalg.norm(self.position - self.last_position)
        if movement > self.teleport_threshold:
            # if teleporting detected, revert to last position and stop
            print(f"teleport detected! movement: {movement:.2f}")
            self.position = self.last_position
            self.velocity = np.array([0.0, 0.0])

        # apply drag/friction
        self.velocity *= 0.95

    def check_if_point_inside(self, point):
        """Check if a point is inside the transport object"""
        half_width = self.width / 2
        half_height = self.height / 2

        return (
            point[0] >= self.position[0] - half_width
            and point[0] <= self.position[0] + half_width
            and point[1] >= self.position[1] - half_height
            and point[1] <= self.position[1] + half_height
        )

    def check_collision_with_agent(self, agent):
        """Check if this transport object collides with an agent"""
        # check if the agent's center is inside the box (complete penetration)
        if self.check_if_point_inside(agent.position):
            return True

        # calculate the closest point on the rectangle to the circle
        closest_x = max(
            self.position[0] - self.width / 2,
            min(agent.position[0], self.position[0] + self.width / 2),
        )
        closest_y = max(
            self.position[1] - self.height / 2,
            min(agent.position[1], self.position[1] + self.height / 2),
        )

        # calculate the distance between the closest point and the circle center
        distance = np.linalg.norm(np.array([closest_x, closest_y]) - agent.position)

        # if the distance is less than the circle's radius, collision occurs
        return distance < agent.radius

    def check_swept_collision_with_agent(self, agent, dt):
        """Check for collision using swept circles/rectangles for continuous detection"""
        # get agent's movement vector for this frame
        agent_movement = agent.velocity * dt

        # calculate expanded collision box to account for agent's movement
        expanded_width = self.width + 2 * agent.radius + self.swept_collision_buffer
        expanded_height = self.height + 2 * agent.radius + self.swept_collision_buffer

        # check if agent's movement path intersects with expanded box
        half_width = expanded_width / 2
        half_height = expanded_height / 2

        # check if the agent's current position or future position is within the expanded box
        current_in_box = (
            agent.position[0] >= self.position[0] - half_width
            and agent.position[0] <= self.position[0] + half_width
            and agent.position[1] >= self.position[1] - half_height
            and agent.position[1] <= self.position[1] + half_height
        )

        future_position = agent.position + agent_movement
        future_in_box = (
            future_position[0] >= self.position[0] - half_width
            and future_position[0] <= self.position[0] + half_width
            and future_position[1] >= self.position[1] - half_height
            and future_position[1] <= self.position[1] + half_height
        )

        return current_in_box or future_in_box

    def resolve_collision_with_agent(self, agent):
        """Resolve collision between this transport object and an agent"""
        # check if agent is completely inside the box, which should never happen
        if self.check_if_point_inside(agent.position):
            # find nearest edge and push agent outside
            half_width = self.width / 2
            half_height = self.height / 2

            # calculate distance to each edge
            dist_to_left = abs(agent.position[0] - (self.position[0] - half_width))
            dist_to_right = abs(agent.position[0] - (self.position[0] + half_width))
            dist_to_top = abs(agent.position[1] - (self.position[1] - half_height))
            dist_to_bottom = abs(agent.position[1] - (self.position[1] + half_height))

            # find closest edge
            min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)

            # push agent out through the closest edge
            if min_dist == dist_to_left:
                agent.position[0] = self.position[0] - half_width - agent.radius - 1
            elif min_dist == dist_to_right:
                agent.position[0] = self.position[0] + half_width + agent.radius + 1
            elif min_dist == dist_to_top:
                agent.position[1] = self.position[1] - half_height - agent.radius - 1
            else:  # bottom
                agent.position[1] = self.position[1] + half_height + agent.radius + 1

            # stop the agent's movement
            agent.velocity = np.array([0.0, 0.0])
            return

        # standard collision resolution for agent at the edge
        # calculate the closest point on the rectangle to the circle
        closest_x = max(
            self.position[0] - self.width / 2,
            min(agent.position[0], self.position[0] + self.width / 2),
        )
        closest_y = max(
            self.position[1] - self.height / 2,
            min(agent.position[1], self.position[1] + self.height / 2),
        )

        closest_point = np.array([closest_x, closest_y])

        # vector from closest point to circle center
        normal = agent.position - closest_point
        distance = np.linalg.norm(normal)

        # avoid division by zero
        if distance < 0.0001:
            # if agent is exactly at the edge, move it slightly away
            # choose direction based on velocity
            if abs(agent.velocity[0]) > abs(agent.velocity[1]):
                normal = np.array([np.sign(agent.velocity[0]), 0.0])
            else:
                normal = np.array([0.0, np.sign(agent.velocity[1])])

            if np.linalg.norm(normal) < 0.0001:
                normal = np.array([1.0, 0.0])  # default if velocity is zero

            distance = 0.0001

        # normalize normal vector
        normal = normal / distance

        # calculate overlap
        overlap = agent.radius - distance

        # only proceed if there's an actual overlap
        if overlap > 0:
            # move the agent out of collision with some extra margin
            agent.position += normal * (overlap + 0.5)

            # calculate relative velocity
            relative_velocity = agent.velocity - self.velocity

            # calculate velocity along the normal
            velocity_along_normal = np.dot(relative_velocity, normal)

            # if objects are moving away from each other, don't apply impulse
            if velocity_along_normal > 0:
                return

            # calculate impulse scalar with reduced restitution for stability
            j = -(1 + self.restitution) * velocity_along_normal
            j /= 1 / agent.mass + 1 / self.mass

            # limit the maximum impulse to prevent instability
            j = np.clip(j, -150, 150)

            # apply impulse
            impulse = j * normal
            agent.velocity -= impulse / agent.mass
            self.velocity += impulse / self.mass * 0.8  # further reduce impact on box

    def render(self, screen):
        """Render the transport object"""
        # calculate rectangle coordinates
        rect_x = int(self.position[0] - self.width / 2)
        rect_y = int(self.position[1] - self.height / 2)

        # draw the object as a rectangle
        pygame.draw.rect(screen, self.color, (rect_x, rect_y, self.width, self.height))

        # draw center marker
        center = self.position.astype(int)
        marker_size = min(self.width, self.height) // 4
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (center[0] - marker_size, center[1]),
            (center[0] + marker_size, center[1]),
            2,
        )
        pygame.draw.line(
            screen,
            (255, 255, 255),
            (center[0], center[1] - marker_size),
            (center[0], center[1] + marker_size),
            2,
        )
