import pygame
import numpy as np
import random


class Obstacle:
    """Base obstacle class with simplified implementation"""

    RECTANGLE = 0
    CIRCLE = 1
    POLYGON = 2

    def __init__(self, position, shape_type, color=(100, 100, 100)):
        self.position = np.array(position, dtype=float)
        self.shape_type = shape_type
        self.color = color
        self.restitution = 0.3  # bounciness factor

    def check_collision_with_agent(self, agent):
        """Check if the obstacle collides with an agent"""
        raise NotImplementedError("Subclasses must implement this method")

    def resolve_collision_with_agent(self, agent):
        """Resolve collision between obstacle and agent"""
        raise NotImplementedError("Subclasses must implement this method")

    def render(self, screen):
        """Render the obstacle"""
        raise NotImplementedError("Subclasses must implement this method")


class RectangleObstacle(Obstacle):
    """A rectangle-shaped obstacle"""

    def __init__(self, position, width, height, angle=0, **kwargs):
        super().__init__(position, Obstacle.RECTANGLE, **kwargs)
        self.width = width
        self.height = height
        self.angle = angle  # rotation angle in degrees

        # calculate corners for collision detection
        self.update_corners()

    def update_corners(self):
        """Update the corners of the rectangle based on position and angle"""
        angle_rad = np.radians(self.angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        half_width = self.width / 2
        half_height = self.height / 2
        local_corners = [
            np.array([-half_width, -half_height]),
            np.array([half_width, -half_height]),
            np.array([half_width, half_height]),
            np.array([-half_width, half_height]),
        ]

        self.corners = []
        for corner in local_corners:
            rotated_x = corner[0] * cos_a - corner[1] * sin_a
            rotated_y = corner[0] * sin_a + corner[1] * cos_a
            self.corners.append(self.position + np.array([rotated_x, rotated_y]))

    def check_collision_with_agent(self, agent):
        """Check if the rectangle collides with an agent"""
        # for non-rotated rectangles, use a simpler approach
        if self.angle % 90 == 0:
            half_width = self.width / 2
            half_height = self.height / 2

            # find closest point on rectangle to agent
            closest_x = max(
                self.position[0] - half_width,
                min(agent.position[0], self.position[0] + half_width),
            )
            closest_y = max(
                self.position[1] - half_height,
                min(agent.position[1], self.position[1] + half_height),
            )

            # calculate distance to closest point
            distance = np.linalg.norm(np.array([closest_x, closest_y]) - agent.position)
            return distance < agent.radius
        else:
            # for rotated rectangles, find closest point among edges
            closest_dist = float("inf")

            for i in range(len(self.corners)):
                p1 = self.corners[i]
                p2 = self.corners[(i + 1) % len(self.corners)]

                # find closest point on this edge
                point = self._closest_point_on_segment(p1, p2, agent.position)
                dist = np.linalg.norm(point - agent.position)
                closest_dist = min(closest_dist, dist)

            return closest_dist < agent.radius

    def _closest_point_on_segment(self, a, b, p):
        """Find closest point on line segment AB to point P"""
        ab = b - a
        ab_squared = np.dot(ab, ab)

        if ab_squared == 0:
            return a

        ap = p - a
        t = max(0, min(1, np.dot(ap, ab) / ab_squared))
        return a + t * ab

    def resolve_collision_with_agent(self, agent):
        """Resolve collision between rectangle and agent"""
        # find closest point on rectangle to agent
        if self.angle % 90 == 0:
            half_width = self.width / 2
            half_height = self.height / 2

            closest_x = max(
                self.position[0] - half_width,
                min(agent.position[0], self.position[0] + half_width),
            )
            closest_y = max(
                self.position[1] - half_height,
                min(agent.position[1], self.position[1] + half_height),
            )
            closest_point = np.array([closest_x, closest_y])
        else:
            # find closest point among all edges for rotated rectangle
            closest_point = None
            min_dist = float("inf")

            for i in range(len(self.corners)):
                p1 = self.corners[i]
                p2 = self.corners[(i + 1) % len(self.corners)]
                point = self._closest_point_on_segment(p1, p2, agent.position)
                dist = np.linalg.norm(point - agent.position)

                if dist < min_dist:
                    min_dist = dist
                    closest_point = point

        # direction from closest point to agent
        normal = agent.position - closest_point
        distance = np.linalg.norm(normal)

        if distance < 0.0001:
            normal = np.array([1.0, 0.0])
            distance = 0.0001

        normal = normal / distance
        overlap = agent.radius - distance

        if overlap > 0:
            # move agent out of collision
            agent.position += normal * (overlap + 0.5)

            # calculate reflection for velocity
            dot_product = np.dot(agent.velocity, normal)
            if dot_product < 0:
                agent.velocity = agent.velocity - 2 * dot_product * normal
                agent.velocity *= self.restitution

    def render(self, screen):
        """Render the rectangle"""
        rect_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(rect_surface, self.color, (0, 0, self.width, self.height))

        if self.angle != 0:
            rect_surface = pygame.transform.rotate(rect_surface, self.angle)

        blit_pos = (
            self.position[0] - rect_surface.get_width() // 2,
            self.position[1] - rect_surface.get_height() // 2,
        )
        screen.blit(rect_surface, blit_pos)


class CircleObstacle(Obstacle):
    """A circle-shaped obstacle"""

    def __init__(self, position, radius, **kwargs):
        super().__init__(position, Obstacle.CIRCLE, **kwargs)
        self.radius = radius

    def check_collision_with_agent(self, agent):
        """Check if the circle collides with an agent"""
        distance = np.linalg.norm(self.position - agent.position)
        return distance < (self.radius + agent.radius)

    def resolve_collision_with_agent(self, agent):
        """Resolve collision between circle and agent"""
        normal = agent.position - self.position
        distance = np.linalg.norm(normal)

        if distance < 0.0001:
            normal = np.array([1.0, 0.0])
            distance = 0.0001

        normal = normal / distance
        overlap = (self.radius + agent.radius) - distance

        if overlap > 0:
            agent.position += normal * (overlap + 0.5)

            dot_product = np.dot(agent.velocity, normal)
            if dot_product < 0:
                agent.velocity = agent.velocity - 2 * dot_product * normal
                agent.velocity *= self.restitution

    def render(self, screen):
        """Render the circle"""
        pygame.draw.circle(
            screen, self.color, self.position.astype(int), int(self.radius)
        )


class PolygonObstacle(Obstacle):
    """A polygon-shaped obstacle"""

    def __init__(self, position, vertices, **kwargs):
        super().__init__(position, Obstacle.POLYGON, **kwargs)
        # vertices are in local space, relative to position
        self.local_vertices = vertices
        self.vertices = [self.position + v for v in self.local_vertices]

    def check_collision_with_agent(self, agent):
        """Check if the polygon collides with an agent"""
        # check if agent center is inside polygon
        if self._point_inside_polygon(agent.position):
            return True

        # check if any edge is close to agent
        for i in range(len(self.vertices)):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % len(self.vertices)]

            closest = self._closest_point_on_segment(p1, p2, agent.position)
            distance = np.linalg.norm(closest - agent.position)

            if distance < agent.radius:
                return True

        return False

    def _point_inside_polygon(self, point):
        """Check if point is inside polygon using ray casting algorithm"""
        inside = False
        n = len(self.vertices)

        j = n - 1
        for i in range(n):
            if (
                (self.vertices[i][1] > point[1]) != (self.vertices[j][1] > point[1])
            ) and (
                point[0]
                < (self.vertices[j][0] - self.vertices[i][0])
                * (point[1] - self.vertices[i][1])
                / (self.vertices[j][1] - self.vertices[i][1])
                + self.vertices[i][0]
            ):
                inside = not inside
            j = i

        return inside

    def _closest_point_on_segment(self, a, b, p):
        """Find closest point on segment AB to point P"""
        ab = b - a
        ab_squared = np.dot(ab, ab)

        if ab_squared == 0:
            return a

        ap = p - a
        t = max(0, min(1, np.dot(ap, ab) / ab_squared))
        return a + t * ab

    def resolve_collision_with_agent(self, agent):
        """Resolve collision between polygon and agent"""
        if self._point_inside_polygon(agent.position):
            # find closest edge and push agent out
            closest_edge = None
            min_dist = float("inf")
            closest_point = None

            for i in range(len(self.vertices)):
                p1 = self.vertices[i]
                p2 = self.vertices[(i + 1) % len(self.vertices)]

                point = self._closest_point_on_segment(p1, p2, agent.position)
                dist = np.linalg.norm(point - agent.position)

                if dist < min_dist:
                    min_dist = dist
                    closest_point = point

            # direction from closest point to agent (outward normal)
            normal = agent.position - closest_point
            distance = np.linalg.norm(normal)

            if distance < 0.0001:
                # if agent at edge, use edge normal
                if closest_edge:
                    edge = closest_edge[1] - closest_edge[0]
                    normal = np.array([-edge[1], edge[0]])  # perpendicular
                else:
                    normal = np.array([1.0, 0.0])
                distance = 0.0001

            normal = normal / distance

            # push agent out with buffer
            agent.position = closest_point + normal * (agent.radius + 0.5)

            # reflect velocity
            dot_product = np.dot(agent.velocity, normal)
            if dot_product < 0:
                agent.velocity = agent.velocity - 2 * dot_product * normal
                agent.velocity *= self.restitution

        else:
            # find closest edge
            min_dist = float("inf")
            closest_point = None

            for i in range(len(self.vertices)):
                p1 = self.vertices[i]
                p2 = self.vertices[(i + 1) % len(self.vertices)]

                point = self._closest_point_on_segment(p1, p2, agent.position)
                dist = np.linalg.norm(point - agent.position)

                if dist < min_dist:
                    min_dist = dist
                    closest_point = point

            # direction from closest point to agent
            normal = agent.position - closest_point
            distance = np.linalg.norm(normal)

            if distance < 0.0001:
                normal = np.array([1.0, 0.0])
                distance = 0.0001

            normal = normal / distance
            overlap = agent.radius - distance

            if overlap > 0:
                agent.position += normal * (overlap + 0.5)

                dot_product = np.dot(agent.velocity, normal)
                if dot_product < 0:
                    agent.velocity = agent.velocity - 2 * dot_product * normal
                    agent.velocity *= self.restitution

    def render(self, screen):
        """Render the polygon"""
        points = [(int(v[0]), int(v[1])) for v in self.vertices]
        pygame.draw.polygon(screen, self.color, points)


def create_random_obstacle(min_x, max_x, min_y, max_y):
    """Create a random obstacle within the given bounds"""
    # choose a random shape type
    shape_type = random.choice([Obstacle.RECTANGLE, Obstacle.CIRCLE, Obstacle.POLYGON])

    # random position with margin
    margin = 30
    x = random.uniform(min_x + margin, max_x - margin)
    y = random.uniform(min_y + margin, max_y - margin)
    position = np.array([x, y])

    # random darker color for contrast
    color = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))

    # create specific obstacle type
    if shape_type == Obstacle.RECTANGLE:
        width = random.randint(20, 80)
        height = random.randint(20, 80)
        angle = random.randint(0, 359)
        return RectangleObstacle(position, width, height, angle, color=color)

    elif shape_type == Obstacle.CIRCLE:
        radius = random.randint(15, 40)
        return CircleObstacle(position, radius, color=color)

    else:  # POLYGON
        # create a random polygon with 3-6 vertices
        num_vertices = random.randint(3, 6)
        radius = random.randint(15, 40)

        # generate vertices in a circle with randomization
        vertices = []
        for i in range(num_vertices):
            angle = 2 * np.pi * i / num_vertices
            vx = np.cos(angle) * radius + random.uniform(-radius * 0.2, radius * 0.2)
            vy = np.sin(angle) * radius + random.uniform(-radius * 0.2, radius * 0.2)
            vertices.append(np.array([vx, vy]))

        return PolygonObstacle(position, vertices, color=color)
