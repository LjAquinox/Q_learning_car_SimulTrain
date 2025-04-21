import pygame
import math
from config import *
from classes.helpers import line_segment_intersection

class Car:
    def __init__(self, x, y):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.angle = 0  # Radians, 0 = right
        self.width = 15
        self.height = 30
        self.color = BLUE

        # Physics
        self.min_speed = 0.3
        self.max_speed = 5.0
        self.acceleration = 0.1
        self.brake_power = 0.1
        self.friction = 0.06 # Friction coefficient
        self.turn_speed = 0.05 # Radians per frame at full speed
        self.min_speed_for_turn = 0.5 # Minimum speed to be able to turn

        self.accelerating = False
        self.braking = False
        self.steering = 0 # -1 left, 0 straight, 1 right

        # Raycasting
        self.num_rays = 15 # Number of rays
        self.ray_length = 800 # Maximum ray length
        self.ray_spread = math.pi * 1.5 # Total angle covered by rays (e.g., 270 degrees)
        self.ray_distances = [self.ray_length] * self.num_rays
        self.rays_end_points = [(0,0)] * self.num_rays # For display

        self.distance_start = 0
        self.distance_end = 0
        self.relative_angle_start = 0
        self.relative_angle_end = 0

        # Load and process car image
        self.car_image = pygame.image.load("car.png")
        self.car_image = pygame.transform.scale(self.car_image, (self.height, self.width))
        # Convert white to transparent
        for x in range(self.car_image.get_width()):
            for y in range(self.car_image.get_height()):
                color = self.car_image.get_at((x, y))
                if color[0] > 240 and color[1] > 240 and color[2] > 240:  # Check if pixel is close to white
                    self.car_image.set_at((x, y), (255, 255, 255, 0))  # Set fully transparent

    def get_state(self):
        """ Returns the current state for the Q-learning agent.
            Returns a tuple containing:
            - Ray distances (list of floats)
            - Next gate distance and angle (4 floats: distance_start, angle_start, distance_end, angle_end)
            - Car's speed (float)
            - Car's angle (float)
        """
        # Get ray distances
        ray_distances = self.ray_distances
        distance_start = self.distance_start
        relative_angle_start = (self.relative_angle_start + math.pi) / (2 * math.pi)
        distance_end = self.distance_end
        relative_angle_end = (self.relative_angle_end + math.pi) / (2 * math.pi)

        # Get and normalize some car properties
        speed = (self.vel.length() - self.min_speed) / (self.max_speed - self.min_speed)  # Normalize speed
        vel_x = (self.vel.x - self.min_speed) / (self.max_speed - self.min_speed)  # Normalize x velocity
        vel_y = (self.vel.y - self.min_speed) / (self.max_speed - self.min_speed)  # Normalize y velocity
        angle = (self.angle + math.pi) / (2 * math.pi)  # Normalize angle to 0-1
        
        # Combine all state information
        state = (
            *ray_distances,  # Unpack raw ray distances idk how to properly normalize them
            distance_start, relative_angle_start, distance_end, relative_angle_end,  # Next gate info
            speed, vel_x, vel_y,  # Velocity info
            angle  # Car's angle
        )
        #print(state)
        return state


    def update(self, dt):
        # 1. Apply friction

        direction = pygame.Vector2(math.cos(self.angle), math.sin(self.angle))

        if self.vel.length() > self.min_speed:
            friction_force = self.vel.normalize() * self.friction
            
            # Ensure friction doesn't reverse velocity
            if self.vel.length() > friction_force.length() * dt :
                 self.vel -= friction_force * dt
            else:
                 self.vel = pygame.Vector2(0, 0) + self.min_speed * direction
        
        # 2. Apply acceleration / braking
        if self.accelerating:
            self.vel += direction * self.acceleration * dt
        if self.braking:
             # Braking is more effective at reducing current speed
             if self.vel.length() > self.min_speed: 
                 brake_force = self.vel.normalize() * self.brake_power
                 if self.vel.length() > brake_force.length() * dt:
                      self.vel -= brake_force * dt
                 else:
                      self.vel = pygame.Vector2(0,0) + self.min_speed * direction

        # 3. Limit speed
        if self.vel.length() > self.max_speed:
            self.vel.scale_to_length(self.max_speed)

        # 4. Apply rotation (only if moving a minimum amount)
        current_speed = self.vel.length()
        if abs(self.steering) > 0 and current_speed > self.min_speed_for_turn:
             # Rotation depends on speed (more realistic)
             # Turns slower at low speed, faster at high speed (up to turn_speed)
             turn_rate = self.turn_speed * (current_speed / self.max_speed)
             self.angle += self.steering * turn_rate * dt

        # 5. Update position
        self.pos += self.vel * dt

    def cast_rays(self, walls):
        """ Casts rays from the car and calculates distances to walls. """
        self.ray_distances = []
        self.rays_end_points = [] # Clear previous endpoints for drawing

        start_angle = self.angle - self.ray_spread / 2

        for i in range(self.num_rays):
            ray_angle = start_angle + i * (self.ray_spread / (self.num_rays - 1)) if self.num_rays > 1 else self.angle
            
            ray_dir = pygame.Vector2(math.cos(ray_angle), math.sin(ray_angle))
            ray_end_theoretical = self.pos + ray_dir * self.ray_length

            closest_dist = self.ray_length
            actual_ray_end = ray_end_theoretical

            for wall in walls:
                p3 = pygame.Vector2(wall[0])
                p4 = pygame.Vector2(wall[1])
                intersection = line_segment_intersection(self.pos, ray_end_theoretical, p3, p4)

                if intersection:
                    dist = (intersection - self.pos).length()
                    if dist < closest_dist:
                        closest_dist = dist
                        actual_ray_end = intersection # Update endpoint to the actual hit point

            self.ray_distances.append(closest_dist)
            self.rays_end_points.append(actual_ray_end)

    def set_next_gate_info(self, gates, current_gate_index):
        """Calculate the relative position of the next gate to the car.
            Returns (distance, angle) where angle is relative to car's direction for
            both start and end of the gate. angle is 0 to 1
        """
        if not gates or current_gate_index >= len(gates):
            return (0, 0)  # No gate to track
            
        gate = gates[current_gate_index]
        
        # Calculate vector from car to gate
        to_gate_start = pygame.Vector2(gate[0]) - self.pos
        to_gate_end = pygame.Vector2(gate[1]) - self.pos
        
        # Calculate distance
        self.distance_start = to_gate_start.length()
        self.distance_end = to_gate_end.length()
        
        # Calculate angle relative to car's direction
        # First get the angle of the vector to gate
        gate_angle_start = math.atan2(to_gate_start.y, to_gate_start.x)
        gate_angle_end = math.atan2(to_gate_end.y, to_gate_end.x)
        # Then subtract car's angle to get relative angle
        relative_angle_start = gate_angle_start - self.angle
        relative_angle_end = gate_angle_end - self.angle
        # Normalize angle to [-pi, pi]
        relative_angle_start = (relative_angle_start + math.pi) % (2 * math.pi) - math.pi
        relative_angle_end = (relative_angle_end + math.pi) % (2 * math.pi) - math.pi
        
        self.relative_angle_start = relative_angle_start
        self.relative_angle_end = relative_angle_end
        

    def check_collision_with_elements(self, elements):
        """ Checks simple collision of the car with walls.
            Returns True if collision, False otherwise.
            Uses a simplified oriented rectangle.
        """
        # Create a collision rectangle aligned with the car
        car_rect = pygame.Rect(0, 0, self.height, self.width) # Note: height/width swapped to match direction 0
        car_rect.center = self.pos

        # Points of the non-rotated rectangle
        points = [car_rect.topleft, car_rect.topright, car_rect.bottomright, car_rect.bottomleft]

        # Rotate points around the car's center
        center = pygame.Vector2(self.pos)
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        rotated_points = []
        for p in points:
            p_vec = pygame.Vector2(p) - center
            x_new = p_vec.x * cos_a - p_vec.y * sin_a
            y_new = p_vec.x * sin_a + p_vec.y * cos_a
            rotated_points.append(center + pygame.Vector2(x_new, y_new))

        # Segments of the car's collision rectangle
        car_segments = [
            (rotated_points[0], rotated_points[1]),
            (rotated_points[1], rotated_points[2]),
            (rotated_points[2], rotated_points[3]),
            (rotated_points[3], rotated_points[0]),
        ]

        # Check intersection of each car segment with each wall
        for car_seg_start, car_seg_end in car_segments:
            for elem in elements:
                elem_start = pygame.Vector2(elem[0])
                elem_end = pygame.Vector2(elem[1])
                if line_segment_intersection(car_seg_start, car_seg_end, elem_start, elem_end):
                    return True # Collision detected
        return False # No collision
    

    def draw(self, screen):
        # Draw the car as an oriented rectangle
        points = [
            (-self.height / 2, -self.width / 2), (+self.height / 2, -self.width / 2),
            (+self.height / 2, +self.width / 2), (-self.height / 2, +self.width / 2)
        ]
        
        # Rotate and translate points
        rotated_points = []
        for x, y in points:
            x_rot = x * math.cos(self.angle) - y * math.sin(self.angle)
            y_rot = x * math.sin(self.angle) + y * math.cos(self.angle)
            rotated_points.append((self.pos.x + x_rot, self.pos.y + y_rot))
        
        # Only rotate the pre-processed image
        rotated_car = pygame.transform.rotate(self.car_image, -self.angle * 180 / math.pi)
        car_rect = rotated_car.get_rect(center=self.pos)
        screen.blit(rotated_car, car_rect)
        
        # Draw a line to indicate the front
        front_point = self.pos + pygame.Vector2(math.cos(self.angle), math.sin(self.angle)) * (self.height / 2)
        pygame.draw.line(screen, WHITE, self.pos, front_point, 2)

    def draw_rays(self, screen):
         # Draw rays (for debug)
         for end_point in self.rays_end_points:
              pygame.draw.line(screen, YELLOW, self.pos, end_point, 1)

    def draw_rays_distances(self, screen, game_font):
        # Draw ray distances (for debug)
        for i, dist in enumerate(self.ray_distances):
            dist_text = game_font.render(f"R{i}: {dist:.1f}", True, YELLOW)
            screen.blit(dist_text, (10, 30 + i * 15))