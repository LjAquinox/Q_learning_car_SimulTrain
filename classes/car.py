import pygame
import math
from config import *
from classes.helpers import line_segment_intersection
import numpy as np
import torch
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
        self.ray_distances = np.ones(self.num_rays) * self.ray_length
        self.rays_end_points = np.zeros((self.num_rays, 2)) # For display

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

    def set_wall_tensors(self, wall_p1_t, wall_s, device):
        self.wall_p1_t = wall_p1_t
        self.wall_s = wall_s
        self.device = device

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
        #normalize ray distances
        ray_distances = ray_distances / self.ray_length
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
            *ray_distances,
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

    def cast_rays(self):
        # 1) directions en numpy
        start_angle = self.angle - self.ray_spread / 2
        angles = np.linspace(start_angle, start_angle + self.ray_spread, self.num_rays)
        ray_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (N,2)

        # 1) Créer un tenseur (N,2) dont chaque ligne = [self.pos.x, self.pos.y]
        coord = torch.tensor([self.pos.x, self.pos.y],
                             dtype=torch.float32,
                             device=self.device)
        p1_t = coord.unsqueeze(0).repeat(self.num_rays, 1)  # shape (N,2)

        # 2) Calculer p2_t
        ray_dirs = torch.as_tensor(ray_dirs, dtype=torch.float32, device=self.device)  # (N,2)
        p2_t = p1_t + ray_dirs * self.ray_length

        # 3) Intersection
        inter_t = line_segment_intersection(
            p1_t, p2_t,
            self.wall_p1_t,
            self.wall_s
        )  # (N,2)

        # 4) Retour en NumPy pour Pygame
        inter_np = inter_t.cpu().numpy()
        self.rays_end_points = inter_np
        dists = torch.linalg.norm(inter_t - p1_t, dim=1)
        self.ray_distances = dists.cpu().numpy()
            
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
        """
        Checks collision of the oriented car rectangle against wall segments.
        Returns True if any segment intersects.
        """
        # 1) Build the un‑rotated corners array (4×2)
        car_rect = pygame.Rect(0, 0, self.height, self.width)
        car_rect.center = self.pos
        corners = np.array([
            car_rect.topleft,
            car_rect.topright,
            car_rect.bottomright,
            car_rect.bottomleft,
        ], dtype=float)  # shape (4,2)

        # 2) Build rotation matrix
        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        R = np.array([[ cos_a, -sin_a],
                    [ sin_a,  cos_a]])        # shape (2,2)

        # 3) Rotate all corners at once around center
        center = np.array(self.pos, dtype=float)  # shape (2,)
        rotated = (corners - center) @ R.T + center  # shape (4,2)

        # 4) Conversion en tenseurs sur le bon device
        device = self.device
        p1_t = torch.as_tensor(rotated, dtype=torch.float32, device=device)  # (4,2)
        p2_t = torch.roll(p1_t, shifts=-1, dims=0)  # (4,2)

        # 5) Intersection via torch
        #    self.wall_p1_t et self.wall_s ont été initialisés une fois dans load_map()
        inter_t = line_segment_intersection(
            p1_t, p2_t,
            self.wall_p1_t,
            self.wall_s
        )  # (4,2) résultant des points d'intersection ou 0

        # 6) Détection de collision : une intersection non‑nulle signifie collision
        hit_mask = torch.any(inter_t != 0.0, dim=1)  # (4,) bool
        return bool(hit_mask.any())
    

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