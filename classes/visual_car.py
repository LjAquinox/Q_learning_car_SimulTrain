import pygame
import numpy as np
import math
from classes.car import Car
from config import *

class VisualCar(Car):
    def __init__(self, x, y):
        super().__init__(x, y)
        # Remove ray-based sensing attributes since we won't use them
        self.num_rays = 0
        self.ray_length = 0
        self.ray_spread = 0
        self.ray_distances = None
        self.rays_end_points = None
        self.relative_angle_gates = None
        self.distance_gates = None

    def get_state(self, screen):
        """Returns the current state for the Q-learning agent.
        Returns a tuple containing:
        - Visual input (numpy array of the game screen)
        - Car's speed (float)
        - Car's angle (float)
        """

        # Convert to grayscale and normalize
        screen = pygame.surfarray.array3d(screen)
        # keep every other pixel
        screen = screen[::2, ::2]
        screen = np.mean(screen, axis=2) / 255.0
        
        # Get and normalize car properties
        speed = (self.vel.length() - self.min_speed) / (self.max_speed - self.min_speed)
        vel_x = (self.vel.x - self.min_speed) / (self.max_speed - self.min_speed)
        vel_y = (self.vel.y - self.min_speed) / (self.max_speed - self.min_speed)
        angle = (self.angle + math.pi) / (2 * math.pi)
        
        # Combine all state information
        state = (
            screen,  # Visual input
            speed, vel_x, vel_y,  # Velocity info
            angle  # Car's angle
        )
        return state

    def draw(self, screen):
        # Call parent's draw method
        super().draw(screen)
        
        # No need to draw rays since we're not using them
        pass

    def draw_rays(self, screen):
        # No need to draw rays since we're not using them
        pass

    def draw_rays_distances(self, screen, game_font):
        # No need to draw ray distances since we're not using them
        pass 