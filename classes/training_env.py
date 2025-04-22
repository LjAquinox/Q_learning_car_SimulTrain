import numpy as np
import time
from classes.game import Game
from classes.car import Car
import pygame
import os
from config import *

class TrainingEnvironment:
    def __init__(self, map_name=DEFAULT_MAP):
        self.map_name = map_name
        # Initialize pygame without display
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.display.set_mode((1, 1))
        pygame.init()
        
        self.game = Game()
        self.game.load_map(self.map_name)
            
    def _calculate_reward(self):
        reward = 0
        
        # Reward for passing gates
        if self.game.gate_passed:
            reward += 10
            
        # Penalty for collisions
        if self.game.game_over:
            reward -= 10
            
        # Reward for speed  
        reward += self.game.car.vel.length() * 0.01
        """
        # Penalty for being too far from next gate
        if self.game.gates and self.game.current_gate_index < len(self.game.gates):
            self.game.car.set_next_gate_info(self.game.gates, self.game.current_gate_index)
            distance_start, _, distance_end, _ = self.game.car.distance_start, self.game.car.relative_angle_start, self.game.car.distance_end, self.game.car.relative_angle_end
            reward -= (distance_start + distance_end) * 0.01
        """
        return reward
        
    def step(self, action):
        # Convert action to car controls
        # Action space: 0=no action, 1=accelerate forward, 2=brake, 3=left, 4=right, 5 = accelerate and left, 6 = accelerate and right
        match action:
            case 0:
                self.game.car.accelerating = False
                self.game.car.braking = False
                self.game.car.steering = 0
            case 1:
                self.game.car.accelerating = True
                self.game.car.braking = False
                self.game.car.steering = 0
            case 2:
                self.game.car.accelerating = False
                self.game.car.braking = True
                self.game.car.steering = 0
            case 3:
                self.game.car.accelerating = False
                self.game.car.braking = False
                self.game.car.steering = -1
            case 4:
                self.game.car.accelerating = False
                self.game.car.braking = False
                self.game.car.steering = 1
            case 5:
                self.game.car.accelerating = True
                self.game.car.braking = False
                self.game.car.steering = -1
            case 6:
                self.game.car.accelerating = True
                self.game.car.braking = False
                self.game.car.steering = 1
        
        # Update game state
        self.game.update(1/60)  # Assuming 60 FPS
        
        # Get new state and reward
        state = self.game.car.get_state()
        reward = self._calculate_reward()
        done = self.game.game_over
        
        return state, reward, done
        
    def reset(self):
        self.game.reset_game()
        return self.game.car.get_state()
        
    def close(self):
        pygame.quit() 