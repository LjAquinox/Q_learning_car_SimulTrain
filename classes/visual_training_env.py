import numpy as np
import time
from classes.game import Game
from classes.visual_car import VisualCar
import pygame
import os
from config import *

class VisualTrainingEnvironment:
    def __init__(self, map_name=DEFAULT_MAP):
        self.map_name = map_name
        # Initialize pygame without display
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pygame.display.set_mode((1, 1))
        pygame.init()

        self.game = Game()
        # Replace the car with VisualCar and pass the game reference
        self.game.car = VisualCar(self.game.start_pos[0], self.game.start_pos[1])
        self.game.load_map(self.map_name)
        self.prev_distance_to_next_gate_start = None
        self.prev_distance_to_next_gate_end = None
        self.gate_reward = 0
        self.speed_reward = 0
        self.gates_distance_reward = 0

    def _calculate_reward(self):
        reward = -1

        # Reward for passing gates
        if self.game.gate_passed:
            reward += 25
            self.gate_reward += 25

        # Penalty for collisions
        if self.game.game_over:
            reward -= 100

        # Reward for speed
        reward += (self.game.car.vel.length() / self.game.car.max_speed) / 25
        self.speed_reward += (self.game.car.vel.length() / self.game.car.max_speed) / 25

        return reward

    def step(self, action):
        # Convert action to car controls
        # Action space: 0=accelerate forward, 1=brake, 2=left, 3=right, 4 = accelerate and left, 5 = accelerate and right
        match action:
            case 0: #accelerate forward
                self.game.car.accelerating = True
                self.game.car.braking = False
                self.game.car.steering = 0
            case 1: #brake
                self.game.car.accelerating = False
                self.game.car.braking = True
                self.game.car.steering = 0
            case 2: #left
                self.game.car.accelerating = False
                self.game.car.braking = False
                self.game.car.steering = -1
            case 3: #right
                self.game.car.accelerating = False
                self.game.car.braking = False
                self.game.car.steering = 1
            case 4: #accelerate and left
                self.game.car.accelerating = True
                self.game.car.braking = False
                self.game.car.steering = -1
            case 5: #accelerate and right
                self.game.car.accelerating = True
                self.game.car.braking = False
                self.game.car.steering = 1

        # Update game state
        self.game.update(1)

        # Get new state and reward
        state = self.game.car.get_state(self.game.screen)
        reward = self._calculate_reward()
        done = self.game.game_over

        return state, reward, done

    def reset(self):
        self.prev_distance_to_next_gate_start = None
        self.prev_distance_to_next_gate_end = None
        self.gate_reward = 0
        self.speed_reward = 0
        self.gates_distance_reward = 0
        self.game.reset_game()
        return self.game.car.get_state(self.game.screen)

    def close(self):
        pygame.quit() 