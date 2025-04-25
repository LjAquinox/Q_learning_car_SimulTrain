import pygame
import json
import os
import math
from config import *
from classes.car import Car
import numpy as np
import torch

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Car Game Runner")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.running = True

        self.walls = []
        self.wall_vectors = np.zeros((1, 2, 2)) #we don't know how many walls there are yet
        self.gates = []
        self.gate_vectors = np.zeros((1, 2, 2)) #we don't know how many gates there are yet
        self.gate_range = 2 # how many gates we will track on AI side
        self.start_pos = (WIDTH // 2, HEIGHT // 2) # Default if map has no start_pos

        self.car = Car(self.start_pos[0], self.start_pos[1])
        
        self.show_rays = True  # Toggle ray display with 'R'
        self.game_over = False

        self.load_map(DEFAULT_MAP) # Load the default map
        
        # Gate tracking
        self.current_gate_index = 0  # Index of the next gate to pass
        self.gate_passed = False  # Flag to prevent multiple triggers
        self.score = 0  # Track score
        
        # Restart button properties
        self.restart_button = pygame.Rect(WIDTH//2 - 50, HEIGHT//2 + 30, 100, 40)
        self.restart_text = self.font.render("Restart", True, WHITE)
        self.restart_text_rect = self.restart_text.get_rect(center=self.restart_button.center)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.car.device = self.device

    def load_map(self, filename):
        filepath = os.path.join(MAP_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                map_data = json.load(f)
                self.walls = map_data.get("walls", [])
                self.gates = map_data.get("gates", [])
                # Use the map's start position if it exists
                loaded_start_pos = map_data.get("start_pos")
                if loaded_start_pos:
                    self.start_pos = tuple(loaded_start_pos) 
                else:
                     print(f"Warning: No start position found in map '{filename}'. Using default {self.start_pos}.")
                     # Use the default value already defined
                # Pre-calculate wall vectors
                self.wall_vectors = np.zeros((len(self.walls), 2, 2))
                for i, wall in enumerate(self.walls):
                    self.wall_vectors[i] = np.array([wall[0], wall[1]])
                self.gate_vectors = np.zeros((len(self.gates), 2, 2))
                for i, gate in enumerate(self.gates):
                    self.gate_vectors[i] = np.array([gate[0], gate[1]])

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # p3 = shape (M,2), p4 = shape (M,2)
                self.wall_p1_t = torch.as_tensor(self.wall_vectors[:,0,:], dtype=torch.float32, device=self.device)  # shape (M,2)
                self.wall_p2_t = torch.as_tensor(self.wall_vectors[:,1,:], dtype=torch.float32, device=self.device)  # shape (M,2)
                # Pre‑compute of vector s = p4 − p3
                self.wall_s    = self.wall_p2_t - self.wall_p1_t       
                
                self.gate_p1_t = torch.as_tensor(self.gate_vectors[:,0,:], dtype=torch.float32, device=self.device)  # shape (N,2)
                self.gate_p2_t = torch.as_tensor(self.gate_vectors[:,1,:], dtype=torch.float32, device=self.device)  # shape (N,2)
                # Pre‑compute of vector s = p4 − p3
                self.gate_s    = self.gate_p2_t - self.gate_p1_t       

                print(f"Map '{filename}' loaded successfully.")
                # Reset car position if map is loaded after init
                if hasattr(self, 'car'):
                    self.car.pos = pygame.Vector2(self.start_pos)
                    self.car.vel = pygame.Vector2(0, 0)
                    self.car.angle = 0 # Reset angle too
                    self.game_over = False
                    self.current_gate_index = 0  # Reset gate tracking
                    self.gate_passed = False
                    self.score = 0
                    self.car.set_wall_tensors(self.wall_p1_t, self.wall_s, self.device)
                    self.car.set_gates_tensors(self.gate_p1_t, self.gate_s, self.device)

        except FileNotFoundError:
            print(f"Error: Map file not found '{filepath}'. Using empty map.")
            self.walls = []
            self.gates = []
            # Keep the default start position
        except Exception as e:
            print(f"Error loading map: {e}. Using empty map.")
            self.walls = []
            self.gates = []
             # Keep the default start position

    def handle_input(self):
         keys = pygame.key.get_pressed()
         self.car.accelerating = keys[pygame.K_UP]
         self.car.braking = keys[pygame.K_DOWN]

         self.car.steering = 0
         if keys[pygame.K_LEFT]:
             self.car.steering = -1
         if keys[pygame.K_RIGHT]:
             self.car.steering = 1

         for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 self.running = False
             if event.type == pygame.KEYDOWN:
                  if event.key == pygame.K_q:
                      self.running = False
                  if event.key == pygame.K_r: # Toggle ray display
                      self.show_rays = not self.show_rays
                  if event.key == pygame.K_l: # Load another map
                      map_name = input("Enter map filename to load: ")
                      if map_name:
                          self.load_map(map_name)
             if event.type == pygame.MOUSEBUTTONDOWN and self.game_over:
                 if self.restart_button.collidepoint(event.pos):
                     self.reset_game()

    def update(self, dt):
        if self.game_over:
             return

        self.car.update(dt)
        self.car.cast_rays()
        self.car.set_next_gate_info(self.gates, self.current_gate_index, self.gate_range)

        # Check collision with walls
        if self.car.check_collision_with_elements("walls"):
             #print("Wall Collision!")
             self.game_over = True

        # Check collision with current gate
        if self.current_gate_index < len(self.gates):
            if self.car.check_collision_with_elements("gates", self.current_gate_index):
                if not self.gate_passed:
                    #print(f"Passed gate {self.current_gate_index + 1}!")
                    self.gate_passed = True
                    self.score += 1
                    self.current_gate_index = (self.current_gate_index + 1) % len(self.gates)
            else:
                self.gate_passed = False  # Reset the flag when not colliding with gate

    def reset_game(self):
        """Reset the game state to start a new game"""
        self.car.pos = pygame.Vector2(self.start_pos)
        self.car.vel = pygame.Vector2(self.car.min_vel, self.car.min_vel)
        self.car.angle = 0
        self.game_over = False
        self.current_gate_index = 0
        self.gate_passed = False
        self.score = 0
        self.car.set_next_gate_info(self.gates, self.current_gate_index, self.gate_range)
        self.car.cast_rays()


    def draw_gate_rays(self):
        """Draw rays from the car to the start and end points of the next gate."""
        if self.gates and self.current_gate_index < len(self.gates):
            self.car.set_next_gate_info(self.gates, self.current_gate_index, 1)
            # we only draw the first gate ray for now the rest is for state
            for i in range(self.car.num_rays_to_gate):
                relative_angle_gates = self.car.relative_angle_gates[0,i]
                distance_gates = self.car.distance_gates[0,i]
                # Calculate end points of the rays
                ray_end = self.car.pos + pygame.Vector2(math.cos(self.car.angle + relative_angle_gates), math.sin(self.car.angle + relative_angle_gates)) * distance_gates
                # Draw the rays in a different color (using CYAN)
                pygame.draw.line(self.screen, CYAN, self.car.pos, ray_end, 2)



    def draw(self):
        self.screen.fill(BLACK)

        # Draw walls
        for wall in self.walls:
            pygame.draw.line(self.screen, RED, wall[0], wall[1], 3)

        # Draw gates with different colors for current and passed gates
        for i, gate in enumerate(self.gates):
            if i == self.current_gate_index:
                color = GREEN  # Current gate
            elif i < self.current_gate_index:
                color = (0, 100, 0)  # Passed gates (darker green)
            else:
                color = (100, 100, 100)  # Future gates (gray)
            pygame.draw.line(self.screen, color, gate[0], gate[1], 3)

        # Draw car
        self.car.draw(self.screen)

        # Draw rays (if enabled)
        if self.show_rays:
            self.car.draw_rays(self.screen)
            self.car.draw_rays_distances(self.screen, self.font)
            self.draw_gate_rays()
            
        # Draw game state info
        speed_text = self.font.render(f"Speed: {self.car.vel.length():.1f}", True, WHITE)
        self.screen.blit(speed_text, (10, 10))
        
        # Draw score and current gate in top right corner
        score_text = self.font.render(f"Score: {self.score} | Next Gate: {self.current_gate_index + 1}/{len(self.gates)}", True, WHITE)
        score_rect = score_text.get_rect(topright=(WIDTH - 10, 10))
        self.screen.blit(score_text, score_rect)

        # Draw next gate info
        if self.gates and self.current_gate_index < len(self.gates):
            self.car.set_next_gate_info(self.gates, self.current_gate_index, 1)
            distance_start, angle_start, distance_end, angle_end = self.car.distance_gates[0,0], self.car.relative_angle_gates[0,0], self.car.distance_gates[0,-1], self.car.relative_angle_gates[0,-1]
            angle_deg_start = math.degrees(angle_start)
            angle_deg_end = math.degrees(angle_end)
            gate_info = self.font.render(f"Next Gate: {distance_start:.1f} units, {angle_deg_start:.1f}°, and {distance_end:.1f} units, {angle_deg_end:.1f}°", True, WHITE)
            gate_rect = gate_info.get_rect(topright=(WIDTH - 10, 40))
            self.screen.blit(gate_info, gate_rect)

        if self.game_over:
            # Draw game over text
            go_text = self.font.render("GAME OVER (Collision)", True, RED)
            go_rect = go_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 30))
            self.screen.blit(go_text, go_rect)
            
            # Draw restart button
            pygame.draw.rect(self.screen, GREEN, self.restart_button)
            pygame.draw.rect(self.screen, WHITE, self.restart_button, 2)  # White border
            self.screen.blit(self.restart_text, self.restart_text_rect)

        pygame.display.flip()

    def run(self):
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0  # Delta time in seconds

            self.handle_input()
            self.update(dt * 60) # Scale dt to match fixed timestep physics if needed, or just use dt
            self.draw()

        pygame.quit() 