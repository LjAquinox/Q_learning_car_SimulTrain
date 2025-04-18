import pygame
import json
import os
from config import *
from classes.car import Car

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Car Game Runner")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.running = True

        self.walls = []
        self.gates = []
        self.start_pos = (WIDTH // 2, HEIGHT // 2) # Default if map has no start_pos

        self.load_map(DEFAULT_MAP) # Load the default map

        self.car = Car(self.start_pos[0], self.start_pos[1])
        self.show_rays = True # Toggle ray display with 'R'
        self.game_over = False
        
        # Restart button properties
        self.restart_button = pygame.Rect(WIDTH//2 - 50, HEIGHT//2 + 30, 100, 40)
        self.restart_text = self.font.render("Restart", True, WHITE)
        self.restart_text_rect = self.restart_text.get_rect(center=self.restart_button.center)

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
                     print(f"Warning: No start position found in map '{filename}'. Using default.")
                     # Use the default value already defined
                
                print(f"Map '{filename}' loaded successfully.")
                # Reset car position if map is loaded after init
                if hasattr(self, 'car'):
                    self.car.pos = pygame.Vector2(self.start_pos)
                    self.car.vel = pygame.Vector2(0, 0)
                    self.car.angle = 0 # Reset angle too
                    self.game_over = False

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
        self.car.cast_rays(self.walls)

        # Check collision
        if self.car.check_collision_with_elements(self.walls):
             print("Wall Collision!")
             self.game_over = True # Stop the game on collision for now
             # For AI, we might give a negative reward and reset instead
             # self.car.vel = pygame.Vector2(0,0) # Or just stop the car

        if self.car.check_collision_with_elements(self.gates):
            print("Gate collision!")
            #self.game_over = True # Stop the game on collision for now

    def reset_game(self):
        """Reset the game state to start a new game"""
        self.car.pos = pygame.Vector2(self.start_pos)
        self.car.vel = pygame.Vector2(0, 0)
        self.car.angle = 0
        self.game_over = False

    def draw(self):
        self.screen.fill(BLACK)

        # Draw walls
        for wall in self.walls:
            pygame.draw.line(self.screen, RED, wall[0], wall[1], 3)

        # Draw gates
        for gate in self.gates:
            pygame.draw.line(self.screen, GREEN, gate[0], gate[1], 3)

        # Draw car
        self.car.draw(self.screen)

        # Draw rays (if enabled)
        if self.show_rays:
            self.car.draw_rays(self.screen)
            self.car.draw_rays_distances(self.screen, self.font)
            
        # Draw game state info (optional)
        speed_text = self.font.render(f"Speed: {self.car.vel.length():.1f}", True, WHITE)
        self.screen.blit(speed_text, (10, 10))

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