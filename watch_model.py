import pygame
import numpy as np
import os
from classes.q_agent import QAgent
from classes.training_env import TrainingEnvironment
from config import *

class ModelViewer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Model Viewer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.big_font = pygame.font.SysFont(None, 36)
        self.running = True

        # Model selection UI state
        self.show_model_selection = True
        self.available_models = []
        self.current_model_index = 0
        self.model_selection_rect = pygame.Rect(WIDTH//2 - 200, HEIGHT//2 - 50, 400, 100)
        self.left_arrow_rect = pygame.Rect(WIDTH//2 - 250, HEIGHT//2 - 25, 50, 50)
        self.right_arrow_rect = pygame.Rect(WIDTH//2 + 200, HEIGHT//2 - 25, 50, 50)

        # Game state
        self.env = None
        self.agent = None
        self.state_size = None
        self.action_size = 7  # Same as in training

        # Load available models
        self.refresh_available_models()

    def refresh_available_models(self):
        """Refresh the list of available model files"""
        if not os.path.exists("models"):
            os.makedirs("models")
        self.available_models = [f for f in os.listdir("models") if f.endswith('.h5')]
        if not self.available_models:
            print("No models found in the models directory")
        self.current_model_index = 0

    def load_model(self, model_name):
        """Load a model and start the game"""
        if not model_name:
            return

        # Initialize environment
        self.env = TrainingEnvironment()
        
        # Get state size from the environment
        initial_state = self.env.reset()
        self.state_size = len(initial_state)
        
        # Create and load agent
        self.agent = QAgent(self.state_size, self.action_size)
        self.agent.load(os.path.join("models", model_name))
        
        # Set epsilon to 0 for deterministic behavior
        self.agent.epsilon = 0
        
        self.show_model_selection = False
        print(f"Loaded model: {model_name}")

    def draw_model_selection(self):
        """Draw the model selection UI"""
        # Draw semi-transparent background
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 128))
        self.screen.blit(s, (0, 0))

        # Draw selection box
        pygame.draw.rect(self.screen, WHITE, self.model_selection_rect, 2)
        
        # Draw arrows
        pygame.draw.polygon(self.screen, WHITE, [
            (self.left_arrow_rect.centerx - 10, self.left_arrow_rect.centery),
            (self.left_arrow_rect.centerx + 10, self.left_arrow_rect.centery - 10),
            (self.left_arrow_rect.centerx + 10, self.left_arrow_rect.centery + 10)
        ])
        pygame.draw.polygon(self.screen, WHITE, [
            (self.right_arrow_rect.centerx + 10, self.right_arrow_rect.centery),
            (self.right_arrow_rect.centerx - 10, self.right_arrow_rect.centery - 10),
            (self.right_arrow_rect.centerx - 10, self.right_arrow_rect.centery + 10)
        ])

        # Draw model name
        if self.available_models:
            model_name = self.available_models[self.current_model_index]
            model_text = self.big_font.render(model_name, True, WHITE)
            model_rect = model_text.get_rect(center=self.model_selection_rect.center)
            self.screen.blit(model_text, model_rect)
        else:
            no_models_text = self.big_font.render("No models found", True, WHITE)
            no_models_rect = no_models_text.get_rect(center=self.model_selection_rect.center)
            self.screen.blit(no_models_text, no_models_rect)

        # Draw instructions
        instr_text = self.font.render("Press ENTER to load, ESC to cancel", True, WHITE)
        instr_rect = instr_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 40))
        self.screen.blit(instr_text, instr_rect)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.show_model_selection:
                        self.running = False
                    else:
                        self.show_model_selection = True
                        self.env = None
                        self.agent = None
                elif event.key == pygame.K_RETURN and self.show_model_selection:
                    if self.available_models:
                        self.load_model(self.available_models[self.current_model_index])
                elif event.key == pygame.K_LEFT and self.show_model_selection:
                    if self.available_models:
                        self.current_model_index = (self.current_model_index - 1) % len(self.available_models)
                elif event.key == pygame.K_RIGHT and self.show_model_selection:
                    if self.available_models:
                        self.current_model_index = (self.current_model_index + 1) % len(self.available_models)

    def run(self):
        while self.running:
            self.handle_events()

            if self.show_model_selection:
                self.screen.fill(BLACK)
                self.draw_model_selection()
            else:
                # Get action from agent
                state = self.env.game.car.get_state()
                action = self.agent.act(state)
                
                # Step environment
                next_state, reward, done = self.env.step(action)
                
                # Update and draw game
                self.env.game.draw()
                
                # Reset if game over
                if done:
                    self.env.reset()

            pygame.display.flip()
            self.clock.tick(600)

        if self.env:
            self.env.close()
        pygame.quit()

if __name__ == "__main__":
    viewer = ModelViewer()
    viewer.run() 