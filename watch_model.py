import pygame
import numpy as np
import os
import re
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
        self.dropdown_rect = pygame.Rect(WIDTH//2 - 200, HEIGHT//2 - 100, 400, 40)
        self.text_input_rect = pygame.Rect(WIDTH//2 - 200, HEIGHT//2 - 150, 400, 40)
        self.text_input_active = False
        self.text_input = ""
        self.show_dropdown = False
        self.dropdown_items = []
        self.dropdown_scroll = 0
        self.max_dropdown_items = 5
        self.hover_index = -1

        # Game state
        self.env = None
        self.agent = None
        self.state_size = None
        self.action_size = 6  # Same as in training can be set automatically from the environment/model loaded

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
        self.dropdown_items = self.available_models

    def search_models(self, search_text):
        """Search for models containing the given number in their filename"""
        if not search_text:
            return self.available_models
        
        # Try to find numbers in the search text
        numbers = re.findall(r'\d+', search_text)
        if not numbers:
            return [model for model in self.available_models if search_text.lower() in model.lower()]
        
        # Search for models containing any of the found numbers
        matching_models = []
        for model in self.available_models:
            model_numbers = re.findall(r'\d+', model)
            if any(num in model_numbers for num in numbers):
                matching_models.append(model)
        
        return matching_models

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

        # Draw text input box
        pygame.draw.rect(self.screen, WHITE, self.text_input_rect, 2)
        text_surface = self.font.render(self.text_input, True, WHITE)
        self.screen.blit(text_surface, (self.text_input_rect.x + 5, self.text_input_rect.y + 5))

        # Draw dropdown box
        pygame.draw.rect(self.screen, WHITE, self.dropdown_rect, 2)
        dropdown_text = "Select Model" if not self.dropdown_items else self.dropdown_items[self.current_model_index]
        dropdown_surface = self.font.render(dropdown_text, True, WHITE)
        self.screen.blit(dropdown_surface, (self.dropdown_rect.x + 5, self.dropdown_rect.y + 5))

        # Draw dropdown arrow
        arrow_points = [
            (self.dropdown_rect.right - 20, self.dropdown_rect.centery - 5),
            (self.dropdown_rect.right - 10, self.dropdown_rect.centery + 5),
            (self.dropdown_rect.right - 30, self.dropdown_rect.centery + 5)
        ]
        pygame.draw.polygon(self.screen, WHITE, arrow_points)

        # Calculate dropdown list height
        dropdown_list_height = 0
        if self.show_dropdown and self.dropdown_items:
            dropdown_list_height = min(len(self.dropdown_items), self.max_dropdown_items) * 30

        # Draw dropdown list if active
        if self.show_dropdown and self.dropdown_items:
            dropdown_list_rect = pygame.Rect(
                self.dropdown_rect.x,
                self.dropdown_rect.bottom,
                self.dropdown_rect.width,
                dropdown_list_height
            )
            pygame.draw.rect(self.screen, BLACK, dropdown_list_rect)
            pygame.draw.rect(self.screen, WHITE, dropdown_list_rect, 1)

            start_idx = self.dropdown_scroll
            end_idx = min(start_idx + self.max_dropdown_items, len(self.dropdown_items))
            
            for i, model in enumerate(self.dropdown_items[start_idx:end_idx]):
                item_rect = pygame.Rect(
                    dropdown_list_rect.x,
                    dropdown_list_rect.y + i * 30,
                    dropdown_list_rect.width,
                    30
                )
                # Highlight current selection and hover
                if i + start_idx == self.current_model_index:
                    pygame.draw.rect(self.screen, (100, 100, 100), item_rect)
                elif i + start_idx == self.hover_index:
                    pygame.draw.rect(self.screen, (50, 50, 50), item_rect)
                text_surface = self.font.render(model, True, WHITE)
                self.screen.blit(text_surface, (item_rect.x + 5, item_rect.y + 5))

            # Draw scrollbar if needed
            if len(self.dropdown_items) > self.max_dropdown_items:
                scrollbar_height = (self.max_dropdown_items / len(self.dropdown_items)) * dropdown_list_rect.height
                scrollbar_y = dropdown_list_rect.y + (self.dropdown_scroll / len(self.dropdown_items)) * dropdown_list_rect.height
                scrollbar_rect = pygame.Rect(
                    dropdown_list_rect.right - 10,
                    scrollbar_y,
                    10,
                    scrollbar_height
                )
                pygame.draw.rect(self.screen, WHITE, scrollbar_rect)

        # Draw instructions with adjusted position based on dropdown state
        instr_text = self.font.render("Press ENTER to load, ESC to cancel", True, WHITE)
        instr_y = HEIGHT//2 + 40 + (dropdown_list_height if self.show_dropdown else 0)
        instr_rect = instr_text.get_rect(center=(WIDTH//2, instr_y))
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
                    if self.dropdown_items:
                        self.load_model(self.dropdown_items[self.current_model_index])
                elif event.key == pygame.K_BACKSPACE and self.text_input_active:
                    self.text_input = self.text_input[:-1]
                    self.dropdown_items = self.search_models(self.text_input)
                    if self.dropdown_items:
                        self.current_model_index = 0
                elif event.key == pygame.K_TAB:
                    self.text_input_active = not self.text_input_active
                    self.show_dropdown = not self.show_dropdown
                elif event.key == pygame.K_UP and self.show_dropdown:
                    if self.current_model_index > 0:
                        self.current_model_index -= 1
                        if self.current_model_index < self.dropdown_scroll:
                            self.dropdown_scroll = max(0, self.dropdown_scroll - 1)
                elif event.key == pygame.K_DOWN and self.show_dropdown:
                    if self.current_model_index < len(self.dropdown_items) - 1:
                        self.current_model_index += 1
                        if self.current_model_index >= self.dropdown_scroll + self.max_dropdown_items:
                            self.dropdown_scroll = min(len(self.dropdown_items) - self.max_dropdown_items, self.dropdown_scroll + 1)
                elif self.text_input_active and event.unicode.isprintable():
                    self.text_input += event.unicode
                    self.dropdown_items = self.search_models(self.text_input)
                    if self.dropdown_items:
                        self.current_model_index = 0
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.show_model_selection:
                    # Check if clicked on text input
                    if self.text_input_rect.collidepoint(event.pos):
                        self.text_input_active = True
                        self.show_dropdown = False
                    # Check if clicked on dropdown
                    elif self.dropdown_rect.collidepoint(event.pos):
                        self.show_dropdown = not self.show_dropdown
                        self.text_input_active = False
                    # Check if clicked on dropdown items
                    elif self.show_dropdown and self.dropdown_items:
                        dropdown_list_rect = pygame.Rect(
                            self.dropdown_rect.x,
                            self.dropdown_rect.bottom,
                            self.dropdown_rect.width,
                            min(len(self.dropdown_items), self.max_dropdown_items) * 30
                        )
                        if dropdown_list_rect.collidepoint(event.pos):
                            relative_y = event.pos[1] - dropdown_list_rect.y
                            item_index = relative_y // 30 + self.dropdown_scroll
                            if 0 <= item_index < len(self.dropdown_items):
                                self.current_model_index = item_index
                                self.show_dropdown = False
                    else:
                        self.text_input_active = False
                        self.show_dropdown = False
            if event.type == pygame.MOUSEMOTION:
                if self.show_dropdown and self.dropdown_items:
                    dropdown_list_rect = pygame.Rect(
                        self.dropdown_rect.x,
                        self.dropdown_rect.bottom,
                        self.dropdown_rect.width,
                        min(len(self.dropdown_items), self.max_dropdown_items) * 30
                    )
                    if dropdown_list_rect.collidepoint(event.pos):
                        relative_y = event.pos[1] - dropdown_list_rect.y
                        self.hover_index = relative_y // 30 + self.dropdown_scroll
                        if self.hover_index >= len(self.dropdown_items):
                            self.hover_index = -1
                    else:
                        self.hover_index = -1

    def run(self):
        while self.running:
            self.handle_events()

            if self.show_model_selection:
                self.screen.fill(BLACK)
                self.draw_model_selection()
            else:
                # Get action from agent
                self.env.game.car.set_next_gate_info(self.env.game.gates, self.env.game.current_gate_index, self.env.game.gate_range)
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
            self.clock.tick(60)

        if self.env:
            self.env.close()
        pygame.quit()

if __name__ == "__main__":
    viewer = ModelViewer()
    viewer.run() 