import pygame
import json
import math
import os
from config import *

# --- Editor Class ---
class Editor:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Map Editor - Mode: WALL (W/G/S/L/Z/RMB/Q)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        self.big_font = pygame.font.SysFont(None, 36)

        self.elements = {"walls": [], "gates": [], "start_pos": None} # Structure for JSON
        self.history = [] # For simple undo
        self.mode = "wall"  # 'wall', 'gate', 'start'
        self.drawing_wall_start = None # Stores the first point of the wall being drawn
        self.drawing_gate_start = None # Stores the first point of the gate being drawn
        self.running = True

        # Map selection UI state
        self.show_map_selection = False
        self.available_maps = []
        self.current_map_index = 0
        self.map_selection_rect = pygame.Rect(WIDTH//2 - 200, HEIGHT//2 - 50, 400, 100)
        self.left_arrow_rect = pygame.Rect(WIDTH//2 - 250, HEIGHT//2 - 25, 50, 50)
        self.right_arrow_rect = pygame.Rect(WIDTH//2 + 200, HEIGHT//2 - 25, 50, 50)

        # Save UI state
        self.show_save_ui = False
        self.save_input_text = ""
        self.save_input_rect = pygame.Rect(WIDTH//2 - 200, HEIGHT//2 - 25, 400, 50)
        self.save_input_active = False

        if not os.path.exists(MAP_DIR):
            os.makedirs(MAP_DIR)
        
        self.refresh_available_maps()

    def refresh_available_maps(self):
        """Refresh the list of available map files"""
        self.available_maps = [f for f in os.listdir(MAP_DIR) if f.endswith('.json')]
        if not self.available_maps:
            self.available_maps = [FILENAME_DEFAULT]
        self.current_map_index = 0

    def save_state(self):
        # Saves a deep copy of the current state for undo
        import copy
        self.history.append(copy.deepcopy(self.elements))
        if len(self.history) > 100: # Limits the history
             self.history.pop(0)

    def undo(self):
        if self.history:
            self.elements = self.history.pop()
            self.drawing_wall_start = None # Cancels the wall being drawn
            print("Undo successful.")
        else:
            print("No more actions to undo.")

    def snap_to_grid(self, pos):
        if GRID_SIZE > 0:
            return (round(pos[0] / GRID_SIZE) * GRID_SIZE,
                    round(pos[1] / GRID_SIZE) * GRID_SIZE)
        return pos

    def get_element_at_pos(self, pos, tolerance=10):
        closest_element = None
        min_dist_sq = tolerance**2

        # Find the closest wall (testing distance to segment)
        for i, wall in enumerate(self.elements["walls"]):
            p1 = pygame.Vector2(wall[0])
            p2 = pygame.Vector2(wall[1])
            p = pygame.Vector2(pos)
            
            # Simple bounding box check first for performance
            if not (min(p1.x, p2.x) - tolerance <= p.x <= max(p1.x, p2.x) + tolerance and \
                    min(p1.y, p2.y) - tolerance <= p.y <= max(p1.y, p2.y) + tolerance):
                continue

            # Distance point to line segment
            l2 = (p1 - p2).length_squared()
            if l2 == 0: # Segment is a point
                dist_sq = (p - p1).length_squared()
            else:
                t = max(0, min(1, (p - p1).dot(p2 - p1) / l2))
                projection = p1 + t * (p2 - p1)
                dist_sq = (p - projection).length_squared()

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_element = ("wall", i)

        # Find the closest gate
        for i, gate in enumerate(self.elements["gates"]):
            p1 = pygame.Vector2(gate[0])
            p2 = pygame.Vector2(gate[1])
            p = pygame.Vector2(pos)
            
            # Simple bounding box check first for performance
            if not (min(p1.x, p2.x) - tolerance <= p.x <= max(p1.x, p2.x) + tolerance and \
                    min(p1.y, p2.y) - tolerance <= p.y <= max(p1.y, p2.y) + tolerance):
                continue

            # Distance point to line segment
            l2 = (p1 - p2).length_squared()
            if l2 == 0: # Segment is a point
                dist_sq = (p - p1).length_squared()
            else:
                t = max(0, min(1, (p - p1).dot(p2 - p1) / l2))
                projection = p1 + t * (p2 - p1)
                dist_sq = (p - projection).length_squared()

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_element = ("gate", i)
                 
        # Find the start point
        if self.elements["start_pos"]:
             dist_sq = (pygame.Vector2(pos) - pygame.Vector2(self.elements["start_pos"])).length_squared()
             if dist_sq < min_dist_sq:
                 min_dist_sq = dist_sq
                 closest_element = ("start", 0) # Index 0 because there's only one

        return closest_element

    def remove_element(self, element_info):
        if not element_info:
            return False

        elem_type, index = element_info
        self.save_state() # Save state before removing

        if elem_type == "wall":
            del self.elements["walls"][index]
            print(f"Removed wall {index}")
            return True
        elif elem_type == "gate":
            del self.elements["gates"][index]
            print(f"Removed gate {index}")
            return True
        elif elem_type == "start":
            self.elements["start_pos"] = None
            print("Removed start position")
            return True
        return False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                elif self.show_save_ui:
                    # Handle save UI input
                    if event.key == pygame.K_ESCAPE:
                        self.show_save_ui = False
                        self.save_input_active = False
                    elif event.key == pygame.K_RETURN:
                        if self.save_input_text:
                            self.save_map(self.save_input_text)
                        self.show_save_ui = False
                        self.save_input_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.save_input_text = self.save_input_text[:-1]
                    elif len(self.save_input_text) < 20:  # Limit filename length
                        if event.unicode.isalnum() or event.unicode in ['.', '_', '-']:
                            self.save_input_text += event.unicode
                elif self.show_map_selection:
                    # Handle map selection UI
                    if event.key == pygame.K_ESCAPE:
                        self.show_map_selection = False
                    elif event.key == pygame.K_RETURN:
                        self.load_map(self.available_maps[self.current_map_index])
                        self.show_map_selection = False
                    elif event.key == pygame.K_LEFT:
                        self.current_map_index = (self.current_map_index - 1) % len(self.available_maps)
                    elif event.key == pygame.K_RIGHT:
                        self.current_map_index = (self.current_map_index + 1) % len(self.available_maps)
                else:
                    # Handle normal editor mode
                    if event.key == pygame.K_w:
                        self.mode = "wall"
                        self.drawing_wall_start = None
                        self.drawing_gate_start = None
                        pygame.display.set_caption(f"Map Editor - Mode: {self.mode.upper()} (W/G/P/S/L/Z/RMB/Q)")
                        print("Mode: Add Walls")
                    elif event.key == pygame.K_g:
                        self.mode = "gate"
                        self.drawing_wall_start = None
                        self.drawing_gate_start = None
                        pygame.display.set_caption(f"Map Editor - Mode: {self.mode.upper()} (W/G/P/S/L/Z/RMB/Q)")
                        print("Mode: Add Gates")
                    elif event.key == pygame.K_p: # 'P' for Start Position
                        self.mode = "start"
                        self.drawing_wall_start = None
                        self.drawing_gate_start = None
                        pygame.display.set_caption(f"Map Editor - Mode: {self.mode.upper()} (W/G/P/S/L/Z/RMB/Q)")
                        print("Mode: Set Start Position")
                    elif event.key == pygame.K_s:
                        self.show_save_ui = True
                        self.save_input_text = ""
                        self.save_input_active = True
                    elif event.key == pygame.K_l:
                        self.show_map_selection = True
                        self.refresh_available_maps()
                    elif event.key == pygame.K_z: # Undo
                        self.undo()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = self.snap_to_grid(event.pos)

                if self.show_map_selection:
                    if self.left_arrow_rect.collidepoint(event.pos):
                        self.current_map_index = (self.current_map_index - 1) % len(self.available_maps)
                    elif self.right_arrow_rect.collidepoint(event.pos):
                        self.current_map_index = (self.current_map_index + 1) % len(self.available_maps)
                    elif self.map_selection_rect.collidepoint(event.pos):
                        self.load_map(self.available_maps[self.current_map_index])
                        self.show_map_selection = False
                    continue
                elif self.show_save_ui:
                    if self.save_input_rect.collidepoint(event.pos):
                        self.save_input_active = True
                    else:
                        self.save_input_active = False
                    continue

                if event.button == 1: # Left Click
                    self.save_state() # Save state before adding
                    if self.mode == "wall":
                        if self.drawing_wall_start is None:
                            self.drawing_wall_start = pos
                            print(f"Wall start: {pos}")
                        else:
                            wall = [self.drawing_wall_start, pos]
                            # Avoid walls with zero length
                            if wall[0] != wall[1]:
                                self.elements["walls"].append(wall)
                                print(f"Added wall: {wall}")
                            else:
                                print("Wall has zero length, not added.")
                            self.drawing_wall_start = None # Reset for next wall
                    elif self.mode == "gate":
                        if self.drawing_gate_start is None:
                            self.drawing_gate_start = pos
                            print(f"Gate start: {pos}")
                        else:
                            gate = [self.drawing_gate_start, pos]
                            # Avoid gates with zero length
                            if gate[0] != gate[1]:
                                self.elements["gates"].append(gate)
                                print(f"Added gate: {gate}")
                            else:
                                print("Gate has zero length, not added.")
                            self.drawing_gate_start = None # Reset for next gate
                    elif self.mode == "start":
                        self.elements["start_pos"] = list(pos)
                        print(f"Set start position: {pos}")

                elif event.button == 3: # Right Click
                    element_to_remove = self.get_element_at_pos(event.pos)
                    if element_to_remove:
                        self.remove_element(element_to_remove)
                    else:
                         print("Nothing close enough to remove.")

    def draw_grid(self):
        if GRID_SIZE > 0:
            for x in range(0, WIDTH, GRID_SIZE):
                pygame.draw.line(self.screen, (50, 50, 50), (x, 0), (x, HEIGHT))
            for y in range(0, HEIGHT, GRID_SIZE):
                pygame.draw.line(self.screen, (50, 50, 50), (0, y), (WIDTH, y))

    def draw_elements(self):
        # Draw walls
        for wall in self.elements["walls"]:
            pygame.draw.line(self.screen, RED, wall[0], wall[1], 3)

        # Draw gates
        for gate in self.elements["gates"]:
            pygame.draw.line(self.screen, GREEN, gate[0], gate[1], 3)

        # Draw start position
        if self.elements["start_pos"]:
             pygame.draw.circle(self.screen, YELLOW, self.elements["start_pos"], 10)
             # Optional: Draw a small line indicating default angle (e.g., pointing right)
             start_vec = pygame.Vector2(self.elements["start_pos"])
             end_vec = start_vec + pygame.Vector2(15, 0) 
             pygame.draw.line(self.screen, YELLOW, start_vec, end_vec, 2)

        # Draw wall starting point if currently drawing
        if self.drawing_wall_start:
            pygame.draw.circle(self.screen, BLUE, self.drawing_wall_start, 5)
            # Draw line to current mouse position
            current_pos = self.snap_to_grid(pygame.mouse.get_pos())
            pygame.draw.line(self.screen, BLUE, self.drawing_wall_start, current_pos, 1)

        # Draw gate starting point if currently drawing a gate
        if self.drawing_gate_start:
            pygame.draw.circle(self.screen, GREEN, self.drawing_gate_start, 5)
            # Draw line to current mouse position
            current_pos = self.snap_to_grid(pygame.mouse.get_pos())
            pygame.draw.line(self.screen, GREEN, self.drawing_gate_start, current_pos, 1)

    def draw_map_selection(self):
        """Draw the map selection UI"""
        # Draw semi-transparent background
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 128))
        self.screen.blit(s, (0, 0))

        # Draw selection box
        pygame.draw.rect(self.screen, WHITE, self.map_selection_rect, 2)
        
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

        # Draw map name
        map_name = self.available_maps[self.current_map_index]
        map_text = self.big_font.render(map_name, True, WHITE)
        map_rect = map_text.get_rect(center=self.map_selection_rect.center)
        self.screen.blit(map_text, map_rect)

        # Draw instructions
        instr_text = self.font.render("Press ENTER to load, ESC to cancel", True, WHITE)
        instr_rect = instr_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 40))
        self.screen.blit(instr_text, instr_rect)

    def draw_save_ui(self):
        """Draw the save file UI"""
        # Draw semi-transparent background
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        s.fill((0, 0, 0, 128))
        self.screen.blit(s, (0, 0))

        # Draw input box
        pygame.draw.rect(self.screen, WHITE, self.save_input_rect, 2)
        if self.save_input_active:
            pygame.draw.rect(self.screen, (100, 100, 255), self.save_input_rect, 2)

        # Draw input text
        input_text = self.big_font.render(self.save_input_text, True, WHITE)
        input_rect = input_text.get_rect(midleft=(self.save_input_rect.left + 10, self.save_input_rect.centery))
        self.screen.blit(input_text, input_rect)

        # Draw instructions
        instr_text = self.font.render("Enter filename and press ENTER to save, ESC to cancel", True, WHITE)
        instr_rect = instr_text.get_rect(center=(WIDTH//2, HEIGHT//2 + 40))
        self.screen.blit(instr_text, instr_rect)

    def draw_ui(self):
        if self.show_map_selection:
            self.draw_map_selection()
        elif self.show_save_ui:
            self.draw_save_ui()
        else:
            mode_text = self.font.render(f"Mode: {self.mode.upper()} | (W)all/(G)ate/(P)os | (S)ave/(L)oad | (Z)Undo | RMB:Del | Q:Quit", True, WHITE)
            self.screen.blit(mode_text, (10, 10))

    def run(self):
        while self.running:
            self.handle_events()

            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_elements()
            self.draw_ui()

            pygame.display.flip()
            self.clock.tick(60) # FPS

        pygame.quit()

    def save_map(self, filename=None):
        if filename is None:
            self.show_save_ui = True
            self.save_input_text = ""
            self.save_input_active = True
            return

        if not filename.endswith('.json'):
            filename += '.json'
            
        filepath = os.path.join(MAP_DIR, filename)
        try:
            # Ensure start_pos is set before saving
            if self.elements["start_pos"] is None:
                print("Error: Start position must be set before saving the map.")
                return 
                
            with open(filepath, 'w') as f:
                json.dump(self.elements, f, indent=4)
            print(f"Map saved to {filepath}")
            self.refresh_available_maps()  # Refresh available maps after saving
        except Exception as e:
            print(f"Error saving map: {e}")

    def load_map(self, filename=None):
        if filename is None:
            self.show_map_selection = True
            self.refresh_available_maps()
            return

        filepath = os.path.join(MAP_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                self.save_state() # Save current state before loading
                loaded_data = json.load(f)
                # Basic validation
                if "walls" in loaded_data and "gates" in loaded_data and "start_pos" in loaded_data:
                    self.elements = loaded_data
                    self.drawing_wall_start = None # Reset drawing state
                    print(f"Map loaded from {filepath}")
                else:
                    print("Error: Invalid map format in file.")
        except FileNotFoundError:
            print(f"Error: File not found '{filepath}'")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{filepath}'")
        except Exception as e:
            print(f"Error loading map: {e}")
