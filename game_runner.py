# game_runner.py
import pygame
import json
import math
import os
from config import *
from classes import Game

MAP_DIR = "maps"
DEFAULT_MAP = "map.json"

# --- Main Execution ---
if __name__ == "__main__":
    game = Game()
    game.run()