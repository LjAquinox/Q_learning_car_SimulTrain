# map_editor.py
import pygame
import json
import math
import os
from config import *
from classes import Editor

MAP_DIR = "maps"
FILENAME_DEFAULT = "map.json"

# --- Main Execution ---
if __name__ == "__main__":
    editor = Editor()
    editor.run()