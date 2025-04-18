# map_editor.py
import pygame
import json
import math
import os
from config import *
from classes.editor import Editor

FILENAME_DEFAULT = "map.json"
MAP_DIR = "maps" # Create a 'maps' folder next to the script

# --- Main Execution ---
if __name__ == "__main__":
    editor = Editor()
    editor.run()