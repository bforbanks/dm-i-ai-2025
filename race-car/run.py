#!/usr/bin/env python3
"""
Run the LaneShift game with the LaneShift model.

Usage (from project root):
    python race-car/run.py [--seed SEED] [--headless]
"""
import argparse
import importlib
import os
import sys

# Make race-car/ importable as the package root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

from src.game.core import initialize_game_state, game_loop

model_name = "LaneShift"
module = importlib.import_module(f"LaneShift.{model_name}")
MODEL = getattr(module, model_name)
expert_model = MODEL()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed",     type=int,  default=None)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    seed_value = args.seed
    verbose    = not args.headless

    if verbose:
        # Re-init display at full resolution for visible window
        pygame.display.set_mode((1600, 1200))
        pygame.display.set_caption("Race Car")

    initialize_game_state(api_url="http://localhost:9052", seed_value=seed_value)
    game_loop(verbose=verbose, model=expert_model)
    pygame.quit()
