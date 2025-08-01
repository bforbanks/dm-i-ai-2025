import pygame
import random
from src.game.core import initialize_game_state, game_loop
from models.baseline import BaselineModel as MODEL

'''
Set seed_value to None for random seed.
Within game_loop, change get_action() to your custom models prediction for local testing and training.
'''





if __name__ == '__main__':
    seed_value = 12345
    pygame.init()
    initialize_game_state("http://example.com/api/predict", seed_value)
    game_loop(verbose=True, model=MODEL()) # For pygame window
    pygame.quit()