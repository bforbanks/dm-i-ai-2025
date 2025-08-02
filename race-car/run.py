import pygame
import random
import importlib
from src.game.core import initialize_game_state, game_loop

'''
Set seed_value to None for random seed.
Within game_loop, change get_action() to your custom models prediction for local testing and training.
'''

# Just change this string to use different models! (remember capitalization)
model_name = "PredictionModel"  # "baseline", "RL.DQN.DQNModel", "playground", etc.

# Dynamic import
module = importlib.import_module(f'models.{model_name}')
class_name = model_name.split('.')[-1]  # Get last part and capitalize
MODEL = getattr(module, class_name)




if __name__ == '__main__':
    seed_value = 12345
    pygame.init()
    initialize_game_state("http://example.com/api/predict", seed_value)
    game_loop(verbose=False, model=MODEL()) # For pygame window
    pygame.quit()