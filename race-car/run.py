import pygame
import random
import torch
import importlib
from src.game.core import initialize_game_state, game_loop

# RL imports
from src.game.rl_wrapper import RaceCarEnv
from models.rl.dqn.train import train
from models.rl.dqn.deepQ import DQNAgent

'''
Set seed_value to None for random seed.
Within game_loop, change get_action() to your custom models prediction for local testing and training.
'''
# This is the full dotted path: module path + class name
model_path = "rl.dqn.deepQ.DQNAgent"  # <== 'DQNAgent' is the class name

# Split the path into module and class name
*module_parts, class_name = model_path.split(".")
module_path = ".".join(module_parts)

# Import module and get class
module = importlib.import_module(f"models.{module_path}")
MODEL = getattr(module, class_name)

print(f"Using model: {MODEL.__name__}")




# # Just change this string to use different models! (remember capitalization)
# model_name = "rl.dqn.DQNModel.DQN"  # "baseline", "rl.dqn.DQNModel", "playground", etc.

# # Dynamic import
# module = importlib.import_module(f'models.{model_name}')
# class_name = model_name.split('.')[-1]  # Get last part and capitalize
# MODEL = getattr(module, class_name)

# Set to True if you want to use the RL environment
RL_ENV = True


if __name__ == '__main__':
    seed_value = 12345
    pygame.init()
    if RL_ENV:
        show_visualization = False  # Change to False for headless training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        env = RaceCarEnv(api_url="http://example.com/api/predict", seed_value=seed_value, render=show_visualization)        
        agent = MODEL(input_dim=21, output_dim=5, device=device, dtype=dtype)
        train(agent=agent, env=env, episodes=10000)  # Train the agent
    else:
        
        initialize_game_state(api_url="http://example.com/api/predict", seed_value=seed_value)
        game_loop(verbose=True, model=False) # For pygame window
    pygame.quit()