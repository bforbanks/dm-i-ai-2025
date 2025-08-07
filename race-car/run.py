import pygame

# import random
# import torch  # Only needed for RL models
import importlib
from src.game.core import initialize_game_state, game_loop

# RL imports - only needed for RL models
# from src.game.rl_wrapper import RaceCarEnv
# from models.rl.dqn.train import train


"""
Set seed_value to None for random seed.
Within game_loop, change get_action() to your custom models prediction for local testing and training.
"""
# This is the full dotted path: module path + class name
# model_path = "rl.dqn.DQNModel.DQNModel"  # <== 'DQNModel' is the class name

# # Split the path into module and class name
# *module_parts, class_name = model_path.split(".")
# module_path = ".".join(module_parts)

# # Import module and get class
# module = importlib.import_module(f"models.{module_path}")

# MODEL = getattr(module, class_name)

# print(f"Using model: {MODEL.__name__}")


# Just change this string to use different models! (remember capitalization)
model_name = "LaneShift"  # Our improved expert system

# Dynamic import
module = importlib.import_module(f"models.{model_name}")
MODEL = getattr(module, model_name)

# Set to True if you want to use the RL environment
RL_ENV = False  # Expert system doesn't use RL environment

import random

if __name__ == "__main__":
    pygame.init()
    # if RL_ENV:
    #     show_visualization = False  # Change to False for headless training
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     dtype = torch.float32
    #     env = RaceCarEnv(
    #         api_url="http://example.com/api/predict",
    #         seed_value=seed_value,
    #         render=show_visualization,
    #     )
    #     agent = MODEL(input_dim=21, output_dim=5, device=device, dtype=dtype)
    #     train(agent=agent, env=env, episodes=100000)  # Train the agent
    # else:
    # Initialize the expert system
    expert_model = MODEL()
    print(f"Using model: {MODEL.__name__}")
    # expert_model = None
    seed_values = [978] #, 8110, 1701, 9283, 3949, 3223, 4516
    if not seed_values:
        runs = 20
        seed_values = [random.randint(0, 10000) for _ in range(runs)]
    runs = len(seed_values)
    for i, seed_value in enumerate(seed_values):
        # seed_value = random.randint(0, 10000)  # Set to None for random seed
        # seed_value = 2857
        # seed_value = None  # Set to None for random seed

        print(f"\n🏁 Starting Race {i + 1}/{runs} with OptimalExpertSystem")
        initialize_game_state(
            api_url="http://example.com/api/predict", seed_value=seed_value
        )
        game_loop(
            verbose=True, model=expert_model
        )  # For pygame window with expert system
    pygame.quit()
