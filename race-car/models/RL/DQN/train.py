import torch

from models.rl.dqn.deepQ import DQNAgent
from models.rl.dqn.DQNModel import DQN
from src.game.core import initialize_game_state, game_loop

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

    #              input_dim: int,
    #              output_dim: int,
    #              learning_rate=1e-4,
    #              gamma=0.95,
    #              batch_size: int = 64,
    #              epsilon_start: float = 1.0,
    #              epsilon_min: float = 0.1,
    #              epsilon_decay: float = 0.995,
    #              memory_size: int = 10000,
    #              device: Optional[torch.device] = torch.device("cpu"),
    #              dtype: Optional[torch.dtype] = torch.float32,
    #              weight_path_prefix: Optional[str] = None,


agent = DQNAgent(input_dim=21, output_dim=5, device=device, dtype=dtype)


def train(agent: DQNAgent, env, episodes: int = 1000):
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(torch.tensor(state, dtype=dtype, device=device))
            next_state, reward, done = env.step(action) # TODO: This does maybe not work as intended
            
            agent.memory.append((state, action, reward, next_state, done))

            if episode % 10 == 0:
                agent.train()

            state = next_state
    
    agent.save("dqn_agent")  # Save the model after training
    