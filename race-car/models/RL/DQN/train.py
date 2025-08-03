import torch

from models.rl.dqn.deepQ import DQNAgent
from models.rl.dqn.DQNModel import DQN

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
    env.reset()  # Initialize the environment
    tick = 0
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        done = False
        state = env.state_to_state_dict(env.STATE)  # Assuming this method exists to convert the game state to a dict
        state = agent.state_dict_to_tensor(state)  # Convert state to tensor
        
        # Uncomment if you want to track total reward
        # total_reward = 0
        print("Tick before reset:", tick)
        tick = 0
        while not done:
            tick += 1
            # print(f"Episode: {episode}, Tick: {tick}, State: {state}")
            if tick > 4601: # Temporary tick limit to catch potential bugs
                raise RuntimeError(f"Tick limit exceeded. Check your game logic. Tick: {tick}, Episode: {episode}")
            
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            next_state = agent.state_dict_to_tensor(next_state)

            agent.memory.append((state, action, reward, next_state, done))

            if len(agent.memory) > agent.batch_size and tick % 120 == 0:
                agent.learn()

            state = next_state
        if episode % 100 == 0:
            agent.save(f"dqn_agent_ep{episode}")
    agent.save("dqn_agent")  # Save the model after training