import torch

from models.rl.dqn.DQNAgent import DQNAgent

from tqdm import tqdm
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


    



def train(agent: DQNAgent, env, episodes: int = 1000):
    env.reset()  # Initialize the environment
    tick = 0
    total_ticks = 0
    tick_before_reset = []
    reward_list = []
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        done = False
        state = env.state_to_state_dict(env.STATE)  # Assuming this method exists to convert the game state to a dict
        state = agent.state_dict_to_tensor(state)  # Convert state to tensor
        
        # Uncomment if you want to track total reward
        total_reward = 0
        
        tick = 0
        
        while not done:
            tick += 1; total_ticks += 1
            # print(f"Episode: {episode}, Tick: {tick}, State: {state}")
            if tick > 4601: # Temporary tick limit to catch potential bugs
                raise RuntimeError(f"Tick limit exceeded. Check your game logic. Tick: {tick}, Episode: {episode}")
            
            action = agent.get_action(state)
            # print(action, type(action))
            next_state, reward, done = env.step(action)

            next_state = agent.state_dict_to_tensor(next_state)

            agent.memory.append((state, action, reward, next_state, done))

            if len(agent.memory) > agent.batch_size and tick % 4 == 0:
                agent.learn()

            total_reward += reward
            state = next_state
        reward_list.append(total_reward)
        tick_before_reset.append(tick)
        if episode % 100 == 0:
            agent.save(agent.model_path)
            print("MODEL SAVED")
            N = min(100, len(reward_list))
            print(f"Episode {episode} completed. Average reward the last {N} episodes: {sum(reward_list[-N:]) / N:.2f}")
            print(f"Average ticks before reset: {sum(tick_before_reset[-N:]) / N:.2f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
    agent.save(agent.model_path)  # Save the model after training