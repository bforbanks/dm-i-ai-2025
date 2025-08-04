import torch
import pickle
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
    distance_list = []
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        done = False
        state = env.state_to_state_dict(env.STATE)  # To convert the game state to a dict
        state = agent.state_dict_to_tensor(state)  # Convert state to tensor
        stacked_state = agent.get_stacked_state(state)  # Get the stacked state
        
        # Uncomment if you want to track total reward
        total_reward = 0
        
        tick = 0
        
        while not done:
            tick += 1; total_ticks += 1
            # print(f"Episode: {episode}, Tick: {tick}, State: {state}")
            # print(env.SCREEN_WIDTH, env.SCREEN_HEIGHT, env.STATE.ego.y, env.STATE.ego.x, env.STATE.ego.velocity.x, env.STATE.ego.velocity.y)
            action = agent.return_action(state)
            # print(action, type(action))
            next_state, reward, done = env.step(action)

            next_state = agent.state_dict_to_tensor(next_state)
            stacked_next_state = agent.get_stacked_state(next_state)

            agent.memory.append((stacked_state, action, reward, stacked_next_state, done))


            if total_ticks % 1000 == 0:
                agent.target_model.load_state_dict(agent.model.state_dict())
                
            if len(agent.memory) > agent.batch_size and tick % 4 == 0:
                agent.learn()

            total_reward += reward
            if not done:
                distance = state[2]
            
            stacked_state = stacked_next_state


        reward_list.append(total_reward)
        tick_before_reset.append(tick)
        distance_list.append(distance*50000)
        
        if episode % 100 == 0:
            agent.save(agent.model_path)
            print("MODEL SAVED")
            N = min(100, len(reward_list))
            print(f"Episode {episode} completed. Average reward the last {N} episodes: {sum(reward_list[-N:]) / N:.2f}")
            print(f"Average ticks before reset: {sum(tick_before_reset[-N:]) / N:.2f}")
            print(f"Average distance: {sum(distance_list[-N:]) / N:.2f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print(f"Average loss: {sum(agent.loss_list[-N:]) / N:.4f}")
            with open("models/rl/dqn/diagnostics/losses.pkl", "wb") as f:
                pickle.dump(agent.loss_list, f)
    agent.save(agent.model_path)  # Save the model after training