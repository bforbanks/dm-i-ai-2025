import torch
import random
from collections import deque
from typing import Optional
from models.rl.dqn.DQNModel import DQNModel

# REMINDER for how the DTOs look like:
# from pydantic import BaseModel
# from typing import Dict, Optional, List

   # Input
# class RaceCarPredictRequestDto(BaseModel):
#     did_crash: bool
#     elapsed_ticks: int
#     distance: float
#     velocity: Dict[str, float]  
#     # coordinates: Dict[str, int] # NOT USED IN THEIR REQUESTS (as of right now o.o)
#     sensors: Dict[str, Optional[float]]  

# class RaceCarPredictResponseDto(BaseModel):
#     actions: List[str]
#     # 'ACCELERATE'
#     # 'DECELERATE'
#     # 'STEER_LEFT'
#     # 'STEER_RIGHT'
#     # 'NOTHING''

class DQNAgent:
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 learning_rate=1e-4,
                 gamma=0.95,
                 batch_size: int = 64,
                 epsilon_start: float = 1,
                 epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.99999,
                 memory_size: int = 10000,
                 device: Optional[torch.device] = torch.device("cpu"),
                 dtype: Optional[torch.dtype] = torch.float32,
                 model_path: str = "models/rl/dqn/weights/dqn.pt",
                 reset_epsilon: bool = True
                ):
        # Set device and dtype
        self.device = device
        self.dtype = dtype
        print(f"Using device: {self.device}, dtype: {self.dtype}")
        # Initialize the DQN model
        self.model = DQNModel(input_dim=input_dim, output_dim=output_dim)
        self.model.to(self.device)

        self.model_path = model_path

        
        # Define hyperparameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if model_path:
            try:
                self.load(model_path)
                print(f"Loaded weights from: {model_path}")
            except (FileNotFoundError, OSError) as e:
                print(f"No saved model found at {model_path} — starting from scratch.")

        self.target_model = DQNModel(input_dim=input_dim, output_dim=output_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.to(self.device)
        self.target_model.eval()

        if reset_epsilon:
            self.epsilon = epsilon_start

        self.action_dict_idx_to_str = {
            0: "ACCELERATE",
            1: "DECELERATE",
            2: "STEER_LEFT",
            3: "STEER_RIGHT",
            4: "NOTHING",
        }
        self.action_dict_str_to_idx = {
            "ACCELERATE": 0,
            "DECELERATE": 1,
            "STEER_LEFT": 2,
            "STEER_RIGHT": 3,
            "NOTHING": 4
        }
        


    def save(self, filepath: str = "models/rl/dqn/weights/dqn_agent.pt"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath: str = "models/rl/dqn/weights/dqn_agent.pt"):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(self.device)
        self.epsilon = checkpoint.get('epsilon', self.epsilon)  # Restore epsilon if present

    def state_dict_to_tensor(self, state_dict: dict) -> torch.Tensor:
        """
        Converts a state dictionary (as the one we recieve from the API) to a tensor.
        
        The state tensor is indexed as follows:
            - 0: did_crash (bool)
            - 1: elapsed_ticks (int)
            - 2: distance (float)
            - 3: x-velocity (float)
            - 4: y-velocity (float)
            - 5-20: sensors (front, right_front, right_side, right_back, back, left_back, left_side, left_front, left_side_front, front_left_front, front_right_front, right_side_front, right_side_back, back_right_back, back_left_back, left_side_back)
        The sensors are expected to be in the order defined in the RaceCarPredictRequestDto.
        """
        # Very important to use the same order as the DQNModel expects!
        sensor_names = [
            "front", "right_front", "right_side", "right_back", "back",
            "left_back", "left_side", "left_front", "left_side_front",
            "front_left_front", "front_right_front", "right_side_front",
            "right_side_back", "back_right_back", "back_left_back",
            "left_side_back"
        ] 
        tensor = torch.zeros(21, dtype=self.dtype, device=self.device)

            # Normalization constants — adjust as needed – SHOULD NOT BE CHANGED AFTER TRAINING
        max_ticks = 3600         # typical episode length
        max_distance = 10**5      # estimated max track length
        max_velocity = 50      # estimated max x velocity
        max_drift = 1200.0        # for y velocity
        max_sensor = 1000.0       # assuming 0–100 for sensor distances
        tensor[0] = float(state_dict["did_crash"])
        tensor[1] = float(state_dict["elapsed_ticks"]) / max_ticks
        tensor[2] = float(state_dict["distance"]) / max_distance
        tensor[3] = float(state_dict["velocity"]["x"]) / max_velocity
        tensor[4] = float(state_dict["velocity"]["y"]) / max_drift

        for i, sensor in enumerate(sensor_names):
            tensor[5 + i] = (state_dict["sensors"].get(sensor, -1.0)) / max_sensor if state_dict["sensors"].get(sensor) is not None else -1.0
        
        return tensor


    def get_action(self, state: torch.Tensor) -> str:
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if random.random() < self.epsilon:
            return self.action_dict_idx_to_str[random.randint(0, self.model.output.out_features - 1)]
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=self.dtype, device=self.device)
            else:
                state = state.to(self.device, dtype=self.dtype)

            state = state.unsqueeze(0)
            q_values = self.model(state)
            string_action = self.action_dict_idx_to_str[q_values.argmax(dim=1).item()]
            assert isinstance(string_action, str), f"Expected string action, got {type(string_action)}"
            return string_action




    def learn(self):
        if len(self.memory) < self.batch_size:
            return  # Avoid training on too little data

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        actions = [self.action_dict_str_to_idx[action] for action in actions]
        # Convert to tensors
        states = torch.stack(states).to(self.device, dtype=self.dtype)
        next_states = torch.stack(next_states).to(self.device, dtype=self.dtype)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=self.dtype, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        self.model.train() # Set the model to training mode
        self.optimizer.zero_grad() # Clearing old gradients

        q_values = self.model(states).gather(1, actions).squeeze(1) 
        with torch.no_grad():
            # Get actions from the online model
            next_actions = self.model(next_states).argmax(dim=1, keepdim=True)

            # Get Q-values from the target model
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze(1)
            
            # Compute target Q-values
            target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values

        loss = torch.nn.functional.mse_loss(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

    
