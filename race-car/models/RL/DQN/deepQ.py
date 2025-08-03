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
                 epsilon_start: float = 1.0,
                 epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 device: Optional[torch.device] = torch.device("cpu"),
                 dtype: Optional[torch.dtype] = torch.float32,
                 weight_path_prefix: Optional[str] = "models/rl/dqn/weights/",
                ):
        # Set device and dtype
        self.device = device
        self.dtype = dtype
        print(f"Using device: {self.device}, dtype: {self.dtype}")
        # Initialize the DQN model
        self.model = DQNModel(input_dim=input_dim, output_dim=output_dim)
        self.model.to(self.device)

        if weight_path_prefix:
            try:
                self.load(weight_path_prefix + "_model.pt", weight_path_prefix + "_optimizer.pt")
                print(f"Loaded weights from: {weight_path_prefix}_model.pt")
            except (FileNotFoundError, OSError) as e:
                print(f"No weights found at {weight_path_prefix} — starting from scratch.")

        # Define hyperparameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        



    def save(self, path_prefix: str):
        torch.save(self.model.state_dict(), f"{path_prefix}_model.pt")
        torch.save(self.optimizer.state_dict(), f"{path_prefix}_optimizer.pt")

    def load(self, model_path: str, optimizer_path: Optional[str] = None):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        if optimizer_path:
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))

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
        sensor_names = [
            "front", "right_front", "right_side", "right_back", "back",
            "left_back", "left_side", "left_front", "left_side_front",
            "front_left_front", "front_right_front", "right_side_front",
            "right_side_back", "back_right_back", "back_left_back",
            "left_side_back"
        ]
        tensor = torch.zeros(21, dtype=self.dtype, device=self.device)
        tensor[0] = float(state_dict["did_crash"])
        tensor[1] = float(state_dict["elapsed_ticks"])
        tensor[2] = float(state_dict["distance"])
        tensor[3] = float(state_dict["velocity"]["x"])
        tensor[4] = float(state_dict["velocity"]["y"])
        for i, sensor in enumerate(sensor_names):
            tensor[5 + i] = (state_dict["sensors"].get(sensor, -1.0)) if state_dict["sensors"].get(sensor) is not None else -1.0
        
        return tensor


    def get_action(self, state: torch.Tensor) -> int:
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if random.random() < self.epsilon:
            return random.randint(0, self.model.output.out_features - 1)
        with torch.no_grad():
            if not torch.is_tensor(state):
                state = torch.tensor(state, dtype=self.dtype, device=self.device)
            else:
                state = state.to(self.device, dtype=self.dtype)

            state = state.unsqueeze(0)
            q_values = self.model(state)
            return q_values.argmax(dim=1).item()




    def learn(self):
        if len(self.memory) < self.batch_size:
            return  # Avoid training on too little data

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

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
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values

        loss = torch.nn.functional.mse_loss(q_values, target_q_values)
        loss.backward()
        self.optimizer.step()

    
