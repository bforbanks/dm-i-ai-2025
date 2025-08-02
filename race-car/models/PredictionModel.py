"""THIS IS NOT A MODEL, IT IS A CLASS USED TO PREDICT THE POSITIONS 
AND VELOCITIES OF OTHER CARS"""

from src.game.core import GameState
import numpy as np
import time

class lane_state:
    max_vel = 200 # assumed cars won't go faster than 200 but who knows

    def __init__(self, lane: int, resolution: tuple[float, float]):
        x_resolution, vel_resolution = resolution

        self.lane: int = lane
        self.existance_prob: float # There is a probability that a car is despawned, although it's usually very small (only if >1 car despawns at the same time).
        self.velocity_bin_dividers = np.arange(0, self.max_vel, vel_resolution) # lower bound included, upper bound excluded (except last one)
        vel_bins = len(self.velocity_bin_dividers)-1 # velocity-values ranging from 0 to max_vel
        self.velocity: np.ndarray # shape (vel_bins,)
        self.x_bin_dividers = np.arange(0, 3600, x_resolution) # lower bound included, upper bound excluded (except last one)
        x_bins = len(self.x_bin_dividers)-1 # x-values ranging from -1000 to 2600
        self.position: np.ndarray # shape (x_bins,)

class PredictionModel:
    # 0 degrees is up, 90 degrees is right
    all_sensors = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]
    sdict = {90: "front", 135: "right_front", 180: "right_side", 225: "right_back", 270: "back", 315: "left_back", 0: "left_side", 45: "left_front", 22.5: "left_side_front", 67.5: "front_left_front", 112.5: "front_right_front", 157.5: "right_side_front", 202.5: "right_side_back", 247.5: "back_right_back", 292.5: "back_left_back", 337.5: "left_side_back"}

    def __init__(self, x_resolution: float = 0.1, vel_resolution: float = 0.01):
        resolution = (x_resolution, vel_resolution)
        self.lanes = [lane_state(i, resolution) for i in range(1, 6)]
        self.uncertain_sensors = [67.5, 112.5, 247.5, 292.5]
        self.fully_initialized = False

    def _init_with_state(self, state: GameState):
        """Usually the state is not available when initializing models, so we need a second
        initialization"""
        if state["elapsed_ticks"] > 1:
            self.available_sensors = [sensor for sensor in self.all_sensors if sensor not in self.uncertain_sensors and state["sensors"][self.sdict[sensor]]]
            self.fully_initialized = True

    def _update_sensors(self, state: dict):
        if not self.fully_initialized:
            self._init_with_state(state)
        for sensor in self.uncertain_sensors:
            if state["sensors"][self.sdict[sensor]]:
                self.available_sensors.append(sensor)
                self.uncertain_sensors.remove(sensor)


    def _update_lanes(self, state: dict):
        pass

    def update(self, state: dict):
        print("--------------------------------")
        print("STATE:\n", state)
        print("AVAILABLE SENSORS:\n", self.available_sensors) if hasattr(self, "available_sensors") else print("AVAILABLE SENSORS: None")
        print("--------------------------------")
        time.sleep(5)
        self._update_sensors(state)
        self._update_lanes(state)

    def return_action(self, state):
        """This function is to show how to use this class as well as for testing. 
            Really you only need to run .update_lanes()
            every tick. Later, I might implement solution if ticks are skipped. 
            Of course, replace self with object name"""
        predictions = self.update(state)
        
        action = "NOTHING" # You would implement actual decision logic here

        return action

