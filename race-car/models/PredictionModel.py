"""THIS IS NOT A MODEL, IT IS A CLASS USED TO PREDICT THE POSITIONS 
AND VELOCITIES OF OTHER CARS"""

from src.game.core import GameState
import numpy as np
import time

class lane_state:
    max_vel = 200 # assumed cars won't go faster than 200 but who knows
    behind_spawn = -980+1000
    front_spawn = 2220+1000

    def __init__(self, lane: int, x_resolution: float, vel_resolution: float):
        self.x_resolution, self.vel_resolution = x_resolution, vel_resolution

        self.lane: int = lane
        self.spotted: bool = False
        
        # 3 things determine the state of a lane. Is there a car, car position and car velocity. 
        self.no_car: float = 1 # There is a probability that a lane has no car.
        self.new_no_car: float = 0 

        self.velocity_bin_dividers = np.arange(0, self.max_vel, vel_resolution) # lower bound included, upper bound excluded (except last one)
        self.vel_bins = len(self.velocity_bin_dividers)-1 # velocity-values ranging from 0 to max_vel
        self.velocity = np.zeros(self.vel_bins)
        self.new_velocity = np.zeros(self.vel_bins) 
        
        self.x_bin_dividers = np.arange(0, 3600, x_resolution) # lower bound included, upper bound excluded (except last one)
        self.x_bins = len(self.x_bin_dividers)-1 # x-values ranging from -1000 to 2600
        self.position = np.zeros(self.x_bins)
        self.new_position = np.zeros(self.x_bins)

        # for ease of use, we will calculate mean bin (average of the bounds)
        self.x_bin_means = 0.5 * (self.x_bin_dividers[:-1] + self.x_bin_dividers[1:])
        self.vel_bin_means = 0.5 * (self.velocity_bin_dividers[:-1] + self.velocity_bin_dividers[1:])
        
        # as well as the bins of the potential spawnpoints of new cars
        self.behind_spawn_bin = self.behind_spawn//self.x_resolution
        self.front_spawn_bin = self.front_spawn//self.x_resolution

class PredictionModel:
    # 0 degrees is up, 90 degrees is right
    all_sensors = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]
    sdict = {90: "front", 135: "right_front", 180: "right_side", 225: "right_back", 270: "back", 315: "left_back", 0: "left_side", 45: "left_front", 22.5: "left_side_front", 67.5: "front_left_front", 112.5: "front_right_front", 157.5: "right_side_front", 202.5: "right_side_back", 247.5: "back_right_back", 292.5: "back_left_back", 337.5: "left_side_back"}

    def __init__(self, x_resolution: float = 0.1, vel_resolution: float = 0.01):
        self.x_resolution, self.vel_resolution = x_resolution, vel_resolution
        self.lanes = [lane_state(i, x_resolution, vel_resolution) for i in range(1, 6)]
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

    def _car_update(self, state: dict):
        for lane in self.lanes:
            if lane.spotted:
                continue
            possible_vel_bins = [i for i in range(lane.vel_bins) if lane.velocity[i] > 0]
            possible_x_bins = [i for i in range(lane.x_bins) if lane.position[i] > 0]

            for vel_bin in possible_vel_bins:
                for x_bin in possible_x_bins:
                    # Percent chance this is the case the lane is in
                    lane_chance = lane.position[x_bin] * lane.velocity[vel_bin]

                    # First update position probability for this case
                    new_pos = lane.x_bin_means[x_bin] + lane.vel_bin_means[vel_bin]
                    if new_pos < -1000 or new_pos >= 2600: # in this case, the car will despawn
                        lane.new_no_car += lane_chance
                        continue
                    else:
                        new_bin = (new_pos+1000)//lane.x_resolution
                        lane.new_position[new_bin] += lane_chance

                    # Then update velocity probability for this case
                    low_vel, hi_vel = 0.95*lane.vel_bin_means[vel_bin], 1.05*lane.vel_bin_means[vel_bin]
                    low_bin, hi_bin = low_vel//lane.vel_resolution, hi_vel//lane.vel_resolution
                    diff = hi_vel-low_vel
                    perc_per_length = 1/diff
                    for i in range(low_bin, hi_bin + 1):
                        if i == low_bin:
                            length = lane.velocity_bin_dividers[low_bin+1]-low_vel
                        elif i == hi_bin:
                            length = hi_vel-lane.velocity_bin_dividers[hi_bin]
                        else:
                            length=lane.vel_resolution
                        # Percent chance that the new speed will be the one corresponding to low_bin (given previous velocity and speed)
                        new_speed_chance = perc_per_length*length 
                        lane.new_velocity[i] += new_speed_chance * lane_chance

            # In case there is no car on the lane, a new one might spawn.
            # First we need to calculate the chance that the 4 other lanes don't already all have a car (max 4 cars)
            other_lanes_car_probs = [1-other_lane.no_car for other_lane in self.lanes if other_lane.lane != lane.lane]
            four_cars = np.prod(other_lanes_car_probs)
            # if there are not 4 cars, and there is not a car in the lane already, a new one will spawn
            spawn_chance = (1-four_cars)*lane.no_car
            lane.new_position[lane.behind_spawn_bin] += 0.5 * spawn_chance
            lane.new_position[lane.front_spawn_bin] += 0.5 * spawn_chance
            lane.new_no_car += lane.no_car - spawn_chance
            


        pass

    def _car_despawn(self, state: dict):
        pass
    
    def _car_spawn(self, state: dict):
        pass


    def _update_lanes(self, state: dict):
        print(self.lanes[0].x_bin_means)
        self._car_update(state)
        self._car_despawn(state)
        self._car_spawn(state)
        
        
        tick = state["elapsed_ticks"]


        if tick == 1:
            for lane in self.lanes:
                lane.position = np.zeros(lane.x_bins)
                lane.velocity = np.zeros(lane.vel_bins)
                lane.no_car = 1
        if tick > 4:
            pass


    def update(self, state: dict):
        # print("--------------------------------")
        
        # print(self.lanes[0].x_bin_dividers)
        # print(self.lanes[0].velocity_bin_dividers)

        # print("STATE:\n", state)
        # print("AVAILABLE SENSORS:\n", self.available_sensors) if hasattr(self, "available_sensors") else print("AVAILABLE SENSORS: None")
        # print("--------------------------------")
        # time.sleep(5)
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

