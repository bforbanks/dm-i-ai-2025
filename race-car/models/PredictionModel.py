"""THIS IS NOT A MODEL, IT IS A CLASS USED TO PREDICT THE POSITIONS 
AND VELOCITIES OF OTHER CARS"""

from src.game.core import GameState
import numpy as np
import time
import matplotlib.pyplot as plt

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

        self.velocity_bin_dividers = np.arange(0, self.max_vel+vel_resolution, vel_resolution) # lower bound included, upper bound excluded (except last one)
        self.vel_bins = len(self.velocity_bin_dividers)-1 # velocity-values ranging from 0 to max_vel
        
        self.x_bin_dividers = np.arange(0, 3600+x_resolution, x_resolution) # lower bound included, upper bound excluded (except last one)
        self.x_bins = len(self.x_bin_dividers)-1 # x-values ranging from -1000 to 2600
        
        # Joint probability distribution: P(position=x_bin, velocity=vel_bin)
        self.joint_prob = {}  # {(x_bin, vel_bin): probability}
        self.new_joint_prob = {}  # For updates

        # for ease of use, we will calculate mean bin (average of the bounds)
        self.x_bin_means = 0.5 * (self.x_bin_dividers[:-1] + self.x_bin_dividers[1:])
        self.vel_bin_means = 0.5 * (self.velocity_bin_dividers[:-1] + self.velocity_bin_dividers[1:])
        
        # as well as the bins of the potential spawnpoints of new cars
        self.behind_spawn_bin = int(self.behind_spawn//self.x_resolution)
        self.front_spawn_bin = int(self.front_spawn//self.x_resolution)
class PredictionModel:
    # 0 degrees is up, 90 degrees is right
    all_sensors = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]
    sdict = {90: "front", 135: "right_front", 180: "right_side", 225: "right_back", 270: "back", 315: "left_back", 0: "left_side", 45: "left_front", 22.5: "left_side_front", 67.5: "front_left_front", 112.5: "front_right_front", 157.5: "right_side_front", 202.5: "right_side_back", 247.5: "back_right_back", 292.5: "back_left_back", 337.5: "left_side_back"}
    ego_x = 620 # Constant, only distance changes
    top_wall = 1071 # Lower bound +[0;1]
    bottom_wall = -48 # Upper bound +[-1;0]

    def __init__(self, x_resolution: float = 1, vel_resolution: float = 0.1):
        self.x_resolution, self.vel_resolution = x_resolution, vel_resolution
        self.lanes = [lane_state(i, x_resolution, vel_resolution) for i in range(1, 6)]
        self.uncertain_sensors = [67.5, 112.5, 247.5, 292.5]
        self.available_sensors = []
        self.ego_y = 510

        self.fully_initialized = False

    def _init_with_state(self, state: GameState):
        """Usually the state is not available when initializing models, so we need a second
        initialization"""
        if state["elapsed_ticks"] > 1:
            self.available_sensors = [sensor for sensor in self.all_sensors if sensor not in self.uncertain_sensors and state["sensors"][self.sdict[sensor]]]
            self.fully_initialized = True

    def _update_available_sensors(self, state: dict):
        if not self.fully_initialized:
            self._init_with_state(state)
        for sensor in self.uncertain_sensors:
            if state["sensors"][self.sdict[sensor]]:
                self.available_sensors.append(sensor)
                self.uncertain_sensors.remove(sensor)

    def _sensor_update(self, state: dict):
        detected_nothing = []
        for sensor in self.available_sensors:
            sensor_angle = sensor*np.pi/180
            if not state["sensors"][self.sdict[sensor]]:
                continue
            x = np.sin(sensor_angle)*state["sensors"][self.sdict[sensor]]
            y = np.cos(sensor_angle)*state["sensors"][self.sdict[sensor]]
            loc_y = -y+self.ego_y
            if loc_y <= self.bottom_wall or loc_y >= self.top_wall:
                detected_nothing.append(sensor)
                continue
            print("Sensor: ", self.sdict[sensor], sensor)
            # print(x, y)
            # print(self.ego_y, state["velocity"]["y"])
            print(x+self.ego_x, -y+self.ego_y)
            
            print("--------------------------------")
            

    def _car_update(self, state: dict, tick: int):
        if tick == 1:
            for lane in self.lanes:
                self._init_joint_prob(state, lane)
                lane.no_car = 0.2
        elif tick < 4:
            return
        for lane in self.lanes:
            if lane.spotted:
                continue
            
            # Iterate over all current joint probability states
            total_prob = 0
            total_joint_prob = 0
            for (x_bin, vel_bin), joint_prob in lane.joint_prob.items():
                if joint_prob <= 1e-7:
                    continue
                    
                # Calculate new position based on current position and velocity
                new_pos = lane.x_bin_means[x_bin] + lane.vel_bin_means[vel_bin] - state["velocity"]["x"] # x is relative to ego
                
                if new_pos < 0 or new_pos >= 3600:  # car will despawn
                    lane.new_no_car += joint_prob
                    continue
                
                new_x_bin = int(new_pos // lane.x_resolution)
                if new_x_bin >= lane.x_bins:  # safety check
                    raise ValueError("New x bin is out of bounds")
                    lane.new_no_car += joint_prob
                    continue

                # Calculate new velocity distribution (slight variation around current velocity)
                low_vel = 0.95 * lane.vel_bin_means[vel_bin]
                hi_vel = 1.05 * lane.vel_bin_means[vel_bin]
                low_bin = int(low_vel // lane.vel_resolution)
                hi_bin = int(hi_vel // lane.vel_resolution)
                                
                diff = hi_vel - low_vel
                perc_per_length = 1 / diff
                total_joint_prob += joint_prob
                total_new_speed=0            

                for new_vel_bin in range(low_bin, hi_bin + 1):
                    if hi_bin == low_bin:
                        length = diff
                    elif new_vel_bin == low_bin:
                        length = lane.velocity_bin_dividers[low_bin+1]-low_vel
                    elif new_vel_bin == hi_bin:
                        length = hi_vel-lane.velocity_bin_dividers[hi_bin]
                    else: 
                        length = lane.vel_resolution
                    new_speed_chance = perc_per_length * length
                    total_new_speed += new_speed_chance
                    # Add to joint probability distribution
                    new_state = (new_x_bin, new_vel_bin)
                    if new_state in lane.new_joint_prob:
                        total_prob += new_speed_chance * joint_prob
                        # total_joint_prob += joint_prob
                        lane.new_joint_prob[new_state] += new_speed_chance * joint_prob
                    else:
                        # total_joint_prob += joint_prob
                        total_prob += new_speed_chance * joint_prob
                        lane.new_joint_prob[new_state] = new_speed_chance * joint_prob
                        # print(lane.new_joint_prob[new_state])
                # if total_new_speed-1 > 1e-7:
                #     print("TOTAL NEW SPEED:", total_new_speed)
                # print("TOTAL NEW SPEED:", total_new_speed)

            the_sum=sum(lane.joint_prob.values())
            # print(the_sum, the_sum+lane.no_car)
            # print("TOTAL PROB:", total_prob, total_joint_prob)
            # Handle car spawning
            # Calculate the chance that the 4 other lanes don't already all have a car (max 4 cars)
            other_lanes_car_probs = [1-other_lane.no_car for other_lane in self.lanes if other_lane.lane != lane.lane]
            four_cars = np.prod(other_lanes_car_probs)
            # if there are not 4 cars, and there is not a car in the lane already, a new one will spawn
            spawn_chance = (1-four_cars) * lane.no_car

            # Spawn cars behind
            speed_chance_sum = 0
            for location in ["front", "behind"]:
                if location == "front":
                    low_vel = max(state["velocity"]["x"]-5, 0)
                    hi_vel = state["velocity"]["x"]
                else:
                    low_vel = state["velocity"]["x"]
                    hi_vel = state["velocity"]["x"]+5
                diff = hi_vel-low_vel
                low_bin = int(low_vel//lane.vel_resolution)
                hi_bin = int(hi_vel//lane.vel_resolution)
                perc_per_length = 1 / diff
                
                for new_vel_bin in range(low_bin, hi_bin+1):
                    if new_vel_bin == low_bin:
                        length = lane.velocity_bin_dividers[low_bin+1]-low_vel
                    elif new_vel_bin == hi_bin:
                        length = hi_vel-lane.velocity_bin_dividers[hi_bin]
                    else: 
                        length = lane.vel_resolution
                    new_speed_chance = perc_per_length * length                    
                    # Add to joint probability distribution
                    if location == "behind":
                        new_x_bin = lane.behind_spawn_bin
                    else:
                        new_x_bin = lane.front_spawn_bin
                    new_state = (new_x_bin, new_vel_bin)
                    if new_state in lane.new_joint_prob:
                        lane.new_joint_prob[new_state] += 0.5 * new_speed_chance * spawn_chance # 0.5 because 50% chance of being behind/front
                        speed_chance_sum += 0.5 * new_speed_chance * spawn_chance
                    else:
                        speed_chance_sum += 0.5 * new_speed_chance * spawn_chance
                        lane.new_joint_prob[new_state] = 0.5 * new_speed_chance * spawn_chance
            lane.new_no_car += lane.no_car - spawn_chance

            lane.joint_prob = lane.new_joint_prob.copy()
            lane.new_joint_prob = {}  # Clear for next iteration
            lane.no_car = lane.new_no_car
            lane.new_no_car = 0
            if lane.lane == 1 and tick % 10 == 0:
                the_sum=sum(lane.joint_prob.values())
                print(the_sum, the_sum+lane.no_car)

    def _init_joint_prob(self, state: dict, lane: lane_state):
        lane.joint_prob = {}  # Clear existing probabilities
        
        for location in ["front", "behind"]:
            if location == "front":
                low_vel = state["velocity"]["x"]-5
                hi_vel = state["velocity"]["x"]
            else:
                low_vel = state["velocity"]["x"]
                hi_vel = state["velocity"]["x"]+5
            diff = 5
            low_bin = int(low_vel//lane.vel_resolution)
            hi_bin = int(hi_vel//lane.vel_resolution)
            perc_per_length = 1 / diff
            
            for new_vel_bin in range(low_bin, hi_bin+1):
                if new_vel_bin == low_bin:
                    length = lane.velocity_bin_dividers[low_bin+1]-low_vel
                elif new_vel_bin == hi_bin:
                    length = hi_vel-lane.velocity_bin_dividers[hi_bin]
                else: 
                    length = lane.vel_resolution
                new_speed_chance = perc_per_length * length
                
                # Add to joint probability distribution
                if location == "behind":
                    new_x_bin = lane.behind_spawn_bin
                else:
                    new_x_bin = lane.front_spawn_bin
                new_state = (new_x_bin, new_vel_bin)
                if new_state in lane.joint_prob:
                    lane.joint_prob[new_state] += 0.5 * new_speed_chance * 0.8 # 0.8 spawn chance
                else:
                    lane.joint_prob[new_state] = 0.5 * new_speed_chance * 0.8


    def _update_internal_state(self, state: dict):
        self.ego_y += state["velocity"]["y"]
        self._sensor_update(state)
        tick = state["elapsed_ticks"]
        self._car_update(state, tick)

        self.visualization(tick)

    def visualization(self, tick: int):
        print("Progress", round(tick/180*100,1))
        if tick % 180 != 0:
            return
        # Plot only lane 1
        # Simple position-only plotting (ignoring velocities for speed)
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        fig.suptitle(f'Lane 1 Position Distribution (Tick {tick})')
        
        # Get lane 1 (index 0 since lanes are numbered 1-5)
        lane = self.lanes[0]
        
        # Just sum probabilities by position (ignore velocity)
        x_probs = np.zeros(lane.x_bins)
        
        for (x_bin, vel_bin), prob in lane.joint_prob.items():
            x_probs[x_bin] += prob
        
        # Only plot non-zero probabilities for speed
        nonzero_indices = np.where(x_probs > 0)[0]
        if len(nonzero_indices) > 0:
            x_positions = (lane.x_bin_means[nonzero_indices] - 1000)
            probs = x_probs[nonzero_indices]
            
            ax.bar(x_positions, probs, width=lane.x_resolution, 
                   color='blue', alpha=0.7)
            
            # Auto-scale y-axis with small margin
            max_prob = np.max(probs)
            ax.set_ylim(0, max_prob * 1.1)  # 10% margin above max
        
        ax.set_title(f'Lane {lane.lane} (No car: {lane.no_car:.2f})')
        ax.set_ylabel('Probability')
        ax.set_xlabel('X Position')
        
        plt.tight_layout()
        plt.show()

    


    def update(self, state: dict):
        # print("--------------------------------")
        
        # print(self.lanes[0].x_bin_dividers)
        # print(self.lanes[0].velocity_bin_dividers)

        # print("STATE:\n", state)
        # print("AVAILABLE SENSORS:\n", self.available_sensors) if hasattr(self, "available_sensors") else print("AVAILABLE SENSORS: None")
        # print("--------------------------------")
        # time.sleep(5)
        self._update_available_sensors(state)
        self._update_internal_state(state)

    def return_action(self, state):
        """This function is to show how to use this class as well as for testing. 
            Really you only need to run .update_lanes()
            every tick. Later, I might implement solution if ticks are skipped. 
            Of course, replace self with object name"""
        predictions = self.update(state)
        
        action = ["NOTHING"] # You would implement actual decision logic here

        return action

