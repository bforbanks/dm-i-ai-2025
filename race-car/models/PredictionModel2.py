"""THIS IS NOT A MODEL, IT IS A CLASS USED TO PREDICT THE POSITIONS 
AND VELOCITIES OF OTHER CARS"""

import logging
import os
print(os.getcwd())

from src.game.core import GameState
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import time
import matplotlib.pyplot as plt
from jax.ops import segment_sum

@jit
def _update_lane_prob(x_resolution: float, ego_velocity: float, lane_vel_bin_means: jnp.ndarray, joint_prob: jnp.ndarray) -> jnp.ndarray:
    x_shift = jnp.round((lane_vel_bin_means - ego_velocity) / x_resolution)
    X, V = joint_prob.shape

    # Compute base probabilities: shape (X, V)
    p = joint_prob / 7.0

    # Generate base index grids
    x = jnp.arange(X)[:, None]  # shape (X, 1)
    v = jnp.arange(V)[None, :]  # shape (1, V)

    # Compute destination x indices using broadcasted x_shift[v]
    x_dest = x + x_shift[v].astype(int)  # shape (X, V)
    x_dest = jnp.clip(x_dest, 0, X - 1)

    # Compute vertical offsets v ± 3
    offsets = jnp.arange(-3, 4)[:, None, None]  # shape (7, 1, 1)
    v_dest = jnp.broadcast_to(v, (7, X, V)) + offsets  # shape (7, X, V)
    v_dest = jnp.clip(v_dest, 0, V - 1)

    # Broadcast x_dest and p to shape (7, X, V)
    x_dest_b = jnp.broadcast_to(x_dest[None, :, :], (7, X, V))  # shape (7, X, V)
    p_b = jnp.broadcast_to(p[None, :, :], (7, X, V))            # shape (7, X, V)

    # Flatten everything
    x_idx = x_dest_b.reshape(-1)  # shape (7*X*V,)
    v_idx = v_dest.reshape(-1)    # shape (7*X*V,)
    p_vals = p_b.reshape(-1)      # shape (7*X*V,)

    # Flatten (x, v) index into single index: idx = x * V + v
    flat_idx = x_idx * V + v_idx
    flat_size = X * V

    # Segment sum to aggregate values at flat indices
    flat_result = segment_sum(p_vals, flat_idx, num_segments=flat_size)

    # Reshape back to (X, V)
    new_joint_prob = flat_result.reshape((X, V))

    return new_joint_prob

def _spawn_car(
    x_resolution: float, 
    ego_velocity: float, 
    joint_prob: jnp.ndarray, 
    spawn_chance: jnp.ndarray, 
    velocity_bin_dividers: jnp.ndarray
) -> jnp.ndarray:
    
    # Constants
    prob_density = spawn_chance / 10.0
    x_front = int(3220 // x_resolution)
    x_back = int(20 // x_resolution)
    # Velocity bin edges
    v_bins_start = velocity_bin_dividers[:-1]
    v_bins_end = velocity_bin_dividers[1:]
    # Compute overlaps
    slow_overlap = jnp.maximum(0.0, jnp.minimum(v_bins_end, ego_velocity) - jnp.maximum(v_bins_start, ego_velocity - 5.0))
    fast_overlap = jnp.maximum(0.0, jnp.minimum(v_bins_end, ego_velocity + 5.0) - jnp.maximum(v_bins_start, ego_velocity))
    # Create the spawn probabilities
    slow_probs = prob_density * slow_overlap
    fast_probs = prob_density * fast_overlap
    # Convert to numpy and do direct indexing
    import numpy as np
    # Force JAX to complete any pending operations
    result = np.array(joint_prob)
    result[x_front, :] += np.array(slow_probs)
    result[x_back, :] += np.array(fast_probs)
    return jnp.array(result)

@partial(jit, static_argnums=0)
def _init_joint_prob(x_resolution: float, ego_velocity: float, velocity_bin_dividers: jnp.ndarray) -> jnp.ndarray:
    
    spawn_chance = 0.8  # The preset spawn chance for initialization

    # 1. Define the spawn velocity ranges (total width is 10)
    slower_vel_range = jnp.array([ego_velocity - 5.0, ego_velocity])
    faster_vel_range = jnp.array([ego_velocity, ego_velocity + 5.0])
    
    # 2. Define the probability density.
    # The total `spawn_chance` is spread over a total velocity width of 10.0.
    prob_density = spawn_chance / 10.0

    # 3. Calculate the intersection length for each bin with each range.
    v_bins_start = velocity_bin_dividers[:-1]
    v_bins_end = velocity_bin_dividers[1:]

    # Slower car range overlap
    slower_overlap_start = jnp.maximum(v_bins_start, slower_vel_range[0])
    slower_overlap_end = jnp.minimum(v_bins_end, slower_vel_range[1])
    slower_intersection = jnp.maximum(0, slower_overlap_end - slower_overlap_start)
    
    # Faster car range overlap
    faster_overlap_start = jnp.maximum(v_bins_start, faster_vel_range[0])
    faster_overlap_end = jnp.minimum(v_bins_end, faster_vel_range[1])
    faster_intersection = jnp.maximum(0, faster_overlap_end - faster_overlap_start)

    # 4. Convert intersection lengths to probabilities for each bin.
    slower_probs_to_add = prob_density * slower_intersection
    faster_probs_to_add = prob_density * faster_intersection

    # 5. Define spawn x-locations and create the initial probability distribution.
    front_spawn_x_bin = int(3220 // x_resolution)
    behind_spawn_x_bin = int(20 // x_resolution)
    
    # Initialize an empty joint_prob array
    num_x_bins = int(3600 // x_resolution)  # Assuming x-range from your lane_state class
    num_v_bins = len(velocity_bin_dividers) - 1
    joint_prob = jnp.zeros((num_x_bins, num_v_bins))

    # Add the spawn probabilities
    joint_prob = joint_prob.at[front_spawn_x_bin, :].add(slower_probs_to_add)
    joint_prob = joint_prob.at[behind_spawn_x_bin, :].add(faster_probs_to_add)

    return joint_prob


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

        self.velocity_bin_dividers = jnp.arange(0, self.max_vel+vel_resolution, vel_resolution) # lower bound included, upper bound excluded (except last one)
        self.vel_bins = len(self.velocity_bin_dividers)-1 # velocity-values ranging from 0 to max_vel
        
        self.x_bin_dividers = jnp.arange(0, 3600+x_resolution, x_resolution) # lower bound included, upper bound excluded (except last one)
        self.x_bins = len(self.x_bin_dividers)-1 # x-values ranging from -1000 to 2600
        
        # Joint probability distribution: P(position=x_bin, velocity=vel_bin)
        # Using 2D JAX arrays instead of defaultdicts: [x_bin, vel_bin]
        self.joint_prob = jnp.zeros((self.x_bins, self.vel_bins), dtype=jnp.float64)
        self.new_joint_prob = jnp.zeros((self.x_bins, self.vel_bins), dtype=jnp.float64)

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

    def __init__(self, x_resolution: float = 0.5, vel_resolution_coef: float = 7, live_plot: bool = True):
        """Note that vel_resolution_coef must be an uneven whole number"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        self.x_resolution, self.vel_resolution = x_resolution, 0.2/vel_resolution_coef
        self.lanes = [lane_state(i, x_resolution, self.vel_resolution) for i in range(1, 6)]
        self.uncertain_sensors = [67.5, 112.5, 247.5, 292.5] #not relevant with new rules, but kept in case of changes
        self.available_sensors = []
        self.ego_y = 510
        self.live_plot = live_plot
        
        # Initialize live plotting components if needed
        if self.live_plot:
            plt.ion()  # Enable interactive mode
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 4))
            self.bar_container = None  # Will hold the bar plot
            plt.show(block=False)

        self.fully_initialized = False
        self.logger.info("PredictionModel initialized")

    def _init_with_state(self, state: GameState):
        """Usually the state is not available when initializing models, so we need a second
        initialization"""
        if state["elapsed_ticks"] > 1:
            self.available_sensors = [sensor for sensor in self.all_sensors if sensor not in self.uncertain_sensors and state["sensors"][self.sdict[sensor]]]
            self.fully_initialized = True
        self.logger.info("Available sensors: %s", self.available_sensors)

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
            sensor_angle = sensor*jnp.pi/180
            if not state["sensors"][self.sdict[sensor]]:
                continue
            x = jnp.sin(sensor_angle)*state["sensors"][self.sdict[sensor]]
            y = jnp.cos(sensor_angle)*state["sensors"][self.sdict[sensor]]
            loc_y = -y+self.ego_y
            if loc_y <= self.bottom_wall or loc_y >= self.top_wall:
                detected_nothing.append(sensor)
                continue
            # print("Sensor: ", self.sdict[sensor], sensor)
            # print(x, y)
            # print(self.ego_y, state["velocity"]["y"])
            # print(x+self.ego_x, -y+self.ego_y)
            
            # print("--------------------------------")
    
    def _car_update(self, state: dict, tick: int):
        if tick == 1:
            for lane in self.lanes:
                lane.joint_prob = _init_joint_prob(self.x_resolution, state["velocity"]["x"], lane.velocity_bin_dividers)
                lane.no_car = 0.2
        elif tick < 4:
            return
        for lane in self.lanes:
            # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            # im0 = axs[0].imshow(lane.joint_prob, aspect='auto', origin='lower')
            # axs[0].set_title(f"Lane {lane.lane} joint_prob (before)")
            # plt.colorbar(im0, ax=axs[0])
            # The update happens below
            self.logger.info("Ok! Updating lane!")
            start_time = time.time()
            lane.new_joint_prob = _update_lane_prob(self.x_resolution, state["velocity"]["x"], lane.vel_bin_means, lane.joint_prob)
            lane.new_joint_prob.block_until_ready()
            self.logger.info("Time taken: %s ms", round((time.time()-start_time)*1000))
            # im1 = axs[1].imshow(lane.new_joint_prob, aspect='auto', origin='lower')
            # axs[1].set_title(f"Lane {lane.lane} joint_prob (after)")
            # plt.colorbar(im1, ax=axs[1])
            # plt.tight_layout()
            # plt.show(block=True)

            # the_sum=jnp.sum(lane.joint_prob)
            # print(the_sum, the_sum+lane.no_car)
            # print("TOTAL PROB:", total_prob, total_joint_prob)
            # Handle car spawning
            # Calculate the chance that the 4 other lanes don't already all have a car (max 4 cars)
            other_lanes_car_probs = [1-other_lane.no_car for other_lane in self.lanes if other_lane.lane != lane.lane]
            four_cars = jnp.prod(jnp.array(other_lanes_car_probs))
            # if there are not 4 cars, and there is not a car in the lane already, a new one will spawn
            spawn_chance = (1-four_cars) * lane.no_car #TODO double-check this calculation
            self.logger.info("Spawning cars")
            #plot the joint_prob
            
            # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            # im0 = axs[0].imshow(lane.joint_prob, aspect='auto', origin='lower')
            # axs[0].set_title(f"Lane {lane.lane} joint_prob (before)")
            # plt.colorbar(im0, ax=axs[0])
            start_time = time.time()
            lane.new_joint_prob = _spawn_car(
                self.x_resolution,
                state["velocity"]["x"],
                lane.new_joint_prob,
                spawn_chance,
                lane.velocity_bin_dividers  # Pass the dividers array
            )
            

            self.logger.info("Time taken: %s ms", round((time.time()-start_time)*1000))

            # im1 = axs[1].imshow(lane.new_joint_prob, aspect='auto', origin='lower')
            # axs[1].set_title(f"Lane {lane.lane} joint_prob (after)")
            # plt.colorbar(im1, ax=axs[1])
            # plt.tight_layout()
            # plt.show(block=True)

            lane.new_no_car += lane.no_car - spawn_chance

            lane.joint_prob = lane.new_joint_prob.copy()
            lane.new_joint_prob = jnp.zeros_like(lane.new_joint_prob)  # Clear for next iteration
            lane.no_car = lane.new_no_car
            lane.new_no_car = 0
            if lane.lane == 1 and tick % 10 == 0:
                the_sum=jnp.sum(lane.joint_prob)
                self.logger.info("%s %s", the_sum, the_sum+lane.no_car)
                print(the_sum, the_sum+lane.no_car)


    def _update_internal_state(self, state: dict):
        self.ego_y += state["velocity"]["y"]
        self._sensor_update(state)
        self._car_update(state, state["elapsed_ticks"])
        self.visualization(state["elapsed_ticks"])

    def visualization(self, tick: int):
       
        if self.live_plot:
            self._live_visualization(tick)
        else:
            if tick % 10 != 0:
                return
            self._static_visualization(tick)

    def _live_visualization(self, tick: int):
        """Update the existing plot instead of creating new ones"""
        lane = self.lanes[0]  # Get lane 1
        
        # Calculate probabilities by summing over velocity bins
        x_probs = jnp.sum(lane.joint_prob, axis=1)
        
        # Only plot non-zero probabilities
        nonzero_indices = jnp.where(x_probs > 0)[0]
        
        if len(nonzero_indices) > 0:
            x_positions = (lane.x_bin_means[nonzero_indices] - 1000)
            probs = x_probs[nonzero_indices]
            
            # Clear previous plot
            self.ax.clear()
            
            # Create new bars
            self.ax.bar(x_positions, probs, width=lane.x_resolution, 
                       color='blue', alpha=0.7)
            
            # Update labels and title
            self.ax.set_title(f'Lane {lane.lane} (No car: {lane.no_car:.2f})')
            self.ax.set_ylabel('Probability')
            self.ax.set_xlabel('X Position')
            self.fig.suptitle(f'Lane 1 Position Distribution (Tick {tick})')
            
            # Auto-scale y-axis
            max_prob = jnp.max(probs)
            self.ax.set_ylim(0, max_prob * 1.1)
            
            # Update the plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)  # Small pause to allow plot to update

    def _static_visualization(self, tick: int):
        """Original visualization method that creates new plots"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        fig.suptitle(f'Lane 1 Position Distribution (Tick {tick})')
        
        lane = self.lanes[0]
        
        # Calculate probabilities by summing over velocity bins
        x_probs = jnp.sum(lane.joint_prob, axis=1)
        
        nonzero_indices = jnp.where(x_probs > 0)[0]
        if len(nonzero_indices) > 0:
            x_positions = (lane.x_bin_means[nonzero_indices] - 1000)
            probs = x_probs[nonzero_indices]
            
            ax.bar(x_positions, probs, width=lane.x_resolution, 
                   color='blue', alpha=0.7)
            
            max_prob = jnp.max(probs)
            ax.set_ylim(0, max_prob * 1.1)
        
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


if __name__ == "__main__":
    model = PredictionModel(live_plot=True)
    # print the time it takes to update the model
    start_time = time.time()
    state = {
            "elapsed_ticks": 0,
            "velocity": {"x": 10, "y": 0},
            "sensors": {
                "front": 100,
                "right_front": 100,
                "right_side": 100,
                "right_back": 100,
                "back": 100,
                "left_back": 100,
                "left_side": 100,
                "left_front": 100,
                "front_left_front": 100,
                "front_right_front": 100,
                "right_side_front": 100,
                "right_side_back": 100,
                "back_right_back": 100,
                "back_left_back": 100,
                "left_side_back": 100,
                "left_side_front": 100,
            }
        }
    for i in range(100):
        state["elapsed_ticks"] = i
        model.update(state)
    end_time = time.time()
    print("Time taken: ", end_time-start_time)
