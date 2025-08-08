import numpy as np
from typing import List, Tuple
from models.utilities.sensor_parser import SensorParser
from models.laneshift_config import LaneShiftConfig, get_default_config

class LaneShift:

    def __init__(self, config: LaneShiftConfig | None = None):
        # Configuration holding tunable hyper-parameters
        self.config: LaneShiftConfig = config or get_default_config()

        height = self.config.HEIGHT
        margins = self.config.MARGINS
        usable_height = height - margins * 2
        self.lane_height = usable_height / self.config.LANE_COUNT

        # Frequently tuned constants
        self.front_buffer = self.config.FRONT_BUFFER
        self.collision_ticks_threshold = self.config.COLLISION_TICKS_THRESHOLD
        self.collision_time_threshold = self.config.COLLISION_TIME_THRESHOLD
        self.lane_shift_distance = self.config.LANE_SHIFT_DISTANCE
        self.action_queue_margin_front = self.config.ACTION_QUEUE_MARGIN_FRONT
        self.action_queue_margin_side = self.config.ACTION_QUEUE_MARGIN_SIDE
        
        # Acceleration phase thresholds
        self.start_acceleration_ticks = self.config.START_ACCELERATION_TICKS
        self.early_accelerate_ticks = self.config.EARLY_ACCELERATE_TICKS
        self.full_stop_acceleration_ticks = self.config.FULL_STOP_ACCELERATION_TICKS
        
        self.car_dimensions = (360, 179)
        
        self.verbose = False

        # Starting lane is 3 (middle)
        self.lane = 3
        self.desired_lane = 3
        self.last_distance_front = None
        self.last_distance_side = {-1: None, 1: None}
        self.last_lane = 3

        self.ypos = 0 # Starting y postition
        self.yvelocity = 0 # Starting y velocity

        self.action_queue = []

        self.parser = SensorParser()  # Initialize the sensor parser

        # In general, front is 0 and reference point. Left is negative, right is positive. Also mostly for my own understanding of their weird namings.
        self.sensor_dict = {
                           0: "front",
                           1: "front_right_front",
                           2: "right_front",
                           3: "right_side_front",
                           4: "right_side",
                           5: "right_side_back",
                           6: "right_back",
                           7: "back_right_back",
                          -1: "front_left_front",
                          -2: "left_front",
                          -3: "left_side_front",
                          -4: "left_side",
                          -5: "left_side_back",
                          -6: "left_back",
                          -7: "back_left_back"
                           }

        # Number of initial lane shifts // only for testing purposes
        self.shifts = 0

        # Distances from car sensor to boundaries of car in a given lane. Number is sensor type,
        # first tuple is interval of one lane, second tuple is interval of the next lane.
        self.sensor_type_to_lane = {1: ((335, 840), (940, 950)),
                                    2: ((180, 470), (500, 770)),
                                    3: ((130, 360), (380, 600)),
                                    4: ((115, 145), (350, 360))}


        
        # Ordered from front to side, positive is right, negative is left
        self.sensor_to_type = {"front_right_front": 1, "front_left_front": -1, "back_left_back": -1, "back_right_back": 1,
                         "right_front": 2, "left_front": -2, "right_back": 2, "left_back": -2,
                         "right_side_front": 3, "left_side_front": -3, "right_side_back": 3, "left_side_back": -3,
                         "right_side": 4, "left_side": -4}

        self.type_to_sensor = {1: ["front_right_front", "back_right_back"],
                               -1: ["front_left_front", "back_left_back"],
                            2: ["right_front", "right_back", ],
                            -2: ["left_front", "left_back"],
                            3: ["right_side_front","right_side_back"],
                            -3: ["left_side_front", "left_side_back"],
                            4: ["right_side"],
                            -4: ["left_side"]}

        self.measurement_to_distance = 1.126567433995778 #  # Found by comparing measured distances to expected distances
        angle = 2*np.pi / 16  # 22.5 degrees, as per the original code
        self.sensor_type_to_angle = {sensor_type: np.sign(sensor_type) * (4 - abs(sensor_type)) * angle for sensor_type in self.type_to_sensor} 

        self.measured_lane_ypos = {1: -441.8, 2: -220.9, 3: 0, 4: 220.9, 5: 441.8, "UPPER_WALL": -559, "LOWER_WALL": 561} # Measured positions from 0
        
        self.lane_ypos = {1: -self.lane_height * 2, 2: -self.lane_height, 3: 0, 4: self.lane_height, 5: self.lane_height * 2,
                          "UPPER_WALL": self.measured_lane_ypos["UPPER_WALL"] * self.measurement_to_distance, "LOWER_WALL": self.measured_lane_ypos["LOWER_WALL"]} # Expected positions from 0

        self.aborting = False # Indicates whether the car has stopped an action to not die
        

    # ____________________________________________________Functions that update internal state tracking____________________________________________________

    def reset(self):
        """Reset the internal state of the car."""
        self.lane = 3
        self.desired_lane = 3
        self.last_distance_front = None
        self.ypos = 0
        self.yvelocity = 0
        self.action_queue.clear()
        self.shifts = 0


    def update_ypos(self, state) -> None:
        self.ypos += state["velocity"]["y"]

    def determine_velocity_front(self, state: dict) -> float | None:
        """"Returns the volocity of the car in front based on the front sensor reading."""
        current_distance = state["sensors"]["front"]
        measured_velocity = (current_distance - self.last_distance_front) if (self.last_distance_front and current_distance) else None
        self.last_distance_front = current_distance
        
        return measured_velocity
    
    def determine_velocity_side(self, state: dict, direction: int) -> float | None:
        """Returns the measured velocity of the car one lane to the side based on forward type 1 sensor reading."""

        sensor = self.sensor_dict[direction]

        interval = self.sensor_type_to_lane[abs(direction)]
        if state["sensors"][sensor] and not interval[0][0] <= state["sensors"][sensor] <= interval[0][1]:
            self.last_distance_side[direction] = None
            return None

        angle = self.sensor_type_to_angle[direction]
        current_distance = np.sin(angle) * state["sensors"][sensor] if state["sensors"][sensor] else None
        measured_velocity = (current_distance - self.last_distance_side[direction]) if (self.last_distance_side[direction] and current_distance) else None
        self.last_distance_side = {direction: current_distance, -direction: None}

        return measured_velocity
        


    def detect_car_in_neighboring_lane(self, state: dict) -> dict[str, bool]:
        """
        Detects if there is a car in the neighboring lane based on the sensor readings.

        Returns a dictionary with keys "right_1", "right_2", "left_1", "left_2".
        
        If the correspronding lane distance is out of map, value is False.
        """
        val_to_str = {1: "right_1", 2: "right_2", -1: "left_1", -2: "left_2"}
        lanes = {"right_1": False, "right_2": False, "left_1": False, "left_2": False}
        
        for sensor_type in self.type_to_sensor:
            for sensor in self.type_to_sensor[sensor_type]:
                if not state["sensors"][sensor] or sensor not in [self.sensor_dict[i] for i in range(-6, 7) if i != 0]:
                    continue
                I1, I2 = self.sensor_type_to_lane[abs(sensor_type)]
                if I1[0] <= state["sensors"][sensor] <= I1[1]:
                    lanes[val_to_str[int(np.sign(sensor_type))]] = True
                elif I2[0] <= state["sensors"][sensor] <= I2[1]:
                    lanes[val_to_str[int(np.sign(sensor_type)) * 2]] = True

        diff = self.lane - 3
        for i in range(1, abs(diff)+1):
            lanes[val_to_str[int(np.sign(diff) * (3-i))]] = False

        return lanes

    def safe_to_shift(self, state: dict, direction, v0: float | None, a = 0.05) -> bool:
        """
        Checks if it is safe to shift lanes in the given direction. (OBS: only checks front atm, shift direction is hence not used)
        """
        # If we don't see a car, it is safe to shift (but we will in reality never shift if we don't see a car)
        if not state["sensors"]["front"]:
            return True
        # If can not yet determine velocity, but spot a car, assume it is not safe
        if not v0: 
            return False
        if v0 > 0:
            return True

        v0 = -1*v0 # The car is coming towards us
        
        d = max(0, state["sensors"]["front"] - self.front_buffer) # Distance to front car (measured in pixels), 250 could be the sweet spot, 300 is safer

        # Determine ticks to collision assuming a = 0.05 – true mean is 0, but set to 0.05 kind of arbitrarily to manage risk
        ticks_to_collision = (-(2/a * v0 - 1) + np.sqrt((2/a * v0 - 1)**2 + 8 * d / a)) / 2
        
        return ticks_to_collision > self.collision_ticks_threshold
        
    def closest_lane(self):
        """Returns the closest lane to the car's current y-position."""
        closest_lane = min(self.lane_ypos, key=lambda x: abs(self.lane_ypos[x] - self.ypos))
        
        return closest_lane

    def actions_left(self) -> int:
        """Returns the number of actions left in the action queue."""
        return len(self.action_queue)

    def ticks_to_collision(self, state: dict, v0, a = 0.05, direction = "front") -> float | None:
        if not state["sensors"][direction]:
            return None
        # If can not yet determine velocity, but spot a car, assume it is not safe
        if not v0:
            return None
        angle = np.pi/2 if direction == "front" else self.sensor_type_to_angle[self.sensor_to_type[direction]]
        v0 = max(0, -1*v0) # The car is coming towards us
        d = max(0, np.sin(angle) * state["sensors"][direction] - self.front_buffer) # Distance to front car (measured in pixels), 250 could be the sweet spot, 300 is safer
        # Determine ticks to collision assuming a = 0.05 – true mean is 0, but set to 0.05 kind of arbitrarily to manage risk
        return (-(2/a * v0 - 1) + np.sqrt((2/a * v0 - 1)**2 + 8 * d / a)) / 2


    # ____________________________________________________Functions that define larger action sequences____________________________________________________

    def queue_actions_to_position(self, distance: float | str, v0, brake = True) -> None:
        """Generates a list of actions to move the car to a specific y-position.
        Args: 
            distance (float | str): The distance to move. If a string, it should be "right" or "left", corresponding to a lane shift.
            v0 (float): The initial velocity of the car.
            brake (bool): Whether to apply brakes or not (resulting in final velocity = 0). Default is True. 
        """

        if isinstance(distance, str):
            distance = self.lane_shift_distance if distance == "right" else -self.lane_shift_distance

        if distance > 0:    
            break_time = np.sqrt(10 * distance + 50 * v0**2 + 5 * v0)
            accelerate_time = -10 * v0 + break_time
            if np.isnan(break_time):
                print("WARNING: Yo you're asking too much of me here, I can't break that fast. I'll just do nothing")
                return
            accelerate_time = round(accelerate_time)
            break_time = round(v0/0.1) +accelerate_time

        elif distance < 0:
            break_time = np.sqrt(50*v0**2 - 5*v0 - 10*distance)
            accelerate_time = break_time + 10*v0
            if np.isnan(break_time):
                print("WARNING: Yo you're asking too much of me here, I can't break that fast. I'll just do nothing")
                return
            accelerate_time = round(accelerate_time)
            break_time = round(-v0/0.1) +accelerate_time

        else:
            print("WARNING: Was asked to move a distance of 0, idk why man you might as well not ask but I'll do nothing")
        
        action_dict = {-1: "STEER_LEFT", 0: "NOTHING", 1: "STEER_RIGHT"}
        
        abs_distance, direction = abs(distance), np.sign(distance)



        # b = 20*v0 - 1; c = -20 * abs_distance
        # if brake:
        #     c /= 2
        
        # Quadratic equation to determine the number of ticks needed to reach the target distance
        # Derived from the discrete time difference equations with constant acceleration (x_{t+1} = x_t + v_t, v_{t+1} = v_t + a: a = 0.1)
        # ticks = int((-b + np.sqrt(b**2 - 4*c)) / 2)
        actions = [action_dict[direction]]*round(accelerate_time) + [action_dict[-direction]]*round(break_time)
        self.last_lane = self.lane
        self.desired_lane = min(self.lane_ypos, key=lambda x: abs(self.lane_ypos[x] - (self.ypos + distance)))
        # print(f"DICIDED TO GO {'LEFT' if direction < 0 else 'RIGHT'} to lane {self.lane}, ticks needed: {ticks * (1 * brake)}")
        self.action_queue.extend(actions)
    
    def get_to_lane(self, lane: int, state: dict, brake = True) -> None:
        self.desired_lane = lane
        self.last_lane = self.lane
        self.queue_actions_to_position(self.lane_ypos[lane] - self.ypos, state["velocity"]["y"], brake = True)
        
    
    def get_to_closest_lane(self, state: dict, brake = True) -> None:
        """Queues actions to move the car to the closest lane based on its current y-position."""
        closest_lane = self.closest_lane()
        if self.verbose: print(f"Closest lane to y-position {self.ypos} is {closest_lane} with y-position {self.lane_ypos[closest_lane]}")
        self.last_lane = self.lane
        self.desired_lane = closest_lane
        self.get_to_lane(closest_lane, state, brake=brake)
    

    def stand_still(self, state: dict) -> bool:
        drift = self.lane_ypos[self.lane] - self.ypos
        if abs(drift) > 10 and np.sign(state["velocity"]["y"]) == -np.sign(drift):  # If the car is not in the lane, steer towards the lane
            self.get_to_closest_lane(state, brake = True)
            return True
    

    # _____________________________________________________________Functions that handle actions_____________________________________________________________
    
    def queue_action(self, action: str | List) -> None:
        """Queue an action to be executed later."""
        if isinstance(action, list):
            self.action_queue.extend(action)
        elif action in ["STEER_LEFT", "STEER_RIGHT", "ACCELERATE", "DECELERATE", "NOTHING"]:
            self.action_queue.append(action)
        else:
            raise ValueError(f"Invalid action: {action}. Must be one of ['STEER_LEFT', 'STEER_RIGHT', 'ACCELERATE', 'DECELERATE', 'NOTHING'].")

    def pop_next_action(self) -> List[str]:
        """Return the next action in the action queue, or "NOTHING" if the queue is empty."""
        return [self.action_queue.pop(0)] if self.action_queue else ["NOTHING"]

    def clear_action_queue(self) -> None:
        """Clear the action queue, and sets the aborting flag to True."""
        
        if self.verbose: print("Clearing action queue")
        
        self.action_queue.clear()
        


    # _____________________________________________________________________Decision logic_____________________________________________________________________

    def return_action(self, state: dict) -> List[str]:
        """Logic for determining the next action based on the current state of the environment."""
        # For repeated runs, reset the internal state (ONLY FOR TESTING PURPOSES)
        
        if not state["distance"]:
            self.reset()

        
        
        # Running internal state updates
        self.lane = self.closest_lane()  # Update the lane based on the current y-position
        # current_front_velocity = parse[self.lane - 1]["velocity"]
        current_front_velocity = self.determine_velocity_front(state)
        collision_time_front = self.ticks_to_collision(state, current_front_velocity)

        self.update_ypos(state)
        if state["elapsed_ticks"] > 500:
            pass
        parse = self.parser.parse_sensors(state, self.ypos)
        # print(parse)
        # print(parse[self.lane]["x"], self.ypos)
        # if not self.actions_left() and state["velocity"]["y"] != 0:
        #     self.queue_action("STEER_RIGHT" if state["velocity"]["y"] < 0 else "STEER_LEFT")
        #     return self.pop_next_action()
        
        # if self.action_queue:
        #     print(f"Action: {self.action_queue[0]}, Lane: {self.lane}, Y-pos: {self.ypos}, Front velocity: {current_front_velocity}, Distance to front: {state['sensors']['front']}, ABORTING: {self.aborting}")


        # Safety logic
        # if self.aborting:
        #     self.aborting = False

        #     self.get_to_closest_lane(state, brake = True)
            
            # if current_front_velocity and current_front_velocity < 0:
            #     self.clear_action_queue()
            #     self.queue_action("DECELERATE")
            #     return self.pop_next_action()

        
        
        if collision_time_front and self.desired_lane == self.lane and not self.aborting:
            if self.actions_left() + self.action_queue_margin_front > collision_time_front:
                if self.verbose: print("Call 1")
                self.aborting = True
                self.clear_action_queue()
                
                self.get_to_lane(self.last_lane, state, brake = True)


        elif self.desired_lane != self.lane and not self.aborting:
            # current_side_velocity = self.determine_velocity_side(state, self.desired_lane - self.lane)
            current_side_velocity = parse[self.desired_lane - 1]["velocity"]
            collision_time_side = self.ticks_to_collision(state, current_side_velocity, direction=self.sensor_dict[self.desired_lane - self.lane])
            if (collision_time_side and self.actions_left() + self.action_queue_margin_side > collision_time_side) and self.desired_lane != self.lane:
                self.aborting = True
                
                if self.verbose: print("Call 2")
                self.clear_action_queue()
                self.get_to_lane(self.last_lane, state, brake = True)
                # self.action_queue = self.action_queue[:self.actions_left() // 2]
                # self.queue_action(["DECELERATE"]*100)

                # self.queue_action("DECELERATE")
    
        
        # if self.actions

        if self.aborting and not (-1 < state["velocity"]["y"] < 1):
            if self.verbose: print("Call 8")
            self.aborting = False
            self.clear_action_queue()

            # self.queue_action(["DECELERATE"]*100)

        if self.action_queue:
            return self.pop_next_action()
        
        # ACTION QUEUE LOGIC FROM HERE ____________________________________________________________________

        # Kill the car to hide score
        # if state["distance"] > 23000:
        #     self.queue_action(["STEER_LEFT"]*1000)
        #     return self.pop_next_action()
        
        
        if not state["sensors"]["front"]:
            
            
            if not self.stand_still(state):
                t = state["elapsed_ticks"]
                if t < self.start_acceleration_ticks:
                    self.queue_action(["ACCELERATE"])
                elif t < self.early_accelerate_ticks:
                    self.queue_action(["ACCELERATE", "NOTHING"])
                elif t < self.full_stop_acceleration_ticks:
                    self.queue_action(["NOTHING"])
                else:
                    self.queue_action(["ACCELERATE"])  # Late stage, keep speed up
            return self.pop_next_action()


        # Determine whether the car can see a car in neighboring lanes
        if self.lane < 5:
            right_1 = True if parse[self.lane]["x"] and parse[self.lane]["x"] > -400 else False
        else:
            right_1 = True
        if self.lane > 1:
            left_1 = True if parse[self.lane - 2]["x"] and parse[self.lane - 2]["x"] > -400 else False
        else:
            left_1 = True
        # print(f"{parse[self.lane]['x']}, lane: {self.lane}, {not parse[self.lane]['x']}")

        # if not self.safe_to_shift(state=state, direction=None, v0=current_front_velocity, a = 0.05):
        #     # print("not self.safe_to_shift")
        #     self.queue_action("DECELERATE")
        #     return self.pop_next_action()


        

        # i.e. "See car, go: nono"
        if (not collision_time_front or collision_time_front > self.collision_time_threshold):
            if left_1 and not right_1 and self.lane < 5:
                if self.verbose: print("Call 3", f"{self.lane}, {left_1}, {right_1}")
                self.get_to_lane(self.lane + 1, state)
                # self.queue_actions_to_position("right", state["velocity"]["y"])

                # print(f"DECIDED TO GO RIGHT, as left_1 is {left_1} and right_1 is {right_1}, lane: {self.lane}")
                return self.pop_next_action()

            if right_1 and not left_1 and self.lane > 1:
                if self.verbose: print(f"Call 4. self.lane: {self.lane}")
                
                self.get_to_lane(self.lane - 1, state)
                # self.queue_actions_to_position("left", state["velocity"]["y"])

                return self.pop_next_action()
        
        if (collision_time_front and collision_time_front < self.collision_time_threshold):
            self.queue_action("DECELERATE")

            return self.pop_next_action()
        if right_1 and left_1:
            if self.verbose: print(f"Call 5, {(right_1, left_1)}")
            
            # self.stand_still(state)

            self.queue_action("DECELERATE")

            # print(f"DECIDED TO DECELERATE, as left_1 is {left_1} and right_1 is {right_1}, lane: {self.lane}")
            return self.pop_next_action()

        
        unsafe = right_1 if self.lane < 4 else left_1
        if unsafe:
            if self.verbose: print("Call 6", f"{(right_1, left_1, self.lane < 4)}")
            self.queue_action("DECELERATE")
            return self.pop_next_action()

        if self.verbose: print(f"Call 7, {(right_1, left_1)}")
        
        self.get_to_lane(1 + self.lane if self.lane == 3 else np.sign(3 - self.lane) + self.lane, state, True) # If both directions seem good, go towards the middle
        # self.queue_actions_to_position(224 if self.lane == 3 else np.sign(3 - self.lane) * 224, state["velocity"]["y"], True) # Cool way of writing "go towards the middle lane (right if you're already in the middle)"
        return self.pop_next_action()
