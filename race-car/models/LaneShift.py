import numpy as np
from typing import List, Tuple

class LaneShift:

    def __init__(self):
        height = 1200
        margins = 40
        usable_height = height - margins * 2
        self.lane_height = usable_height / 5
        
        self.car_dimensions = (360, )
        

        # Starting lane is 3 (middle)
        self.lane = 3
        self.last_distance_front = None
        self.ypos = 0 # Starting y postition
        self.yvelocity = 0 # Starting y velocity

        self.action_queue = []

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

        # Copy of old intervals   
        # {1: ((345, 830), (940, 950)),
        #                          2: ((180, 460), (500, 770)),
        #                          3: ((140, 350), (380, 600)),
        #                          4: ((125, 135), (350, 360))}
        
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


        self.lane_ypos = {1: -441.8, 2: -220.9, 3: 0, 4: 220.9, 5: 441.8}

        self.skidaddle = True

    # _____________________________________________________Functions that update internal state tracking_____________________________________________________

    def reset(self):
        """Reset the internal state of the car."""
        self.lane = 3
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
        measured_velocity = current_distance - self.last_distance_front if (self.last_distance_front and current_distance) else None
        self.last_distance_front = current_distance
        
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
                if not state["sensors"][sensor]:
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
        
        v0 *= -1 # The car is coming towards us
        d = state["sensors"]["front"] - 250 # Distance to front car (measured in pixels), 250 could be the sweet spot, 300 is safer
        # Determine ticks to collision assuming a = 0.05 – true mean is 0, but set to 0.05 kind of arbitrarily to manage risk
        ticks_to_collision = (-(2/a * v0 - 1) + np.sqrt((2/a * v0 - 1)**2 + 8 * d / a)) / 2
        return ticks_to_collision > 47  
        
    def closest_lane(self):
        """Returns the closest lane to the car's current y-position."""
        pass


    # _____________________________________________________Functions that define larger action sequences_____________________________________________________

    def queue_actions_to_position(self, distance: float | str, v0, brake = True) -> None:
        """Generates a list of actions to move the car to a specific y-position.
        Args: 
            distance (float | str): The distance to move. If a string, it should be "right" or "left", corresponding to a lane shift.
            v0 (float): The initial velocity of the car.
            brake (bool): Whether to apply brakes or not (resulting in final velocity = 0). Default is True. 
        """
        
        if isinstance(distance, str):
            distance = 224 if distance == "right" else -224
        
        action_dict = {-1: "STEER_LEFT", 0: "NOTHING", 1: "STEER_RIGHT"}
        
        abs_distance, direction = abs(distance), np.sign(distance)
        b = 20*v0 - 1; c = -20 * abs_distance
        if brake:
            c /= 2
        # Quadratic equation to determine the number of ticks needed to reach the target distance
        # Derived from the discrete time difference equations with constant acceleration (x_{t+1} = x_t + v_t, v_{t+1} = v_t + a: a = 0.1)
        ticks = int((-b + np.sqrt(b**2 - 4*c)) / 2)
        actions = [action_dict[direction]]*ticks + [action_dict[-direction]]*(ticks * brake)
        self.lane += direction
        
        # print(f"DICIDED TO GO {'LEFT' if direction < 0 else 'RIGHT'} to lane {self.lane}, ticks needed: {ticks * (1 * brake)}")
        self.action_queue.extend(actions)
    
    def get_to_lane(self, lane: int, state: dict, brake = True) -> None:
        self.queue_actions_to_position(self.lane_ypos[lane] - self.ypos, state["velocity"]["y"], brake = True)
    
    
    # def ticks_to_collision(self, )
    

    # ______________________________________________________________Functions that handle actions______________________________________________________________
    
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
        """Clear the action queue."""
        self.action_queue.clear()


    # ______________________________________________________________________Decision logic______________________________________________________________________

    def return_action(self, state: dict) -> List[str]:
        """Logic for determining the next action based on the current state of the environment."""

        # For repeated runs, reset the internal state (ONLY FOR TESTING PURPOSES)
        if not state["distance"]:
            self.reset()

        # Running internal state updates
        current_front_velocity = self.determine_velocity_front(state)
        # print(f"Current front velocity: {current_front_velocity}, distance to front: {state['sensors']['front']}, current lane: {self.lane}, ypos: {self.ypos}, {state['sensors']['front_left_front']}")
        self.update_ypos(state)
        print(self.ypos, self.lane)
        # if self.actions

        if self.action_queue:
            return self.pop_next_action()
        
        # ACTION QUEUE LOGIC FROM HERE

        # Testing purposes
            # Force the car to start in a different lane
        # if self.shifts < 1:
        #     self.shifts += 1
        #     action = self.queue_action(["STEER_LEFT"]*60)
        #     action = self.queue_action(["NOTHING"])
        #     return self.pop_next_action()
        
        # self.get_to_lane(5, state, brake = True) # Get to lane 5 (rightmost lane) to start with
        # action = self.queue_action(["NOTHING"]*10)
        # return self.pop_next_action()
        # Kill the car to hide score
        # if state["distance"] > 23000:
        #     self.queue_action(["STEER_LEFT"]*1000)
        #     return self.pop_next_action()
        
        
        if not state["sensors"]["front"]:
            self.queue_action("ACCELERATE")
            return self.pop_next_action()


        # Determine whether the car can see a car in neighboring lanes
        right_1 = self.detect_car_in_neighboring_lane(state)["right_1"]
        right_2 = self.detect_car_in_neighboring_lane(state)["right_2"]
        left_1 = self.detect_car_in_neighboring_lane(state)["left_1"]
        left_2 = self.detect_car_in_neighboring_lane(state)["left_2"]
        # print(f"right_1: {right_1}, right_2: {right_2}, left_1: {left_1}, left_2: {left_2}")
        #state["sensors"]["front"] < 700 and
        if not self.safe_to_shift(state=state, direction=None, v0=current_front_velocity, a = 0.05):
            self.queue_action("DECELERATE")
            return self.pop_next_action()
        
        # i.e. "See car, go: nono"
        if left_1 and not right_1 and self.lane < 5:
            self.queue_actions_to_position("right", state["velocity"]["y"])
            # print("Call 1")
            # print(f"DECIDED TO GO RIGHT, as left_1 is {left_1} and right_1 is {right_1}, lane: {self.lane}")
            return self.pop_next_action()

        if right_1 and not left_1 and self.lane > 1:
            self.queue_actions_to_position("left", state["velocity"]["y"])
            # print("Call 2")
            # print(f"DECIDED TO GO LEFT, as left_1 is {left_1} and right_1 is {right_1}, lane: {self.lane}")
            return self.pop_next_action()
        if right_1 and left_1: 
            self.queue_action("DECELERATE")
            # print("Call 3")
            # print(f"DECIDED TO DECELERATE, as left_1 is {left_1} and right_1 is {right_1}, lane: {self.lane}")
            return self.pop_next_action()

        # If both directions seem good, go towards the middle
        unsafe = right_1 if self.lane < 4 else left_1
        if unsafe:
            self.queue_action("NOTHING")
            return self.pop_next_action()

        self.queue_actions_to_position(224 if self.lane == 3 else np.sign(3 - self.lane) * 224, state["velocity"]["y"], True) # Cool way of writing "go towards the middle lane (right if you're already in the middle)"
        # print("Call 4")
        return self.pop_next_action()
