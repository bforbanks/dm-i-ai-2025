import numpy as np
from typing import List, Tuple

class LaneShift:

    def __init__(self):
        height = 1200
        margins = 40
        usable_height = height - margins * 2
        self.lane_height = usable_height / 5
        
        # Starting lane is 3 (middle)
        self.lane = 3
        self.last_distance_front = None

        # In general, front is 0 and reference point. Left is negative, right is positive.
        self.sensor_dict = {
                           0: "front",
                          -1: "front_right_front",
                          -2: "right_front",
                          -3: "right_side_front",
                          -4: "right_side",
                          -5: "right_side_back",
                          -6: "right_back",
                          -7: "back_right_back",
                           1: "front_left_front",
                           2: "left_front",
                           3: "left_side_front",
                           4: "left_side",
                           5: "left_side_back",
                           6: "left_back",
                           7: "back_left_back"
                           }

        # Distances from car sensor to boundaries of car in a given lane. Number is sensor type,
        # first tuple is interval of one lane, second tuple is interval of the next lane.
        self.sensor_type_to_lane = {1: ((345, 830), (940, 950)),
                                 2: ((180, 460), (500, 770)),
                                 3: ((140, 350), (380, 600)),
                                 4: ((125, 135), (350, 360))}
        
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

    def update_ypos(self, action: list[str]) -> None:
        pass

    def actions_to_position(self, distance: float | str, v0, brake = True):
        if isinstance(distance, str):
            distance = 224 if distance == "right" else -224
        self.lane += (1 if distance == 224 else -1)
        action_dict = {-1: "STEER_LEFT", 0: "NOTHING", 1: "STEER_RIGHT"}
        
        abs_distance, direction = abs(distance), np.sign(distance)
        b = 20*v0 - 1; c = -20 * abs_distance
        if brake:
            c /= 2
        # Solve the quadratic equation for ticks
        ticks = int((-b + np.sqrt(b**2 - 4*c)) / 2)
        actions = [action_dict[direction]]*ticks + [action_dict[-direction]]*(ticks * brake)

        
        return actions
    
    def determine_velocity_front(self, state: dict) -> float:
        diff = state["sensors"]["front"] - self.last_distance_front if (self.last_distance_front and state["sensors"]["front"]) else None

    def return_action(self, state: dict) -> List[str]:        
        if not state["sensors"]["front"]:
            return ["ACCELERATE"]*3
        

        right_1 = self.detect_car_in_neighboring_lane(state)["right_1"]
        left_1 = self.detect_car_in_neighboring_lane(state)["left_1"]
        
        # if state["sensors"]["front"] < 400:
        #     return ["DECELERATE"]
        
        # i.e. "See car, go: nono"
        if left_1 and not right_1 and self.lane < 5:
            return self.actions_to_position("right", state["velocity"]["y"])
        if right_1 and not left_1 and self.lane > 1:
            return self.actions_to_position("left", state["velocity"]["y"])
        if right_1 and left_1: 
            return ["DESCELERATE"]*2
        return self.actions_to_position(224 if self.lane == 3 else np.sign(3 - self.lane) * 224, state["velocity"]["y"], True)

