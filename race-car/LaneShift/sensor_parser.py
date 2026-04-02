import numpy as np

class SensorParser:
    sensor_options = [(90, "front"),(135, "right_front"),(180, "right_side"),(225, "right_back"),(270, "back"),(315, "left_back"),(0, "left_side"),(45, "left_front"),(22.5, "left_side_front"),(67.5, "front_left_front"),(112.5, "front_right_front"),(157.5, "right_side_front"),(202.5, "right_side_back"),(247.5, "back_right_back"),(292.5, "back_left_back"),(337.5, "left_side_back")]
    name_to_degree = {name: degree for degree, name in sensor_options}
    NPC_car_y_coordinate_ranges = [(62,241), (286,465), (510,689), (734,913), (958,1137)]

    def __init__(self):
        self.previous_info = [[] for _ in range(5)]

    def parse_sensors(self, state: dict, y_pos):
        """MUST BE RUN ONCE PER TICK. OTHERWISE IT CRIES. Returns a list of informations, on about each lane. So a list of 5 lists of dictionaries (these dictionaries are informations about the cars in the lane)"""
        y_pos = y_pos + 510
        information = [[] for _ in range(5)]
        for degree, name in self.sensor_options:
            spotted_something = True
            detected_value = state["sensors"][name]
            if detected_value:
                x, y = np.cos((degree-90)/180*np.pi)*state["sensors"][name], np.sin((degree-90)/180*np.pi)*state["sensors"][name]+y_pos+90
                
                if 50 <= y <= 1150:
                    for i, (lo, hi) in enumerate(self.NPC_car_y_coordinate_ranges):
                        if lo-3 <= y <= hi+3: #buffer as there are some inaccuracies
                            if y-lo < 3 or hi-y < 3: # Inexact
                                information[i].append({})
                                information[i][-1]["type"] = "inexact"
                                information[i][-1]["x"] = x
                            else: # exact
                                information[i].append({})
                                information[i][-1]["type"] = "exact"
                                information[i][-1]["x"] = x+180 if "front" in name else x-180
                                if self.previous_info[i]: # update velocity if possible.
                                    for prev_information in self.previous_info[i]:
                                        if prev_information["type"] == "exact":
                                            information[i][-1]["velocity"] = information[i][-1]["x"]-prev_information["x"]
        self.previous_info = information
        parsed_info = []
        for lane_info in information:
            exact = [info for info in lane_info if info["type"] == "exact"]
            if exact:
                velocity = exact[0].get("velocity", None)
                parsed_info.append({"velocity": velocity, "x": exact[0]["x"], "type": "exact"})
            elif lane_info:
                parsed_info.append({"velocity": None, "x": np.mean([info["x"] for info in lane_info if info["type"] == "inexact"]), "type": "inexact"})
            else:
                parsed_info.append({"velocity": None, "x": None, "type": None})

        return parsed_info