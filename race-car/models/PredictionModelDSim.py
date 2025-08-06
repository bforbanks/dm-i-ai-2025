"""THIS IS NOT A MODEL, IT IS A CLASS USED TO PREDICT THE POSITIONS 
AND VELOCITIES OF OTHER CARS"""

import numpy as np
import matplotlib.pyplot as plt
import time
class Lane:  
    def __init__(self, lane_number: int):
        self.has_car = False
        self.car_distance: float
        self.car_velocity: float
        self.lane_number = lane_number
    def update_lane(self, ego_distance: float, available_lanes: list[int]) -> int:
        if self.has_car:
            self.car_distance += self.car_velocity
            if self.car_distance-ego_distance < -1620 or self.car_distance-ego_distance > 1980:
                self.has_car = False
                available_lanes.append(self.lane_number)
                return -1
            self.car_velocity += np.random.uniform(-0.1,0.1)
            return 0
        return 0

    def spawn_car(self, ego_distance: float, ego_velocity: float):
        assert not self.has_car
        self.has_car = True
        self.car_velocity = np.random.uniform(-5, 5)
        if self.car_velocity < 0:
            self.car_distance = ego_distance + 1600
        else:
            self.car_distance = ego_distance - 1600
        self.car_velocity += ego_velocity

class Game:
    width = 3600
    height = 1200
    road_height = 1200 - 2*40
    lane_count = 5
    car_height, car_width = 179, 360

    def __init__(self, no_bins: int = 20, start_state: dict = None):
        if start_state:
            self.ego_distance = start_state["distance"]
            self.ego_velocity = start_state["velocity"]["x"]
            self.ego_y = start_state["velocity"]["y"]
            self.lanes = [Lane(_) for _ in range(self.lane_count)]
            self.available_lanes = [i for i in range(self.lane_count)]
            self.no_bins = no_bins 
        else:
            self.lanes = [Lane(_) for _ in range(self.lane_count)]
            self.available_lanes = [i for i in range(self.lane_count)]
            self.no_bins = no_bins 
            self.cars_spawned = 0
            self.ego_distance = 0


    def update_self(self, ego_distance: float, ego_velocity: float):
        for lane in self.lanes:
            self.cars_spawned += lane.update_lane(ego_distance, self.available_lanes)
        self.ego_distance = ego_distance
        if self.cars_spawned < 4:
            lane = np.random.choice(self.available_lanes)
            self.lanes[lane].spawn_car(self.ego_distance, ego_velocity)
            self.cars_spawned += 1
            self.available_lanes.remove(lane)


    def update(self, ego_distance: float, ego_velocity: float, sensor_info: list[list[dict]]):
        self.update_sensor(sensor_info)
        self.update_self(ego_distance, ego_velocity)

    def return_state(self, ego_distance: float):
        return [
            {"distance": lane.car_distance-ego_distance, "velocity": lane.car_velocity} if lane.has_car else {"distance": -2000, "velocity": -2000} for lane in self.lanes
        ]




class PredictionModelDSim:
    sensor_options = [(90, "front"),(135, "right_front"),(180, "right_side"),(225, "right_back"),(270, "back"),(315, "left_back"),(0, "left_side"),(45, "left_front"),(22.5, "left_side_front"),(67.5, "front_left_front"),(112.5, "front_right_front"),(157.5, "right_side_front"),(202.5, "right_side_back"),(247.5, "back_right_back"),(292.5, "back_left_back"),(337.5, "left_side_back")]
    name_to_degree = {name: degree for degree, name in sensor_options}
    lane_y_coordinates = [(62,62+181), (286,286+181), (510,510+181), (734,734+181), (958,958+181)]

    def __init__(self, sim_count: int):
        self.games = [Game() for _ in range(sim_count)]
        self.ego_distance = 0
        self.ego_velocity = 10
        self.ego_y = 510
        self.previous_info = [{} for _ in range(5)]
        self.sim_count = sim_count

    def progress_tick(self, state):
        self.ego_distance = state["distance"]
        self.ego_x_velocity, self.ego_y_velocity = state["velocity"]["x"], state["velocity"]["y"]
        self.ego_y += self.ego_y_velocity
        for game in self.games:
            game.update_self(self.ego_distance, self.ego_velocity)

    def parse_sensors(self, state: dict):
        """Returns a list of informations, on about each lane. So a list of 5 lists of dictionaries (these dictionaries are informations about the cars in the lane)"""
        information = [[] for _ in range(5)]
        for degree, name in self.sensor_options:
            spotted_something = True
            detected_value = state["sensors"][name]
            if detected_value:
                x, y = np.cos((degree-90)/180*np.pi)*state["sensors"][name]+180, np.sin((degree-90)/180*np.pi)*state["sensors"][name]+self.ego_y+90
                if 50 <= y <= 1150:
                    for i, (lo, hi) in enumerate(self.lane_y_coordinates):
                        if lo <= y <= hi:
                            if abs(y-lo) < 3 or abs(y-hi) < 3: # Inexact
                                information[i].append({})
                                information[i][-1]["type"] = "inexact"
                                information[i][-1]["x"] = (x-360,x)
                            else: # exact
                                information[i].append({})
                                information[i][-1]["type"] = "exact"
                                information[i][-1]["x"] = (x, x) if "front" in name else (x-360, x-360)
                            if self.previous_info[i]: # update velocity if possible.
                                for prev_information in self.previous_info[i]:
                                    if prev_information["type"] != "no_car":
                                        information[i][-1]["velocity"] = (information[i][-1]["x"][0]-prev_information["x"][1]), (information[i][-1]["x"][1]-prev_information["x"][0])
                                        break
                else: # if no value detected, that is also information!
                    spotted_something = False
            else: # if no value detected, that is also information!
                spotted_something = False

            if not spotted_something: # if no value detected, that is also information!
                for i, (lo, hi) in enumerate(self.lane_y_coordinates): #s
                    information[i].append({})
                    information[i][-1]["type"] = "no_car"
                    if name == "behind":
                        if lo+3 < self.ego_y < hi-3:
                            information[i][-1]["x"] = (-1000,0)
                            break
                        else:
                            information[i].pop(-1)
                            break
                    elif name == "front":
                        if lo+3 < self.ego_y < hi-3:
                            information[i][-1]["x"] = (0,1000)
                            break
                        else:
                            information[i].pop(-1)
                            break
                    
                    if ("right" in name and lo > self.ego_y) or ("left" in name and hi < self.ego_y):
                        information[i].pop(-1)
                        continue
                    if lo > self.ego_y:
                        y_diff = self.ego_y - lo
                    elif hi < self.ego_y:
                        y_diff = self.ego_y - hi
                    else:
                        information[i].pop(-1)
                        continue
        
                    supposed_spot = y_diff/np.tan((degree-90)/180*np.pi)
                    if supposed_spot**2 + y_diff**2 > 1000000:
                        information[i].pop(-1)
                        continue
                    information[i][-1]["x"] = (supposed_spot-360, supposed_spot)

        self.previous_info = information
        return information

    def update_sensor(self, sensor_dict: dict):
        for i, lane_info in enumerate(sensor_dict):
            for information in lane_info:
                # if information["type"] == "inexact":
                #     surviving_games = []
                #     for game in self.games:
                #         if not game.lanes[i].has_car:
                #             continue
                #         if not (information["x"][0] < game.lanes[i].car_distance < information["x"][1]):
                #             continue
                #         if information.get("velocity", None) and not (information["velocity"][0] < game.lanes[i].car_velocity < information["velocity"][1]):
                #             continue
                #         surviving_games.append(game)
                #     self.games = surviving_games

                # if information["type"] == "exact":
                #     surviving_games = []
                #     for game in self.games:
                #         if not game.lanes[i].has_car:
                #             continue
                #         if game.lanes[i].car_distance < information["x"][0] - 180 or game.lanes[i].car_distance > information["x"][1] + 540:
                #             continue
                        
                #         game.lanes[i].car_distance = information["x"][0]
                #         if information.get("velocity", None):
                #             if not (information["velocity"][0] - 5 < game.lanes[i].car_velocity < information["velocity"][1] + 5):
                #                 continue
                #             else:
                #                 game.lanes[i].car_velocity = np.random.uniform(information["velocity"][0], information["velocity"][1])
                        
                #         surviving_games.append(game)
                #     self.games = surviving_games

                if information["type"] == "no_car":
                    self.games = [
                        game for game in self.games
                        if not (
                            game.lanes[i].has_car and
                            game.lanes[i].car_distance > information["x"][0] and
                            game.lanes[i].car_distance < information["x"][1]
                        )
                    ]

        # Copy existing games to fill up to sim_count by np.random.choice
        print("Removed games:", self.sim_count - len(self.games))
        if self.sim_count - len(self.games) > 1000:
            print("ahhh")
        if self.games and len(self.games) < self.sim_count:
            self.games.extend(np.random.choice(self.games, self.sim_count - len(self.games)))


    def visualize_states(self):
        plt.clf()
        distances=[]
        for state in [_.return_state(self.ego_distance) for _ in self.games]:
            distances.append(state[0]["distance"])
        plt.hist(distances, bins=100)
        plt.pause(0.001)

    def return_action(self, state):
        self.visualize_states()
        self.progress_tick(state)
        sensor_info = self.parse_sensors(state)
        self.update_sensor(sensor_info)
        if state["elapsed_ticks"] < 10:
            return ["STEER_LEFT"]
        else:
            return ["NOTHING"]
    
def main():
    model = PredictionModelDSim(10000)
    plt.ion()
    state = {"distance": 0, "velocity": {"x": 10, "y": 0}}
    for i in range(1000):
        state["distance"] += state["velocity"]["x"]
        model.visualize_states()
        model.progress_tick(state)

if __name__ == "__main__":
    main()