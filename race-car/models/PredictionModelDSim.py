"""THIS IS NOT A MODEL, IT IS A CLASS USED TO PREDICT THE POSITIONS 
AND VELOCITIES OF OTHER CARS"""

import numpy as np
import matplotlib.pyplot as plt
import time
import copy
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

    def __init__(self):
        self.lanes = [Lane(_) for _ in range(self.lane_count)]
        self.available_lanes = [i for i in range(self.lane_count)]
        self.cars_spawned = 0

    def update_self(self, ego_distance: float, ego_velocity: float):
        for lane in self.lanes:
            self.cars_spawned += lane.update_lane(ego_distance, self.available_lanes)
        if self.cars_spawned < 4:
            lane = np.random.choice(self.available_lanes)
            self.lanes[lane].spawn_car(ego_distance, ego_velocity)
            self.cars_spawned += 1
            self.available_lanes.remove(lane)

    def return_state(self, ego_distance: float):
        return [
            {"distance": lane.car_distance-ego_distance, "velocity": lane.car_velocity} if lane.has_car else {"distance": -2000, "velocity": -2000} for lane in self.lanes
        ]



class PredictionModelDSim:
    sensor_options = [(90, "front"),(135, "right_front"),(180, "right_side"),(225, "right_back"),(270, "back"),(315, "left_back"),(0, "left_side"),(45, "left_front"),(22.5, "left_side_front"),(67.5, "front_left_front"),(112.5, "front_right_front"),(157.5, "right_side_front"),(202.5, "right_side_back"),(247.5, "back_right_back"),(292.5, "back_left_back"),(337.5, "left_side_back")]
    name_to_degree = {name: degree for degree, name in sensor_options}
    NPC_car_y_coordinate_ranges = [(62,241), (286,465), (510,689), (734,913), (958,1137)]

    def __init__(self, sim_count: int, live_visualization: bool = False):
        self.games = [Game() for _ in range(sim_count)]
        self.ego_distance = 0
        self.ego_velocity = 10
        self.ego_y = 510
        self.previous_info = [{} for _ in range(5)]
        self.sim_count = sim_count
        self.live_visualization = live_visualization
        if live_visualization:
            plt.ion()
            self.fig, self.axs = plt.subplots(5, 1)

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
                x, y = np.cos((degree-90)/180*np.pi)*state["sensors"][name]+state["distance"]+180, np.sin((degree-90)/180*np.pi)*state["sensors"][name]+self.ego_y+90
                if 50 <= y <= 1150:
                    for i, (lo, hi) in enumerate(self.NPC_car_y_coordinate_ranges):
                        if lo-3 <= y <= hi+3: #buffer as there are some inaccuracies
                            if y-lo < 3 or hi-y < 3: # Inexact
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
                for i, (lo, hi) in enumerate(self.NPC_car_y_coordinate_ranges): #s
                    information[i].append({})
                    information[i][-1]["type"] = "no_car"
                    if name == "back":
                        if lo+2 < self.ego_y+90 < hi-2:
                            information[i][-1]["x"] = (-1000+state["distance"]+180,180+state["distance"])
                            break
                        else:
                            information[i].pop(-1)
                            break
                    elif name == "front":
                        if lo+2 < self.ego_y+90 < hi-2:
                            information[i][-1]["x"] = (180+state["distance"],1180+state["distance"])
                            break
                        else:
                            information[i].pop(-1)
                            break
                    
                    if ("right" in name and lo > self.ego_y) or ("left" in name and hi < self.ego_y):
                        information[i].pop(-1)
                        continue
                    if lo > self.ego_y:
                        y_diff = lo - self.ego_y+ 179
                    elif hi < self.ego_y:
                        y_diff = hi - self.ego_y
                    else:
                        information[i].pop(-1)
                        continue
        
                    supposed_spot_distance = -y_diff/np.tan((degree-90)/180*np.pi)+state["distance"]+180
                    if (supposed_spot_distance-state["distance"])**2 + y_diff**2 > 1000000:
                        information[i].pop(-1)
                        continue
                    information[i][-1]["x"] = (supposed_spot_distance-360, supposed_spot_distance)

        self.previous_info = information
        return information

    def update_sensor(self, sensor_dict: dict, state: dict, internal_state: dict, ego_distance: float, ego_y: float):
        old_games = copy.deepcopy(self.games)
        for i, lane_info in enumerate(sensor_dict):
            for information in lane_info:
                if information["type"] == "inexact":
                    surviving_games = []
                    games_to_modify = []
                    for game in self.games:
                        if not game.lanes[i].has_car:
                            continue
                        if not (information["x"][0] - 180 < game.lanes[i].car_distance < information["x"][1] + 180):
                            games_to_modify.append(game)
                            continue
                        if information.get("velocity", None) and not (information["velocity"][0]-5 < game.lanes[i].car_velocity < information["velocity"][1]+5):
                            games_to_modify.append(game)
                            continue
                        surviving_games.append(game)
                    for game in games_to_modify:
                        random_suitable_game = np.random.choice(surviving_games)
                        game.lanes[i] = copy.deepcopy(random_suitable_game.lanes[i])
                        surviving_games.append(game)
                    self.games = surviving_games

                elif information["type"] == "exact":
                    surviving_games = []
                    games_to_modify = []
                    for game in self.games:
                        if not game.lanes[i].has_car:
                            continue
                        game.lanes[i].car_distance = information["x"][0]
                        if information.get("velocity", None) and (abs(information["velocity"][0] - information["velocity"][1]) < 0.1):
                            game.lanes[i].car_velocity = information["velocity"][0]
                        surviving_games.append(game)
                    self.games = surviving_games

                elif information["type"] == "no_car":
                    self.games = [
                        game for game in self.games
                        if not (
                            game.lanes[i].has_car and
                            game.lanes[i].car_distance > information["x"][0]+50 and
                            game.lanes[i].car_distance < information["x"][1]-50 and
                            True
                        )
                    ]

        # Copy existing games to fill up to sim_count by np.random.choice
        print("Removed games:", self.sim_count - len(self.games))
        if self.sim_count - len(self.games) > 4000:
            print("sensor info (distance coordinates):", sensor_dict, "state (distance coordinates):", state, "internal_state (pixel coordinates - ground truth):", vars(internal_state), "ego_distance (self-calucated):", ego_distance, "ego_y (self-calucated):", ego_y)
            for car in internal_state.cars:
                print("One car is (pixel coordinates):")
                print(vars(car), vars(car.lane))
                print("--------------------------------")
            # make histograms and write out the x-value of any bin with more than 2000 cars.
            fig, axs = plt.subplots(5, 1, figsize=(8, 10))
            x_values = []
            bin_width = (2100 - (-2100)) / 200  # Calculate bin width
            for i in range(5):
                distances = [game.return_state(ego_distance)[i]["distance"] for game in old_games]
                counts, bin_edges = np.histogram(distances, bins=200, range=(-2100, 2100))
                axs[i].hist(distances, bins=200, range=(-2100, 2100))
                axs[i].set_xlim(-2100, 2100)
                axs[i].set_title(f'Lane {i}')
                for j in range(200):
                    if counts[j] > 2000:
                        bin_center = bin_edges[j] + bin_width / 2  # Calculate bin center
                        x_values.append((i, bin_center))
            print("x_values (lane, distance):", x_values)
            plt.tight_layout()
            plt.savefig("histogram.png")
            plt.close()
            time.sleep(10000)
        if self.games and len(self.games) < self.sim_count:
            new_games = np.random.choice(self.games, self.sim_count - len(self.games))
            self.games.extend([copy.deepcopy(g) for g in new_games])

    def visualize_states(self):
        for ax in self.axs:
            ax.cla()
        all_states = [g.return_state(self.ego_distance) for g in self.games]
        for i in range(5):
            distances = [state[i]["distance"] for state in all_states]
            self.axs[i].hist(distances, bins=200, range=(-2100, 2100))
            self.axs[i].set_xlim(-2100, 2100)
        plt.pause(0.001)

    def return_action(self, state, internal_state):
        if self.live_visualization:
            self.visualize_states()
        self.progress_tick(state)
        sensor_info = self.parse_sensors(state)
        self.update_sensor(sensor_info, state, internal_state, self.ego_distance, self.ego_y)
        if state["elapsed_ticks"] < 10:
            return ["STEER_LEFT"]
        else:
            return ["NOTHING"]
    
def main():
    model = PredictionModelDSim(10000)
    state = {"distance": 0, "velocity": {"x": 10, "y": 0}}
    for i in range(1000):
        state["distance"] += state["velocity"]["x"]
        # model.visualize_states()
        model.progress_tick(state)

if __name__ == "__main__":
    main()