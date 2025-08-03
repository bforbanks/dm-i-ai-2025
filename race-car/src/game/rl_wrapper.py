import pygame
import torch
# from time import sleep

#import requests
#from typing import List, Optional
from ..mathematics.randomizer import seed, random_choice, random_number
from ..elements.car import Car
from ..elements.road import Road
from ..elements.sensor import Sensor
from ..mathematics.vector import Vector
import json
from typing import List


from src.game.core import GameState
class RaceCarEnv:
    def __init__(self,
                 api_url,
                 seed_value=None,
                 sensor_removal=0,
                 verbose: bool = False,
                 render: bool = False,
                 SCREEN_WIDTH = 1600,
                 SCREEN_HEIGHT = 1200,
                 LANE_COUNT = 5,
                 CAR_COLORS = ['yellow', 'blue', 'red'],
                 MAX_TICKS = 60 * 60,  # 60 seconds @ 60 fps
                 MAX_MS = 60 * 1000600,   # 60 seconds flat
                 ):
        self.api_url = api_url
        self.seed_value = seed_value
        self.sensor_removal = sensor_removal
        self.done = False
    


        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.LANE_COUNT = LANE_COUNT
        self.CAR_COLORS = CAR_COLORS
        self.MAX_TICKS = MAX_TICKS
        self.MAX_MS = MAX_MS

        self.verbose = verbose
        self.render = render
        if self.render:
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("Race Car Game")
        else:
            pygame.display.quit()
        

        self.initialize_game_state(self.api_url, self.seed_value, self.sensor_removal)
    
    
    def initialize_game_state(self, api_url: str, seed_value: str, sensor_removal = 0):
        seed(seed_value)
        self.STATE = GameState(api_url=api_url)

        # Create environment
        self.STATE.road = Road(self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.LANE_COUNT)
        self.middle_lane = self.STATE.road.middle_lane()
        self.lane_height = self.STATE.road.get_lane_height()

        # Create ego car
        self.ego_velocity = Vector(10, 0)
        self.STATE.ego = Car("yellow", self.ego_velocity, lane=self.middle_lane, target_height=int(self.lane_height * 0.8))
        self.ego_sprite = self.STATE.ego.sprite
        self.STATE.ego.x = (self.SCREEN_WIDTH // 2) - (self.ego_sprite.get_width() // 2)
        self.STATE.ego.y = int((self.middle_lane.y_start + self.middle_lane.y_end) / 2 - self.ego_sprite.get_height() / 2)
        self.sensor_options = [
            (90, "front"),
            (135, "right_front"),
            (180, "right_side"),
            (225, "right_back"),
            (270, "back"),
            (315, "left_back"),
            (0, "left_side"),
            (45, "left_front"),
            (22.5, "left_side_front"),
            (67.5, "front_left_front"),
            (112.5, "front_right_front"),
            (157.5, "right_side_front"),
            (202.5, "right_side_back"),
            (247.5, "back_right_back"),
            (292.5, "back_left_back"),
            (337.5, "left_side_back"),
        ]

        for _ in range(sensor_removal): # Removes random sensors
            random_sensor = random_choice(self.sensor_options)
            self.sensor_options.remove(random_sensor)
        self.STATE.sensors = [
            Sensor(self.STATE.ego, angle, name, self.STATE)
            for angle, name in self.sensor_options
        ]

        # Create other cars and add to car bucket
        for i in range(0, self.LANE_COUNT - 1):
            car_colors = ["blue", "red"]
            color = random_choice(car_colors)
            car = Car(color, Vector(8, 0), target_height=int(self.lane_height * 0.8))
            self.STATE.car_bucket.append(car)

        self.STATE.cars = [self.STATE.ego]

    def handle_action(self, actions: str | List):
        if isinstance(actions, list):
            action = actions.pop(0) if actions else "NOTHING"
        else: action = actions
        
        if action == "ACCELERATE":
            self.STATE.ego.speed_up()
        elif action == "DECELERATE":
            self.STATE.ego.slow_down()
        elif action == "STEER_LEFT":
            self.STATE.ego.turn(-0.1)
        elif action == "STEER_RIGHT":
            self.STATE.ego.turn(0.1)
        else:
            pass

    def update_cars(self):
        for car in self.STATE.cars:
            car.update(self.STATE.ego)

    def remove_passed_cars(self):
        min_distance = -1000
        max_distance = self.SCREEN_WIDTH + 1000
        cars_to_keep = []
        cars_to_retire = []

        for car in self.STATE.cars:
            if car.x < min_distance or car.x > max_distance:
                cars_to_retire.append(car)
            else:
                cars_to_keep.append(car)

        for car in cars_to_retire:
            self.STATE.car_bucket.append(car)
            car.lane = None

        self.STATE.cars = cars_to_keep

    def place_car(self):
        if len(self.STATE.cars) > self.LANE_COUNT:
            return

        speed_coeff_modifier = 5
        x_offset_behind = -0.5
        x_offset_in_front = 1.5

        open_lanes = [lane for lane in self.STATE.road.lanes if not any(c.lane == lane for c in self.STATE.cars if c != self.STATE.ego)]
        lane = random_choice(open_lanes)
        x_offset = random_choice([x_offset_behind, x_offset_in_front])
        horizontal_velocity_coefficient = random_number() * speed_coeff_modifier

        car = self.STATE.car_bucket.pop() if self.STATE.car_bucket else None
        if not car:
            return

        velocity_x = self.STATE.ego.velocity.x + horizontal_velocity_coefficient if x_offset == x_offset_behind else self.STATE.ego.velocity.x - horizontal_velocity_coefficient
        car.velocity = Vector(velocity_x, 0)
        self.STATE.cars.append(car)

        car_sprite = car.sprite
        car.x = (self.SCREEN_WIDTH * x_offset) - (car_sprite.get_width() // 2)
        car.y = int((lane.y_start + lane.y_end) / 2 - car_sprite.get_height() / 2)
        car.lane = lane

    def intersects(self, rect1, rect2):
        return rect1.colliderect(rect2)

    def update_game(self, current_action: str):
        self.handle_action(current_action)
        self.STATE.distance += self.STATE.ego.velocity.x
        self.update_cars()
        self.remove_passed_cars()
        self.place_car()
        for sensor in self.STATE.sensors:
            sensor.update()
        # Handle collisions
        for car in self.STATE.cars:
            if car != self.STATE.ego and self.intersects(self.STATE.ego.rect, car.rect):
                self.STATE.crashed = True
        
        # Check collision with walls
        for wall in self.STATE.road.walls:
            if self.intersects(self.STATE.ego.rect, wall.rect):
                self.STATE.crashed = True

        # Render game (only if verbose) TODO: Implement rendering
        # if self.verbose:
        #     self.screen.fill((0, 0, 0))  # Clear the screen with black

        #     # Draw the road background
        #     self.screen.blit(self.STATE.road.surface, (0, 0))

        #     # Draw all walls
        #     for wall in self.STATE.road.walls:
        #         wall.draw(screen)

        #     # Draw all cars
        #     for car in self.STATE.cars:
        #         if car.sprite:
        #             screen.blit(car.sprite, (car.x, car.y))
        #             bounds = car.get_bounds()
        #             color = (255, 0, 0) if car == self.STATE.ego else (0, 255, 0)
        #             pygame.draw.rect(screen, color, bounds, width=2)
        #         else:
        #             pygame.draw.rect(screen, (255, 255, 0) if car == self.STATE.ego else (0, 0, 255), car.rect)

        #     # Draw sensors if enabled
        #     if STATE.sensors_enabled:
        #         for sensor in STATE.sensors:
        #             sensor.draw(screen)

            # pygame.display.flip()

    def reset(self):
        self.initialize_game_state(api_url=self.api_url, seed_value=self.seed_value, sensor_removal=self.sensor_removal)
        self.done = False
    
    def step(self, action):        
        # if self.render:
        #     delta = self.clock.tick(60)
        # else:
        #     delta = delta = 16  # Approx. 60 FPS time delta in ms

        # self.STATE.elapsed_game_time += delta
        self.STATE.ticks += 1

        self.update_game(action)    
        # Define your own reward logic here
        reward = self._get_reward()

        # Define your own termination condition
        done = self.STATE.crashed or self.STATE.ticks > 3600 #or self.STATE.elapsed_game_time > 60000  # for example

        state_to_return = self.state_to_state_dict(self.STATE)

        if done:
            self.reset()  # Reset the environment if done

        return state_to_return, reward, done
    
    def _get_reward(self):
        if self.STATE.crashed:
            return -100
        return self.STATE.ego.velocity.x / 10
    
    def state_to_state_dict(self, state: GameState):
        """The state in this file is a GameState object. This function converts it
        to a dictionary, in the same way as the state we receive from the API."""
        
        # Build sensors dictionary by name
        sensors_dict = {}
        for sensor in state.sensors:
            sensors_dict[sensor.name] = sensor.reading
        
        return {
            "did_crash": state.crashed,
            "elapsed_ticks": state.ticks,
            "distance": state.distance,
            "velocity": {
                "x": state.ego.velocity.x,
                "y": state.ego.velocity.y
            },
            "sensors": sensors_dict
        }
