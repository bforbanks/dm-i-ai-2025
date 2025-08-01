class Playground:
    def __init__(self):
        self.lanes = {1: None, 2: None, 3: None, 4: None, 5: None}
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
        ] # Variable
    def update_position(self, sensors):
        for sensor in self.sensor_options:
            if sensors[sensor] is not None:


