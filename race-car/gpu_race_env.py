import torch
from typing import Dict, Tuple
import math


class GPURaceEnvironment:
    """
    GPU-accelerated parallel race car environment for NEAT training.
    Can simulate hundreds of cars simultaneously on GPU.
    """

    def __init__(
        self,
        batch_size: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_ticks: int = 3600,
        screen_width: int = 1600,
        screen_height: int = 1200,
        lane_count: int = 5,
        action_repeat: int = 3,  # How many ticks each action persists
    ):
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.max_ticks = max_ticks
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.lane_count = lane_count
        self.action_repeat = action_repeat

        # Physics constants
        self.acceleration_amount = 0.1
        self.deceleration_amount = 0.1
        self.steering_amount = 0.1
        self.sensor_range = 1000.0

        # Initialize lanes
        self.lane_height = screen_height / lane_count
        self.lane_centers = torch.linspace(
            self.lane_height / 2,
            screen_height - self.lane_height / 2,
            lane_count,
            device=self.device,
        )

        # Sensor angles (in radians) - 16 sensors as in original
        self.sensor_angles = (
            torch.tensor(
                [
                    0,
                    22.5,
                    45,
                    67.5,
                    90,
                    112.5,
                    135,
                    157.5,
                    180,
                    202.5,
                    225,
                    247.5,
                    270,
                    292.5,
                    315,
                    337.5,
                ],
                device=self.device,
            )
            * math.pi
            / 180
        )

        self.sensor_names = [
            "left_side",
            "left_side_front",
            "left_front",
            "front_left_front",
            "front",
            "front_right_front",
            "right_front",
            "right_side_front",
            "right_side",
            "right_side_back",
            "right_back",
            "back_right_back",
            "back",
            "back_left_back",
            "left_back",
            "left_side_back",
        ]

        # Initialize game state tensors
        self.reset()

    def reset(self) -> torch.Tensor:
        """Reset all environments and return initial state"""

        # Car positions (batch_size, 2) - [x, y]
        self.ego_positions = torch.zeros(self.batch_size, 2, device=self.device)
        self.ego_positions[:, 0] = self.screen_width // 2  # Start in middle x
        self.ego_positions[:, 1] = self.screen_height // 2  # Start in middle y

        # Car velocities (batch_size, 2) - [vx, vy]
        self.ego_velocities = torch.zeros(self.batch_size, 2, device=self.device)
        self.ego_velocities[:, 0] = 10.0  # Initial forward velocity

        # Game state
        self.ticks = torch.zeros(self.batch_size, device=self.device, dtype=torch.int32)
        self.distances = torch.zeros(self.batch_size, device=self.device)
        self.crashed = torch.zeros(
            self.batch_size, device=self.device, dtype=torch.bool
        )
        self.done = torch.zeros(self.batch_size, device=self.device, dtype=torch.bool)

        # Action persistence state
        self.current_actions = torch.zeros(
            self.batch_size, device=self.device, dtype=torch.long
        )
        self.action_ticks_remaining = torch.zeros(
            self.batch_size, device=self.device, dtype=torch.int32
        )

        # Other cars (simplified - we'll add a few static obstacles)
        self.initialize_obstacles()

        # Calculate initial sensor readings
        sensor_readings = self.calculate_sensors()

        return self.get_state_tensor(sensor_readings)

    def initialize_obstacles(self):
        """Initialize static obstacles for collision detection"""
        # Create some static obstacles in different lanes
        num_obstacles_per_env = 8

        # Obstacles: (batch_size, num_obstacles, 4) - [x, y, width, height]
        self.obstacles = torch.zeros(
            self.batch_size, num_obstacles_per_env, 4, device=self.device
        )

        # Place obstacles at various distances and lanes
        for i in range(num_obstacles_per_env):
            # Random x positions ahead of cars
            x_pos = torch.rand(self.batch_size, device=self.device) * 2000 + 800

            # Random lane selection
            lane_idx = torch.randint(
                0, self.lane_count, (self.batch_size,), device=self.device
            )
            y_pos = self.lane_centers[lane_idx]

            # Car dimensions
            width, height = 60, 40

            self.obstacles[:, i, 0] = x_pos
            self.obstacles[:, i, 1] = y_pos
            self.obstacles[:, i, 2] = width
            self.obstacles[:, i, 3] = height

        # Wall boundaries
        self.wall_top = 0
        self.wall_bottom = self.screen_height

    def step(
        self, new_actions: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step the environment with action persistence

        Args:
            new_actions: (batch_size,) action indices or None
                        Only used when action_ticks_remaining == 0
                        [0=NOTHING, 1=ACCELERATE, 2=DECELERATE, 3=STEER_LEFT, 4=STEER_RIGHT]

        Returns:
            next_state: (batch_size, state_dim)
            rewards: (batch_size,)
            done: (batch_size,)
            info: dict with additional 'needs_action' indicating which envs need new actions
        """

        # Don't update finished environments
        active_mask = ~self.done

        if active_mask.sum() == 0:
            # All environments are done
            sensor_readings = self.calculate_sensors()
            return (
                self.get_state_tensor(sensor_readings),
                torch.zeros_like(self.distances),
                self.done,
                {"needs_action": torch.zeros_like(self.done)},
            )

        # Handle action persistence
        needs_new_action = (self.action_ticks_remaining == 0) & active_mask

        if needs_new_action.any() and new_actions is not None:
            # Update actions for environments that need them
            self.current_actions[needs_new_action] = new_actions[needs_new_action]
            self.action_ticks_remaining[needs_new_action] = self.action_repeat
        elif needs_new_action.any() and new_actions is None:
            # No new actions provided but some environments need them - use NOTHING
            self.current_actions[needs_new_action] = 0  # NOTHING
            self.action_ticks_remaining[needs_new_action] = self.action_repeat

        # Apply current actions (whether new or continuing)
        self.apply_actions(self.current_actions, active_mask)

        # Decrement action ticks for active environments
        self.action_ticks_remaining[active_mask] -= 1

        # Update physics
        self.update_physics(active_mask)

        # Update obstacles (move them relative to ego car)
        self.update_obstacles(active_mask)

        # Check collisions
        self.check_collisions(active_mask)

        # Update game state
        self.ticks += active_mask.int()

        # Check termination conditions
        time_limit_exceeded = self.ticks >= self.max_ticks
        wall_collision = (self.ego_positions[:, 1] <= self.wall_top + 25) | (
            self.ego_positions[:, 1] >= self.wall_bottom - 25
        )

        self.crashed = self.crashed | wall_collision
        self.done = self.done | self.crashed | time_limit_exceeded

        # Calculate rewards
        rewards = self.calculate_rewards(active_mask)

        # Get sensor readings and state
        sensor_readings = self.calculate_sensors()
        next_state = self.get_state_tensor(sensor_readings)

        # Check which environments will need new actions next tick
        will_need_action = (self.action_ticks_remaining == 0) & (~self.done)

        info = {
            "distances": self.distances.clone(),
            "crashed": self.crashed.clone(),
            "ticks": self.ticks.clone(),
            "needs_action": will_need_action,
            "action_ticks_remaining": self.action_ticks_remaining.clone(),
        }

        return next_state, rewards, self.done, info

    def apply_actions(self, actions: torch.Tensor, active_mask: torch.Tensor):
        """Apply actions to active environments"""

        # actions is (batch_size,) indices: [0=NOTHING, 1=ACCELERATE, 2=DECELERATE, 3=STEER_LEFT, 4=STEER_RIGHT]

        # Extract action types using boolean masks
        accelerate = (actions == 1) & active_mask
        decelerate = (actions == 2) & active_mask
        steer_left = (actions == 3) & active_mask
        steer_right = (actions == 4) & active_mask

        # Apply velocity changes
        self.ego_velocities[:, 0] += accelerate.float() * self.acceleration_amount
        self.ego_velocities[:, 0] -= decelerate.float() * self.deceleration_amount
        self.ego_velocities[:, 0] = torch.clamp(
            self.ego_velocities[:, 0], 0, 25
        )  # Speed limits

        # Apply steering
        self.ego_velocities[:, 1] -= steer_left.float() * self.steering_amount
        self.ego_velocities[:, 1] += steer_right.float() * self.steering_amount
        self.ego_velocities[:, 1] = torch.clamp(
            self.ego_velocities[:, 1], -5, 5
        )  # Steering limits

    def update_physics(self, active_mask: torch.Tensor):
        """Update car positions based on velocities"""

        # Update positions
        self.ego_positions += self.ego_velocities * active_mask.unsqueeze(1)

        # Update distance traveled (only x-component counts)
        distance_delta = self.ego_velocities[:, 0] * active_mask
        self.distances += distance_delta

        # Apply some drag to steering
        self.ego_velocities[:, 1] *= 0.95

    def update_obstacles(self, active_mask: torch.Tensor):
        """Update obstacle positions relative to ego cars"""

        # Move obstacles leftward relative to ego car movement
        velocity_x = (
            self.ego_velocities[:, 0].unsqueeze(1).unsqueeze(2)
        )  # (batch, 1, 1)
        self.obstacles[:, :, 0] -= velocity_x.squeeze(2) * active_mask.unsqueeze(1)

        # Remove obstacles that are too far behind and add new ones ahead
        # (Simplified - just reset far obstacles to new positions ahead)
        too_far_behind = self.obstacles[:, :, 0] < -500

        if too_far_behind.any():
            # Reset these obstacles ahead
            reset_mask = too_far_behind & active_mask.unsqueeze(1)

            # New positions ahead
            new_x = torch.rand_like(self.obstacles[:, :, 0]) * 1500 + 1000
            new_lane_idx = torch.randint(
                0, self.lane_count, self.obstacles[:, :, 0].shape, device=self.device
            )
            new_y = self.lane_centers[new_lane_idx]

            self.obstacles[:, :, 0] = torch.where(
                reset_mask, new_x, self.obstacles[:, :, 0]
            )
            self.obstacles[:, :, 1] = torch.where(
                reset_mask, new_y, self.obstacles[:, :, 1]
            )

    def check_collisions(self, active_mask: torch.Tensor):
        """Check for collisions with obstacles"""

        # Car dimensions
        car_width, car_height = 50, 40

        # Ego car bounds: (batch_size, 4) [x_min, y_min, x_max, y_max]
        ego_bounds = torch.stack(
            [
                self.ego_positions[:, 0] - car_width / 2,
                self.ego_positions[:, 1] - car_height / 2,
                self.ego_positions[:, 0] + car_width / 2,
                self.ego_positions[:, 1] + car_height / 2,
            ],
            dim=1,
        )

        # Obstacle bounds: (batch_size, num_obstacles, 4)
        obs_bounds = torch.stack(
            [
                self.obstacles[:, :, 0] - self.obstacles[:, :, 2] / 2,
                self.obstacles[:, :, 1] - self.obstacles[:, :, 3] / 2,
                self.obstacles[:, :, 0] + self.obstacles[:, :, 2] / 2,
                self.obstacles[:, :, 1] + self.obstacles[:, :, 3] / 2,
            ],
            dim=2,
        )

        # Check rectangle overlap for each obstacle
        # ego_bounds: (batch, 4), obs_bounds: (batch, num_obs, 4)
        ego_expanded = ego_bounds.unsqueeze(1)  # (batch, 1, 4)

        # Check if rectangles overlap
        no_overlap = (
            (ego_expanded[:, :, 2] < obs_bounds[:, :, 0])
            | (ego_expanded[:, :, 0] > obs_bounds[:, :, 2])
            | (ego_expanded[:, :, 3] < obs_bounds[:, :, 1])
            | (ego_expanded[:, :, 1] > obs_bounds[:, :, 3])
        )

        collision = ~no_overlap  # (batch, num_obstacles)
        any_collision = collision.any(dim=1)  # (batch,)

        self.crashed = self.crashed | (any_collision & active_mask)

    def calculate_sensors(self) -> torch.Tensor:
        """Calculate sensor readings for all cars"""

        batch_size = self.batch_size
        num_sensors = len(self.sensor_angles)

        # Sensor readings: (batch_size, num_sensors)
        sensor_readings = torch.full(
            (batch_size, num_sensors), self.sensor_range, device=self.device
        )

        # For each sensor angle, cast a ray and find the closest intersection
        for i, angle in enumerate(self.sensor_angles):
            # Calculate ray direction
            ray_dx = torch.cos(angle)
            ray_dy = -torch.sin(angle)  # Negative because y increases downward

            # Ray end points (for debugging/visualization if needed)
            # ray_end_x = self.ego_positions[:, 0] + ray_dx * self.sensor_range
            # ray_end_y = self.ego_positions[:, 1] + ray_dy * self.sensor_range

            # Check intersection with walls
            wall_distances = torch.full(
                (batch_size,), self.sensor_range, device=self.device
            )

            # Top wall intersection
            if ray_dy < 0:  # Ray going upward
                t_top = (self.wall_top - self.ego_positions[:, 1]) / ray_dy
                valid_top = (t_top > 0) & (t_top <= self.sensor_range)
                wall_distances = torch.where(valid_top, t_top, wall_distances)

            # Bottom wall intersection
            if ray_dy > 0:  # Ray going downward
                t_bottom = (self.wall_bottom - self.ego_positions[:, 1]) / ray_dy
                valid_bottom = (t_bottom > 0) & (t_bottom <= self.sensor_range)
                wall_distances = torch.where(valid_bottom, t_bottom, wall_distances)

            # Check intersection with obstacles
            obstacle_distances = self.ray_rectangle_intersection(
                self.ego_positions[:, 0],
                self.ego_positions[:, 1],
                ray_dx,
                ray_dy,
                self.obstacles,
            )

            # Take minimum distance
            min_distance = torch.min(wall_distances, obstacle_distances)
            sensor_readings[:, i] = min_distance

        return sensor_readings

    def ray_rectangle_intersection(
        self,
        ray_x: torch.Tensor,
        ray_y: torch.Tensor,
        ray_dx: float,
        ray_dy: float,
        rectangles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate ray-rectangle intersection distances

        Args:
            ray_x, ray_y: (batch_size,) ray start positions
            ray_dx, ray_dy: ray direction components
            rectangles: (batch_size, num_rects, 4) [x, y, width, height]

        Returns:
            (batch_size,) minimum distance to any rectangle
        """

        batch_size, num_rects = rectangles.shape[:2]

        # Rectangle bounds
        rect_left = rectangles[:, :, 0] - rectangles[:, :, 2] / 2
        rect_right = rectangles[:, :, 0] + rectangles[:, :, 2] / 2
        rect_top = rectangles[:, :, 1] - rectangles[:, :, 3] / 2
        rect_bottom = rectangles[:, :, 1] + rectangles[:, :, 3] / 2

        # Expand ray positions for broadcasting
        ray_x_exp = ray_x.unsqueeze(1)  # (batch, 1)
        ray_y_exp = ray_y.unsqueeze(1)  # (batch, 1)

        # Calculate t values for intersection with each rectangle edge
        distances = torch.full(
            (batch_size, num_rects), self.sensor_range, device=self.device
        )

        # Avoid division by zero
        eps = 1e-8

        if abs(ray_dx) > eps:
            # Left edge intersection
            t_left = (rect_left - ray_x_exp) / ray_dx
            y_at_left = ray_y_exp + t_left * ray_dy
            valid_left = (
                (t_left > 0) & (y_at_left >= rect_top) & (y_at_left <= rect_bottom)
            )
            distances = torch.where(valid_left, torch.min(distances, t_left), distances)

            # Right edge intersection
            t_right = (rect_right - ray_x_exp) / ray_dx
            y_at_right = ray_y_exp + t_right * ray_dy
            valid_right = (
                (t_right > 0) & (y_at_right >= rect_top) & (y_at_right <= rect_bottom)
            )
            distances = torch.where(
                valid_right, torch.min(distances, t_right), distances
            )

        if abs(ray_dy) > eps:
            # Top edge intersection
            t_top = (rect_top - ray_y_exp) / ray_dy
            x_at_top = ray_x_exp + t_top * ray_dx
            valid_top = (t_top > 0) & (x_at_top >= rect_left) & (x_at_top <= rect_right)
            distances = torch.where(valid_top, torch.min(distances, t_top), distances)

            # Bottom edge intersection
            t_bottom = (rect_bottom - ray_y_exp) / ray_dy
            x_at_bottom = ray_x_exp + t_bottom * ray_dx
            valid_bottom = (
                (t_bottom > 0)
                & (x_at_bottom >= rect_left)
                & (x_at_bottom <= rect_right)
            )
            distances = torch.where(
                valid_bottom, torch.min(distances, t_bottom), distances
            )

        # Return minimum distance across all rectangles
        min_distances, _ = distances.min(dim=1)
        return min_distances

    def get_state_tensor(self, sensor_readings: torch.Tensor) -> torch.Tensor:
        """
        Convert game state to neural network input tensor

        Returns:
            (batch_size, state_dim) state tensor
        """

        # State components:
        # - Velocity (2D): normalized
        # - Sensor readings (16): normalized
        # - Current lane position: normalized

        # Normalize velocity
        vel_normalized = self.ego_velocities / 25.0  # Max velocity normalization

        # Normalize sensors
        sensors_normalized = sensor_readings / self.sensor_range

        # Normalize y position (lane information)
        y_normalized = self.ego_positions[:, 1] / self.screen_height

        # Concatenate all state components
        state = torch.cat(
            [
                vel_normalized,  # 2 features
                sensors_normalized,  # 16 features
                y_normalized.unsqueeze(1),  # 1 feature
            ],
            dim=1,
        )

        return state  # (batch_size, 19) features total

    def calculate_rewards(self, active_mask: torch.Tensor) -> torch.Tensor:
        """Calculate rewards for the current step"""

        rewards = torch.zeros(self.batch_size, device=self.device)

        # Distance reward (main objective)
        distance_reward = self.ego_velocities[:, 0] * 0.1  # Reward for forward movement

        # Survival bonus
        survival_bonus = torch.ones_like(rewards) * 1.0

        # Crash penalty
        crash_penalty = self.crashed.float() * -100.0

        # Wall proximity penalty (encourage staying in lanes)
        wall_penalty = torch.zeros_like(rewards)
        too_high = self.ego_positions[:, 1] < self.lane_height
        too_low = self.ego_positions[:, 1] > self.screen_height - self.lane_height
        wall_penalty[too_high | too_low] = -5.0

        # Combine rewards
        total_reward = (
            distance_reward + survival_bonus + crash_penalty + wall_penalty
        ) * active_mask

        return total_reward

    def get_action_space_size(self) -> int:
        """Return the size of the action space"""
        return 5  # [NOTHING, ACCELERATE, DECELERATE, STEER_LEFT, STEER_RIGHT]

    def get_state_space_size(self) -> int:
        """Return the size of the state space"""
        return 19  # 2 velocity + 16 sensors + 1 y_position

    def get_batch_stats(self) -> Dict:
        """Get statistics for the current batch"""
        return {
            "mean_distance": self.distances.mean().item(),
            "max_distance": self.distances.max().item(),
            "crash_rate": self.crashed.float().mean().item(),
            "mean_velocity": self.ego_velocities[:, 0].mean().item(),
            "active_envs": (~self.done).sum().item(),
        }
