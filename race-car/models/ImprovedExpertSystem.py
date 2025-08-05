from models.BaseModel import BaseModel
from typing import List, Dict


class ImprovedExpertSystem(BaseModel):
    """
    Improved Expert System with simpler, more robust logic:
    - Clear priority-based decision making
    - Conservative but effective collision avoidance
    - Optimal velocity management without over-complication
    - Simple but effective lane change logic
    """

    def __init__(self):
        super().__init__()
        # Core parameters - tuned based on baseline analysis
        self.target_velocity = 14.0  # Conservative but fast
        self.max_velocity = 18.0
        self.min_velocity = 6.0

        # Distance thresholds (based on working baselines)
        self.danger_threshold = 300  # Emergency action needed
        self.caution_threshold = 500  # Start preparing
        self.safe_threshold = 800  # All clear

        # Lane change safety
        self.lane_safety_distance = 200  # Conservative safety margin
        self.lane_change_cooldown = 0

        # State tracking
        self.stuck_counter = 0  # Track if we're stuck
        self.last_distance = 0

    def return_action(self, state: dict) -> List[str]:
        """Main decision engine with clear priority hierarchy"""

        # Update tracking
        self._update_tracking(state)

        # Get sensor readings
        front = state["sensors"].get("front")
        back = state["sensors"].get("back")
        left_side = state["sensors"].get("left_side")
        right_side = state["sensors"].get("right_side")
        left_front = state["sensors"].get("left_front")
        right_front = state["sensors"].get("right_front")

        current_velocity = state["velocity"]["x"]

        # Debug info
        self._log_debug(state, front, current_velocity)

        # PRIORITY 1: Emergency collision avoidance
        if front is not None and front < self.danger_threshold:
            return self._emergency_avoidance(
                state, front, left_side, right_side, left_front, right_front
            )

        # PRIORITY 2: Cautious maneuvering
        if front is not None and front < self.caution_threshold:
            return self._cautious_maneuvering(
                state, front, current_velocity, left_side, right_side
            )

        # PRIORITY 3: Velocity optimization
        return self._optimize_velocity(state, current_velocity, front, back)

    def _update_tracking(self, state: dict):
        """Update internal state tracking"""
        current_distance = state["distance"]

        # Check if we're stuck (not making progress)
        if abs(current_distance - self.last_distance) < 1:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        self.last_distance = current_distance

        # Update lane change cooldown
        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1

    def _emergency_avoidance(
        self,
        state: dict,
        front: float,
        left_side: float,
        right_side: float,
        left_front: float,
        right_front: float,
    ) -> List[str]:
        """Handle emergency situations - immediate threat ahead"""

        # Check if lane change is possible and safe
        can_go_left = self._can_change_lane(
            "left", left_side, left_front, state["sensors"]
        )
        can_go_right = self._can_change_lane(
            "right", right_side, right_front, state["sensors"]
        )

        if can_go_left and (
            not can_go_right or left_side and right_side and left_side > right_side
        ):
            # Prefer left if both available and left has more space
            self.lane_change_cooldown = 10
            return ["STEER_LEFT"]
        elif can_go_right:
            self.lane_change_cooldown = 10
            return ["STEER_RIGHT"]
        else:
            # No safe lane change - hard brake
            return ["DECELERATE"]

    def _cautious_maneuvering(
        self,
        state: dict,
        front: float,
        velocity: float,
        left_side: float,
        right_side: float,
    ) -> List[str]:
        """Handle cautious situations - potential threat ahead"""

        # If we're going too fast for the situation, slow down
        if velocity > 12 and front < 400:
            return ["DECELERATE"]

        # Consider lane change if safe and beneficial
        if self.lane_change_cooldown == 0:
            can_go_left = self._can_change_lane(
                "left", left_side, None, state["sensors"]
            )
            can_go_right = self._can_change_lane(
                "right", right_side, None, state["sensors"]
            )

            if can_go_left and (
                not can_go_right
                or (left_side and right_side and left_side > right_side)
            ):
                self.lane_change_cooldown = 8
                return ["STEER_LEFT"]
            elif can_go_right:
                self.lane_change_cooldown = 8
                return ["STEER_RIGHT"]

        # Maintain current speed/slight deceleration
        if velocity > 10:
            return ["NOTHING"]
        else:
            return ["ACCELERATE"]

    def _optimize_velocity(
        self, state: dict, velocity: float, front: float, back: float
    ) -> List[str]:
        """Optimize velocity for maximum distance"""

        # If we're stuck, try to get unstuck
        if self.stuck_counter > 5:
            return ["ACCELERATE"]

        # Accelerate if safe and below target
        if velocity < self.target_velocity:
            # Only accelerate if path is clear
            if front is None or front > self.safe_threshold:
                return ["ACCELERATE"]
            else:
                return ["NOTHING"]

        # Maintain optimal speed
        if velocity <= self.max_velocity:
            return ["NOTHING"]
        else:
            # Gentle slowdown if too fast
            return ["DECELERATE"]

    def _can_change_lane(
        self, direction: str, side_sensor: float, front_sensor: float, all_sensors: dict
    ) -> bool:
        """Check if lane change is safe"""

        if self.lane_change_cooldown > 0:
            return False

        # Check side clearance
        if side_sensor is not None and side_sensor < self.lane_safety_distance:
            return False

        # Check diagonal sensors for more safety
        if direction == "left":
            left_back = all_sensors.get("left_back")
            if left_back is not None and left_back < self.lane_safety_distance:
                return False
        else:  # right
            right_back = all_sensors.get("right_back")
            if right_back is not None and right_back < self.lane_safety_distance:
                return False

        # Check front diagonal if available
        if front_sensor is not None and front_sensor < self.lane_safety_distance:
            return False

        return True

    def _log_debug(self, state: dict, front: float, velocity: float):
        """Log debug information (can be enabled/disabled)"""
        # Uncomment for debugging
        # print(f"Tick {state['elapsed_ticks']}: V={velocity:.1f}, Front={front}, Dist={state['distance']:.1f}")
        pass


# Alternative even simpler system based on proven baselines
class SimpleExpertSystem(BaseModel):
    """Ultra-simple system based on proven baseline logic"""

    def return_action(self, state: dict) -> List[str]:
        front = state["sensors"].get("front")
        back = state["sensors"].get("back")
        velocity = state["velocity"]["x"]

        # Based on BaselineL but with improvements
        if front is not None:
            if front < 400:  # Danger zone
                return ["DECELERATE"]
            elif front > 600 and velocity < 16:  # Safe to accelerate
                return ["ACCELERATE"]
            else:
                return ["NOTHING"]

        # Use back sensor like BaselineV
        if back is not None:
            if back < 300:  # Something approaching fast
                return ["ACCELERATE"]
            elif back > 700:
                return ["NOTHING"]

        # Default: maintain or gain speed
        if velocity < 14:
            return ["ACCELERATE"]
        else:
            return ["NOTHING"]
