import math
from models.BaseModel import BaseModel
from typing import List, Dict, Tuple, Optional


class OptimalExpertSystem(BaseModel):
    """
    Advanced Expert System implementing optimal racing strategy:
    - Aggressive velocity management (target 15-18 velocity.x)
    - Predictive collision detection with multi-step lookahead
    - Advanced sensor fusion with threat assessment
    - Batch action optimization for network efficiency
    - Dynamic lane change logic with risk assessment
    """

    def __init__(self):
        super().__init__()
        # Strategy parameters
        self.target_velocity = 16.0  # Optimal velocity for distance maximization
        self.max_safe_velocity = 20.0
        self.min_velocity = 8.0

        # Threat detection thresholds
        self.emergency_threshold = 200  # Immediate danger
        self.warning_threshold = 400  # Prepare for action
        self.comfort_threshold = 700  # Safe zone

        # Lane change parameters
        self.lane_change_safety_margin = 80  # Minimum clearance needed
        self.lane_change_cooldown = 0

        # Batch action parameters
        self.action_buffer = []
        self.max_batch_size = 6

        # State tracking for prediction
        self.previous_state = None
        self.velocity_history = []

    def return_action(self, state: dict) -> List[str]:
        """Main decision engine - returns optimized batch of actions"""

        # Update internal tracking
        self._update_tracking(state)

        # If we have buffered actions, use them first
        if self.action_buffer:
            return [self.action_buffer.pop(0)]

        # Generate new batch of actions
        self.action_buffer = self._generate_action_batch(state)

        # Return first action from the batch
        return [self.action_buffer.pop(0)] if self.action_buffer else ["NOTHING"]

    def _update_tracking(self, state: dict):
        """Update internal state tracking for prediction"""
        current_velocity = state["velocity"]["x"]
        self.velocity_history.append(current_velocity)

        # Keep only recent history (last 10 ticks)
        if len(self.velocity_history) > 10:
            self.velocity_history = self.velocity_history[-10:]

        # Update lane change cooldown
        if self.lane_change_cooldown > 0:
            self.lane_change_cooldown -= 1

        self.previous_state = state

    def _generate_action_batch(self, state: dict) -> List[str]:
        """Generate optimized batch of actions using predictive strategy"""

        # Analyze current situation
        threat_assessment = self._assess_threats(state)
        velocity_status = self._analyze_velocity(state)
        lane_options = self._evaluate_lane_options(state)

        # Decide primary strategy
        if threat_assessment["emergency"]:
            return self._emergency_response(state, threat_assessment, lane_options)
        elif threat_assessment["warning"]:
            return self._warning_response(state, threat_assessment, lane_options)
        else:
            return self._normal_driving(state, velocity_status, lane_options)

    def _assess_threats(self, state: dict) -> Dict:
        """Advanced threat assessment using sensor fusion"""
        sensors = state["sensors"]

        # Critical sensors for forward threat detection
        front_sensors = {
            "front": sensors.get("front"),
            "front_left_front": sensors.get("front_left_front"),
            "front_right_front": sensors.get("front_right_front"),
            "left_front": sensors.get("left_front"),
            "right_front": sensors.get("right_front"),
        }

        # Filter out None readings and get minimum distance
        valid_front = [d for d in front_sensors.values() if d is not None]
        min_front_distance = min(valid_front) if valid_front else float("inf")

        # Side sensors for lane change assessment
        left_clear = self._check_lane_clearance(sensors, "left")
        right_clear = self._check_lane_clearance(sensors, "right")

        # Predictive collision time
        current_velocity = state["velocity"]["x"]
        collision_time = (
            min_front_distance / max(current_velocity, 1)
            if min_front_distance != float("inf")
            else float("inf")
        )

        return {
            "emergency": min_front_distance < self.emergency_threshold,
            "warning": min_front_distance < self.warning_threshold,
            "min_distance": min_front_distance,
            "collision_time": collision_time,
            "left_clear": left_clear,
            "right_clear": right_clear,
            "front_sensors": front_sensors,
        }

    def _check_lane_clearance(self, sensors: dict, side: str) -> bool:
        """Check if a lane change to the specified side is safe"""
        if side == "left":
            relevant_sensors = [
                "left_side",
                "left_front",
                "left_back",
                "left_side_front",
                "left_side_back",
            ]
        else:  # right
            relevant_sensors = [
                "right_side",
                "right_front",
                "right_back",
                "right_side_front",
                "right_side_back",
            ]

        for sensor_name in relevant_sensors:
            distance = sensors.get(sensor_name)
            if distance is not None and distance < self.lane_change_safety_margin:
                return False
        return True

    def _analyze_velocity(self, state: dict) -> Dict:
        """Analyze current velocity status and optimization needs"""
        current_vel = state["velocity"]["x"]

        return {
            "current": current_vel,
            "too_slow": current_vel < self.target_velocity - 2,
            "optimal": self.target_velocity - 2
            <= current_vel
            <= self.target_velocity + 2,
            "too_fast": current_vel > self.max_safe_velocity,
            "can_accelerate": current_vel < self.max_safe_velocity,
        }

    def _evaluate_lane_options(self, state: dict) -> Dict:
        """Evaluate available lane change options"""
        threat_assessment = self._assess_threats(state)

        return {
            "left_available": threat_assessment["left_clear"]
            and self.lane_change_cooldown == 0,
            "right_available": threat_assessment["right_clear"]
            and self.lane_change_cooldown == 0,
            "current_safe": threat_assessment["min_distance"] > self.comfort_threshold,
        }

    def _emergency_response(self, state: dict, threats: Dict, lanes: Dict) -> List[str]:
        """Handle emergency situations - immediate threat detected"""
        actions = []

        # Priority 1: Avoid collision through lane change if possible
        if lanes["left_available"]:
            actions.extend(["STEER_LEFT", "STEER_LEFT", "NOTHING"])
            self.lane_change_cooldown = 8
        elif lanes["right_available"]:
            actions.extend(["STEER_RIGHT", "STEER_RIGHT", "NOTHING"])
            self.lane_change_cooldown = 8
        else:
            # Priority 2: Hard braking if no lane change possible
            actions.extend(["DECELERATE", "DECELERATE", "DECELERATE"])

        # Add stabilization actions
        actions.extend(["NOTHING", "NOTHING"])

        return actions[: self.max_batch_size]

    def _warning_response(self, state: dict, threats: Dict, lanes: Dict) -> List[str]:
        """Handle warning situations - prepare for potential threat"""
        actions = []

        # Predictive lane change if safe and beneficial
        if threats["collision_time"] < 15:  # Will collide in ~15 ticks
            if lanes["left_available"]:
                actions.extend(["STEER_LEFT", "NOTHING", "NOTHING"])
                self.lane_change_cooldown = 6
            elif lanes["right_available"]:
                actions.extend(["STEER_RIGHT", "NOTHING", "NOTHING"])
                self.lane_change_cooldown = 6
            else:
                # Controlled deceleration
                actions.extend(["DECELERATE", "NOTHING"])
        else:
            # Maintain speed but prepare
            actions.extend(["NOTHING", "NOTHING"])

        return actions[: self.max_batch_size]

    def _normal_driving(self, state: dict, velocity: Dict, lanes: Dict) -> List[str]:
        """Normal driving - optimize for distance"""
        actions = []

        # Aggressive velocity optimization
        if velocity["too_slow"]:
            # Accelerate to optimal speed
            accel_needed = min(
                4, int((self.target_velocity - velocity["current"]) / 0.1)
            )
            actions.extend(["ACCELERATE"] * accel_needed)
        elif velocity["too_fast"]:
            # Gentle deceleration
            actions.extend(["DECELERATE", "NOTHING"])
        else:
            # Maintain optimal speed with occasional acceleration
            if velocity["can_accelerate"] and len(self.velocity_history) > 5:
                avg_velocity = sum(self.velocity_history[-5:]) / 5
                if avg_velocity < self.target_velocity:
                    actions.extend(["ACCELERATE", "NOTHING"])
                else:
                    actions.extend(["NOTHING"] * 3)
            else:
                actions.extend(["NOTHING"] * 3)

        # Fill batch to optimal size for network efficiency
        while len(actions) < self.max_batch_size:
            actions.append("NOTHING")

        return actions[: self.max_batch_size]

    def _predict_future_state(self, state: dict, steps: int = 5) -> Dict:
        """Predict future state for advanced planning"""
        # Simple prediction based on current velocity and trends
        current_vel = state["velocity"]["x"]

        # Predict velocity change based on recent history
        if len(self.velocity_history) >= 3:
            velocity_trend = (self.velocity_history[-1] - self.velocity_history[-3]) / 2
        else:
            velocity_trend = 0

        predicted_velocity = current_vel + (velocity_trend * steps)
        predicted_distance = state["distance"] + (current_vel * steps)

        return {
            "predicted_velocity": predicted_velocity,
            "predicted_distance": predicted_distance,
            "velocity_trend": velocity_trend,
        }
