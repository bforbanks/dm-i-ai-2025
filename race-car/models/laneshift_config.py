from dataclasses import dataclass

@dataclass
class LaneShiftConfig:
    # General dimensions
    HEIGHT: int = 1200
    MARGINS: int = 40
    LANE_COUNT: int = 5  # Total lanes

    # Car parameters
    CAR_WIDTH: int = 360
    CAR_HEIGHT: int = 179

    # Movement / action parameters
    LANE_SHIFT_DISTANCE: float = 224.0  # Pixels per one-lane shift
    FRONT_BUFFER: float = 250.0         # Pixels considered as front safe buffer
    SIDE_BUFFER: float = 250.0          # Pixels considered as side safe buffer

    # Collision / safety parameters
    COLLISION_TICKS_THRESHOLD: int = 45      # Min ticks until collision considered safe
    COLLISION_TIME_THRESHOLD: int = 50       # Alternative threshold used in logic
    ACTION_QUEUE_MARGIN_FRONT: int = 20      # Additional queued actions margin when checking front collision
    ACTION_QUEUE_MARGIN_SIDE: int = 200      # Additional queued actions margin when checking side collision

    # Miscellaneous thresholds
    EARLY_ACCELERATE_TICKS: int = 300
    START_ACCELERATION_TICKS: int = 0  # Must be < EARLY_ACCELERATE_TICKS
    FULL_STOP_ACCELERATION_TICKS: int = 301  # Must be > EARLY_ACCELERATE_TICKS
    MAX_EVALUATED_TICKS: int = 3600


def get_default_config() -> LaneShiftConfig:
    """Return a default configuration instance."""
    return LaneShiftConfig()
