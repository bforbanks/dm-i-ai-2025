from models.BaseModel import BaseModel
from typing import List


class TestModel(BaseModel):
    """
    Ultra-simple test model based exactly on BaselineL logic
    to verify the testing system works
    """

    def return_action(self, state: dict) -> List[str]:
        # Exactly the same logic as BaselineL
        front = state["sensors"].get("front")
        back = state["sensors"].get("back")

        if front:
            if front < 500:
                return ["DECELERATE"]
            elif front > 500 and state["velocity"]["x"] < 15:
                return ["ACCELERATE"]
            else:
                return ["NOTHING"]

        if back:
            if back < 500:
                return ["ACCELERATE"]
            elif back > 500:
                return ["DECELERATE"]
            else:
                return ["NOTHING"]
        else:
            return ["ACCELERATE"]
