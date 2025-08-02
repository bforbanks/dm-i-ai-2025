class BaseModel:
    def __init__(self):
        pass

    def return_action(self, state: dict) -> list[str]:
        """Takes in game state in dictionary form and returns a list of actions
        which will be appended to the action list and executed in order"""
        raise NotImplementedError("Subclasses must implement this method")


