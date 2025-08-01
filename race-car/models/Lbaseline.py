class Lbaseline:
    def __init__(self):
        pass

    def return_action(self, state):
        # Returns a list of actions
        front = state['sensors']['front']
        back = state['sensors']['back']
        if front is not None:
            if front < 500:
                return "DECELERATE"
            elif front > 500 and state["velocity"]["x"] < 15:
                return "ACCELERATE"
            else:
                return "NOTHING"
        if back is not None:
            if back < 500:
                return "ACCELERATE"
            elif back > 500:
                return "DECELERATE"
            else:
                return "NOTHING"
        else:
            return "ACCELERATE"
