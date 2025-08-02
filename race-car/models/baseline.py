class BaselineModel:
    def return_action(self, state = None):
        actions = []
        if state is None:
            print("WTF IS WRONG???")
            return ["ACCELERATE"]
        
        
        if state['sensors']['front'] is not None:
            if state['sensors']['front'] < 600:
                action = 'DECELERATE'
            else:
                action = "NOTHING"
        elif state['sensors']['back'] is not None:
            if state['sensors']['back'] > 800:
                action = 'ACCELERATE'
            else:
                action = "NOTHING"
        else:
            action = 'NOTHING'

        actions.append(action)
        return actions