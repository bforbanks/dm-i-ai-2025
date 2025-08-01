class BaselineModel:
    def return_action(self, state = None):
        if state is None:
            print("WTF IS WRONG???")
            return "ACCELERATE"
        actions = []
        
        if state['sensors']['front'] is not None:
            if state['sensors']['front'] < 600:
                action = 'DECELERATE'
            elif state['sensors']['back'] > 800:
                action = 'ACCELERATE'
            else:
                action = 'NOTHING'
        else:
            action = 'NOTHING'    
        actions.append(action)
        return action