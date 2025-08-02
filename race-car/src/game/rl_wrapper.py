from src.game.core import initialize_game_state, update_game, state_to_state_dict, STATE, GameState
class RaceCarEnv:
    def __init__(self, api_url, seed_value=None, sensor_removal=0):
        self.api_url = api_url
        self.seed_value = seed_value
        self.sensor_removal = sensor_removal
        self.state = None
        self.done = False
        initialize_game_state(api_url=self.api_url, seed_value=self.seed_value, sensor_removal=self.sensor_removal)
    def reset(self):
        initialize_game_state(api_url=self.api_url, seed_value=self.seed_value, sensor_removal=self.sensor_removal)
        self.state = state_to_state_dict(STATE)
        self.done = False
        return self.state
    
    def step(self, action):
        update_game(action)
        self.state = state_to_state_dict(STATE)
        # Define your own reward logic here
        reward = self._get_reward()

        # Define your own termination condition
        self.done = STATE.crashed or STATE.ticks > 3600 or STATE.elapsed_game_time > 60000  # for example
        return self.state, reward, self.done
    
    def _get_reward(self):
        if STATE.crashed:
            return -100
        return STATE.ego.velocity.x / 10