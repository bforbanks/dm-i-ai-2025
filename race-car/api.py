import time
import uvicorn
import datetime
from fastapi import Body, FastAPI, Request
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
import random

# from test_endpoint import return_action
def return_action(state):
    # Returns a list of actions
    actions = []
    action_choices = ['ACCELERATE', 'DECELERATE', 'STEER_LEFT', 'STEER_RIGHT', 'NOTHING']
    for _ in range(10):
        actions.append(random.choice(action_choices))
    return actions

HOST = "0.0.0.0"
PORT = 9052


app = FastAPI()
start_time = time.time()

@app.post('/predict', response_model=RaceCarPredictResponseDto)
async def predict(request: RaceCarPredictRequestDto = Body(...)):
    action = return_action(request.model_dump())
    return RaceCarPredictResponseDto(
        actions=action
    )

@app.get('/api')
def hello():
    return {
        "service": "race-car-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }


@app.get('/')
def index():
    return "Your endpoint is running!"

if __name__ == '__main__':

    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )
