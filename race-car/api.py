import time
import uvicorn
import datetime
from fastapi import Body, FastAPI, Request
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
import random
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

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

# To see error messages cursor suggest that I add this. I don't know how it works but if it works it works
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print(f"\n❌ EXTERNAL REQUEST VALIDATION ERROR:")
    print(f"From IP: {request.client.host}")
    print(f"URL: {request.url}")
    print("Errors:",exc.errors())
    try:
        body = await request.body()
        print(f"Their request body: {body.decode('utf-8')}")
    except:
        print("Could not read their request body")
    print("=" * 50)
    
    return JSONResponse(status_code=422, content={"detail": exc.errors()})
# End of error handling

# The actual endpoint
@app.post('/predict', response_model=RaceCarPredictResponseDto)
async def predict(request: RaceCarPredictRequestDto = Body(...)):
    request_dict = request.model_dump() # Converts the request to a dictionary
    action = return_action(request_dict) # Returns a list of actions (our model is invoked here)
    return RaceCarPredictResponseDto( 
        actions=action
    ) # return in the correct format

# Random endpoint
@app.get('/api')
def hello():
    return {
        "service": "race-car-usecase",
        "uptime": '{}'.format(datetime.timedelta(seconds=time.time() - start_time))
    }

# health check
@app.get('/')
def index():
    return "Your endpoint is running!"

if __name__ == '__main__':
    
    uvicorn.run(
        'api:app',
        host=HOST,
        port=PORT
    )
