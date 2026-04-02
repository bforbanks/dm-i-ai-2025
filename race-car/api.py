import time
import uvicorn
import datetime
from fastapi import Body, FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from dtos import RaceCarPredictRequestDto, RaceCarPredictResponseDto
from LaneShift.LaneShift import LaneShift

HOST = "0.0.0.0"
PORT = 9052

model = LaneShift()

app = FastAPI()
start_time = time.time()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})

@app.post('/predict', response_model=RaceCarPredictResponseDto)
def predict(request: RaceCarPredictRequestDto = Body(...)):
    action = model.return_action(request.model_dump())
    return RaceCarPredictResponseDto(actions=action)

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
