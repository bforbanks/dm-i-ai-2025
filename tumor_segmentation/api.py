from pathlib import Path
import sys
import os
import uvicorn
import time
import datetime
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI

# Load environment variables from .env file
load_dotenv()

# Add project root to path FIRST (parent of tumor_segmentation directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dtos import TumorPredictRequestDto, TumorPredictResponseDto
from utils import validate_segmentation, encode_request, decode_request
from models import NNUNetStyle
from models import SimpleUNet

# Load configuration from environment variables
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9052"))

# Load the PyTorch Lightning model from checkpoint (class method)
checkpoint_path = os.getenv(
    "CHECKPOINT_PATH",
)
if not checkpoint_path:
    raise ValueError("CHECKPOINT_PATH environment variable is not set")
    
# model = SimpleUNet.load_from_checkpoint(
model = NNUNetStyle.load_from_checkpoint(
    checkpoint_path, map_location="cpu"
)  # Force CPU loading
model.eval()  # Set to evaluation mode
model.freeze()  # Freeze the model for inference

app = FastAPI()
start_time = time.time()

@app.post("/predict", response_model=TumorPredictResponseDto)
def predict_endpoint(request: TumorPredictRequestDto):
    img: np.ndarray = decode_request(request)

    predicted_segmentation = model.predict(img)

    validate_segmentation(img, predicted_segmentation)

    encoded_segmentation = encode_request(predicted_segmentation)

    response = TumorPredictResponseDto(img=encoded_segmentation)
    return response


@app.get("/api")
def hello():
    return {
        "service": "race-car-usecase",
        "uptime": "{}".format(datetime.timedelta(seconds=time.time() - start_time)),
    }


@app.get("/")
def index():
    return "Your endpoint is running!"


if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT)
