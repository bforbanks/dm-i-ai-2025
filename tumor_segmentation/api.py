from pathlib import Path
import sys
import os
import cv2
import uvicorn
import time
import datetime
import numpy as np
import cv2
import requests
import json
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

# Validation API configuration
VALIDATION_API_URL = os.getenv("VALIDATION_API_URL", "https://cases.dmiai.dk/api/v1/usecases/tumor-segmentation/validate/queue")
VALIDATION_TOKEN = os.getenv("VALIDATION_TOKEN", "18238e2f472643739573ad26f3680c51")
PREDICT_URL = os.getenv("PREDICT_URL", "https://5ff4ebe3f622.ngrok-free.app/predict")

# Padding configuration - set this to match your training configuration
USE_PADDING = os.getenv("USE_PADDING", "false").lower() == "true"
TARGET_WIDTH = int(os.getenv("TARGET_WIDTH", "400"))
TARGET_HEIGHT = int(os.getenv("TARGET_HEIGHT", "992"))

print(f"API Configuration:")
print(f"  USE_PADDING: {USE_PADDING}")
print(f"  TARGET_WIDTH: {TARGET_WIDTH}")
print(f"  TARGET_HEIGHT: {TARGET_HEIGHT}")
print(f"  VALIDATION_API_URL: {VALIDATION_API_URL}")
print(f"  PREDICT_URL: {PREDICT_URL}")

# Load the PyTorch Lightning model from checkpoint (class method)
checkpoint_path = os.getenv(
    "CHECKPOINT_PATH",
)
if not checkpoint_path:
    raise ValueError("CHECKPOINT_PATH environment variable is not set")

# model = SimpleUNet.load_from_checkpoint(
# model = SimpleUNet.load_from_checkpoint(
#     checkpoint_path, map_location="cpu"
# )  # Force CPU loading
model = SimpleUNet()
model.eval()  # Set to evaluation mode
model.freeze()  # Freeze the model for inference

app = FastAPI()
start_time = time.time()



def queue_validation_attempt(predict_url: str | None = None) -> dict:
    """
    Queue a validation attempt against the online validation dataset.
    
    Args:
        predict_url: URL for the prediction endpoint (defaults to PREDICT_URL env var)
    
    Returns:
        dict: Response from the validation API containing queue information
    """
    if predict_url is None:
        predict_url = PREDICT_URL
    
    headers = {
        "x-token": VALIDATION_TOKEN,
        "Content-Type": "application/json"
    }
    
    data = {
        "url": predict_url
    }
    
    try:
        response = requests.post(
            VALIDATION_API_URL,
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error queueing validation attempt: {e}")
        return {"error": str(e)}


@app.post("/predict", response_model=TumorPredictResponseDto)
def predict_endpoint(request: TumorPredictRequestDto):
    img: np.ndarray = decode_request(request)

    # predict will be called mutiple times. Save the images in a folder
    # os.makedirs("images", exist_ok=True)
    # img_path = os.path.join("images", f"{time.time()}.png")
    # cv2.imwrite(img_path, img)

    predicted_segmentation = model.predict(img)
    
    # Determine target size based on padding configuration
    if USE_PADDING:
        target_size = (TARGET_WIDTH, TARGET_HEIGHT)
    else:
        target_size = (256, 256)
    
    # Use the model's predict method with the correct target size
    # The model's predict method will handle the preprocessing internally
    predicted_segmentation = model.predict(img, target_size=target_size)

    validate_segmentation(img, predicted_segmentation)

    encoded_segmentation = encode_request(predicted_segmentation)

    response = TumorPredictResponseDto(img=encoded_segmentation)
    return response


@app.post("/queue-validation")
def queue_validation_endpoint(predict_url: str | None = None):
    """
    Endpoint to queue a validation attempt against the online validation dataset.
    """
    result = queue_validation_attempt(predict_url)
    return result


@app.get("/validation-status/{attempt_uuid}")
def check_validation_status_endpoint(attempt_uuid: str):
    """
    Endpoint to check the status of a validation attempt and get the score if completed.
    """
    result = check_validation_status(attempt_uuid)
    return result


def check_validation_status(attempt_uuid: str) -> dict:
    """
    Check the status of a validation attempt and get the score if completed.
    
    Args:
        attempt_uuid: UUID of the validation attempt to check
    
    Returns:
        dict: Status information including score if completed
    """
    headers = {
        "x-token": VALIDATION_TOKEN
    }
    
    status_url = f"{VALIDATION_API_URL}/{attempt_uuid}"
    
    try:
        response = requests.get(
            status_url,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        # Add status logging for debugging
        print(f"Validation status: {result.get('status')}")
        if result.get('status') == 'done' and 'attempt' in result:
            score = result['attempt'].get('score')
            if score is not None:
                print(f"Validation score: {score:.6f}")
        
        return result
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


@app.get("/api")
def hello():
    return {
        "service": "tumor-segmentation-api",
        "uptime": "{}".format(datetime.timedelta(seconds=time.time() - start_time)),
        "padding_enabled": USE_PADDING,
        "target_width": TARGET_WIDTH,
        "target_height": TARGET_HEIGHT,
        "validation_api_url": VALIDATION_API_URL,
        "predict_url": PREDICT_URL,
    }


@app.get("/")
def index():
    return "Your tumor segmentation endpoint is running!"


if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT)
