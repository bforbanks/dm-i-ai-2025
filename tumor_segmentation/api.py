from pathlib import Path
import sys

# Add project root to path FIRST (parent of tumor_segmentation directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import uvicorn
import time
import datetime
import numpy as np
from fastapi import FastAPI
from dtos import TumorPredictRequestDto, TumorPredictResponseDto
from utils import validate_segmentation, encode_request, decode_request
from models import SimpleUNet

HOST = "0.0.0.0"
PORT = 9052

# Load the PyTorch Lightning model from checkpoint (class method)
checkpoint_path = "C:/Users/Benja/dev/dm-i-ai-2025/tumor_segmentation/checkpoints/simple-unet-epoch=39-val_dice=0.4218.ckpt"
model = SimpleUNet.load_from_checkpoint(
    checkpoint_path, map_location="cpu"
)  # Force CPU loading
model.eval()  # Set to evaluation mode
model.freeze()  # Freeze the model for inference

app = FastAPI()
start_time = time.time()


@app.post("/predict", response_model=TumorPredictResponseDto)
def predict_endpoint(request: TumorPredictRequestDto):
    # Decode request str to numpy array
    img: np.ndarray = decode_request(request)

    print(f"img shape: {img.shape}")

    # Obtain segmentation prediction
    # predicted_segmentation = predict(img)
    predicted_segmentation = model.predict(img)

    print(f"predicted_segmentation shape: {predicted_segmentation.shape}")

    # convert grayscale to rgb
    predicted_segmentation = cv2.cvtColor(predicted_segmentation, cv2.COLOR_GRAY2RGB)

    # make image# invert colors
    predicted_segmentation = cv2.bitwise_not(predicted_segmentation)

    # display image
    # cv2.imshow("predicted_segmentation", predicted_segmentation)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Also convert input image to RGB for validation
    # Input is now always (H, W, 1) from decode_request
    if predicted_segmentation.shape[2] == 1:  # Grayscale image with 1 channel
        img_for_validation = cv2.cvtColor(
            predicted_segmentation.squeeze(2), cv2.COLOR_GRAY2RGB
        )
    else:  # Already RGB with 3 channels
        img_for_validation = predicted_segmentation

    print(f"img_for_validation shape: {img_for_validation.shape}")

    # convert to np array
    predicted_segmentation = np.array(predicted_segmentation)

    # display image
    # cv2.imshow("img_for_validation", img_for_validation)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Validate segmentation format
    validate_segmentation(img_for_validation, predicted_segmentation)

    # Encode the segmentation array to a str
    encoded_segmentation = encode_request(predicted_segmentation)

    # Return the encoded segmentation to the validation/evalution service
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
