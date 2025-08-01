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
from utils import validate_segmentation, encode_request, decode_request, plot_prediction
from models import SimpleUNet
from example import predict

HOST = "0.0.0.0"
PORT = 9052

# Load the PyTorch Lightning model from checkpoint (class method)
checkpoint_path = "C:/Users/Benja/dev/dm-i-ai-2025/tumor_segmentation/checkpoints/simple-unet-epoch=20-val_dice=0.3963.ckpt"
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

    # print(f"Input img shape: {img.shape}")

    # print(f"Input img max: {img.max()}")
    # print(f"Input img min: {img.min()}")

    # Get prediction from model (all preprocessing/postprocessing happens in model)
    predicted_segmentation = model.predict(img)

    # example_prediction = predict(img)

    # print(f"Predicted segmentation shape: {predicted_segmentation.shape}")

    # print(f"Predicted segmentation max: {predicted_segmentation.max()}")
    # print(f"Predicted segmentation min: {predicted_segmentation.min()}")
    # print(f"Predicted segmentation mean: {predicted_segmentation.mean()}")
    # print(f"Predicted segmentation std: {predicted_segmentation.std()}")

    # Plot using utils function (same as in plot_baseline_predictions.py)
    # try:
    #     # Create dummy mask (all zeros) since we don't have ground truth in inference
    #     dummy_mask = np.zeros_like(predicted_segmentation)

    #     print("Displaying prediction using utils plot_prediction...")
    #     plot_prediction(img, dummy_mask, predicted_segmentation)

    # except Exception as e:
    #     print(f"Error with utils plotting: {e}")
    #     # Fallback to simple cv2 display
    #     cv2.imshow("Predicted segmentation", predicted_segmentation)
    #     cv2.imshow("Example prediction", example_prediction)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # Validate segmentation format
    validate_segmentation(img, predicted_segmentation)

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
