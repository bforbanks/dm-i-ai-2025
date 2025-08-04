from pathlib import Path
import sys
import os
import uvicorn
import time
import datetime
import numpy as np
import cv2
import requests
import json
import tempfile
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI
from typing import Union, Tuple, Optional

# Load environment variables from .env file
load_dotenv()

# Add project root to path FIRST (parent of tumor_segmentation directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dtos import TumorPredictRequestDto, TumorPredictResponseDto
from utils import validate_segmentation, encode_request, decode_request

# nnUNet v2 imports (required)
try:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
    from batchgenerators.utilities.file_and_folder_operations import join
    import torch
except ImportError as e:
    raise ImportError(f"nnUNet v2 is required for this API. Please install nnunetv2: {e}")

# Load configuration from environment variables
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9052"))  # Different port to avoid conflicts

# Performance optimization settings
SKIP_VALIDATION = os.getenv("SKIP_VALIDATION", "false").lower() == "true"  # Skip validation for speed

# Validation API configuration
VALIDATION_API_URL = os.getenv("VALIDATION_API_URL", "https://cases.dmiai.dk/api/v1/usecases/tumor-segmentation/validate/queue")
VALIDATION_TOKEN = os.getenv("VALIDATION_TOKEN", "18238e2f472643739573ad26f3680c51")
PREDICT_URL = os.getenv("PREDICT_URL", "https://5ff4ebe3f622.ngrok-free.app/predict")

# nnUNet v2 model configuration
MODEL_FOLDER = os.getenv(
    "MODEL_FOLDER",
    "tumor_segmentation/data_nnUNet/results/Dataset001_TumorSegmentation/nnUNetTrainer__nnUNetResEncUNetMPlans__2d_resenc_optimized"
)

# Ensure MODEL_FOLDER points to the base directory, not a fold directory
if MODEL_FOLDER.endswith('/fold_0') or MODEL_FOLDER.endswith('\\fold_0'):
    MODEL_FOLDER = os.path.dirname(MODEL_FOLDER)
    print(f"⚠️  MODEL_FOLDER was pointing to fold_0 directory, corrected to: {MODEL_FOLDER}")

CONFIGURATION_NAME = os.getenv("CONFIGURATION_NAME", "2d_resenc_optimized")
USE_FOLDS = os.getenv("USE_FOLDS", "0")  # Use fold 0 by default
CHECKPOINT_NAME = os.getenv("CHECKPOINT_NAME", "checkpoint_best.pth")

# nnUNet v2 prediction settings
TILE_STEP_SIZE = float(os.getenv("TILE_STEP_SIZE", "0.5"))  # 50% overlap for tiling
USE_MIRRORING = os.getenv("USE_MIRRORING", "true").lower() == "true"
USE_GAUSSIAN = os.getenv("USE_GAUSSIAN", "true").lower() == "true"
PERFORM_EVERYTHING_ON_DEVICE = os.getenv("PERFORM_EVERYTHING_ON_DEVICE", "true").lower() == "true"

print(f"nnUNet v2 .pth API Configuration:")
print(f"  MODEL_FOLDER: {MODEL_FOLDER}")
print(f"  CONFIGURATION_NAME: {CONFIGURATION_NAME}")
print(f"  USE_FOLDS: {USE_FOLDS}")
print(f"  CHECKPOINT_NAME: {CHECKPOINT_NAME}")
print(f"  TILE_STEP_SIZE: {TILE_STEP_SIZE}")
print(f"  USE_MIRRORING: {USE_MIRRORING}")
print(f"  USE_GAUSSIAN: {USE_GAUSSIAN}")
print(f"  PERFORM_EVERYTHING_ON_DEVICE: {PERFORM_EVERYTHING_ON_DEVICE}")
print(f"  VALIDATION_API_URL: {VALIDATION_API_URL}")
print(f"  PREDICT_URL: {PREDICT_URL}")
print(f"  SKIP_VALIDATION: {SKIP_VALIDATION}")

# Initialize the nnUNet v2 predictor in memory
def initialize_predictor():
    """Initialize the nnUNet v2 predictor and keep it in memory"""
    try:
        print(f"🔄 Loading nnUNet v2 predictor into memory...")
        predictor_start_time = time.time()
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {device}")
        
        # Instantiate the nnUNetPredictor with optimized settings
        predictor = nnUNetPredictor(
            tile_step_size=TILE_STEP_SIZE,
            use_gaussian=USE_GAUSSIAN,
            use_mirroring=USE_MIRRORING,
            perform_everything_on_device=PERFORM_EVERYTHING_ON_DEVICE,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )
        
        # Initialize from trained model folder
        predictor.initialize_from_trained_model_folder(
            MODEL_FOLDER,
            use_folds=(int(USE_FOLDS),),  # Convert to tuple
            checkpoint_name=CHECKPOINT_NAME,
        )
        
        predictor_load_time = time.time() - predictor_start_time
        print(f"✅ nnUNet v2 predictor loaded in {predictor_load_time:.2f} seconds")
        print(f"   Model folder: {MODEL_FOLDER}")
        print(f"   Configuration: {CONFIGURATION_NAME}")
        print(f"   Fold: {USE_FOLDS}")
        print(f"   Checkpoint: {CHECKPOINT_NAME}")
        
        return predictor, device
        
    except Exception as e:
        print(f"❌ Failed to initialize nnUNet v2 predictor: {e}")
        raise RuntimeError(f"Predictor initialization failed: {e}")

# Initialize predictor at startup
try:
    predictor, device = initialize_predictor()
    predictor_loaded = True
except Exception as e:
    print(f"❌ Predictor initialization failed: {e}")
    predictor_loaded = False
    predictor = None
    device = None

app = FastAPI()
start_time = time.time()

# Global prediction counter and timing
prediction_counter = 0
last_prediction_time = 0.0


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


def preprocess_image_for_nnunetv2(img: np.ndarray) -> np.ndarray:
    """
    Preprocess image for nnUNet v2 prediction to match training data format.
    
    CRITICAL: nnUNet v2 expects input images to be in exactly the same format as training images.
    The training data uses NaturalImage2DIO with PNG format, so we must ensure our images match.
    
    Args:
        img: Input image as numpy array (H, W, C) or (H, W)
        
    Returns:
        Preprocessed image ready for nnUNet v2 (matches training data format)
    """
    # Ensure image is grayscale (single channel) - same as training data
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            # Convert RGB to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.shape[2] == 1:
            img = img[:, :, 0]
    
    # Ensure image is in uint8 format with 0-255 range
    # This matches the training data format exactly
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # Ensure the image is in the expected range (0-255)
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # IMPORTANT: The image should now be in the exact same format as the training images
    # nnUNet v2 will apply the same preprocessing pipeline (Z-score normalization, resampling, etc.)
    # that was used during training, but only if the input format matches!
    
    return img


def save_image_for_nnunetv2(img: np.ndarray, output_path: str) -> str:
    """
    Save image in the format expected by nnUNet v2.
    
    Args:
        img: Image to save
        output_path: Path to save the image
        
    Returns:
        Path to saved image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save image
    cv2.imwrite(output_path, img)
    return output_path


@app.post("/predict", response_model=TumorPredictResponseDto)
def predict_endpoint(request: TumorPredictRequestDto):
    """Predict endpoint using nnUNet v2 predictor with .pth model"""
    global prediction_counter, last_prediction_time
    
    # Start timing
    prediction_start_time = time.time()
    
    if not predictor_loaded or predictor is None:
        raise RuntimeError("nnUNet v2 predictor is not loaded. Please check model initialization.")
    
    # Decode the request
    decode_start = time.time()
    img: np.ndarray = decode_request(request)
    decode_time = time.time() - decode_start
    print(f"   ⏱️  Decode request: {decode_time:.3f}s")
    
    # Preprocess image for nnUNet v2
    preprocess_start = time.time()
    img = preprocess_image_for_nnunetv2(img)
    preprocess_time = time.time() - preprocess_start
    print(f"   ⏱️  Preprocess image: {preprocess_time:.3f}s")
    
    # Create temporary directories
    temp_start = time.time()
    temp_dir = tempfile.mkdtemp(prefix="nnunetv2_pth_")
    input_image_path = os.path.join(temp_dir, "image_0000.png")
    output_image_path = os.path.join(temp_dir, "segmentation_0000.png")
    
    # Create directories
    os.makedirs(temp_dir, exist_ok=True)
    temp_time = time.time() - temp_start
    print(f"   ⏱️  Create temp directories: {temp_time:.3f}s")
        
    try:
        # Save input image with proper naming for nnUNet v2
        save_start = time.time()
        save_image_for_nnunetv2(img, input_image_path)
        save_time = time.time() - save_start
        print(f"   ⏱️  Save input image: {save_time:.3f}s")
        
        # Run prediction using nnUNet v2 predictor (single image, one by one as requested)
        predict_start = time.time()
        
        # Load image using the same I/O class as training (NaturalImage2DIO)
        # This ensures proper axis ordering and properties matching the training data
        img_loaded, props = NaturalImage2DIO().read_images([input_image_path])
        
        # Use the recommended single numpy array prediction method from nnUNet documentation
        # This follows the exact pattern shown in the readme for single image prediction
        # Parameters: input_image, image_properties, segmentation_previous_stage, output_file_truncated, save_or_return_probabilities
        predicted_segmentation = predictor.predict_single_npy_array(
            img_loaded, props, None, None, False
        )
        
        # Ensure we have a valid result
        if predicted_segmentation is None:
            raise RuntimeError("nnUNet prediction returned None")
        
        # nnUNetPredictor returns (C, H, W) where C is 1 for binary segmentation.
        # The validation function expects (H, W). Squeeze the channel dimension.
        if predicted_segmentation.ndim == 3 and predicted_segmentation.shape[0] == 1:
            predicted_segmentation = predicted_segmentation.squeeze(0)
        
        predict_time = time.time() - predict_start
        print(f"   ⏱️  nnUNet v2 prediction (single numpy array): {predict_time:.3f}s")
        
        # Convert binary values (0, 1) to 8-bit values (0, 255) for validation
        if predicted_segmentation.max() == 1:
            predicted_segmentation = (predicted_segmentation * 255).astype(np.uint8)
        
        # CRITICAL FIX: The validation system expects the segmentation to have the same shape as the input image
        # The input image was RGB (H, W, 3) but we converted it to grayscale (H, W) for nnUNet
        # We need to expand the segmentation back to (H, W, 3) to match the original input shape
        original_img_shape = decode_request(request).shape  # Get original input shape
        print(f"   📊 Original input shape: {original_img_shape}")
        print(f"   📊 Segmentation shape before fix: {predicted_segmentation.shape}")
        
        # If the original input was RGB (3 channels), expand the segmentation to match
        if len(original_img_shape) == 3 and original_img_shape[2] == 3:
            # Expand (H, W) to (H, W, 3) by repeating the segmentation across channels
            predicted_segmentation = np.stack([predicted_segmentation] * 3, axis=-1)
            print(f"   📊 Segmentation shape after fix: {predicted_segmentation.shape}")
        
        # Ensure the segmentation has the exact same shape as the original input
        if predicted_segmentation.shape != original_img_shape:
            print(f"   ⚠️  Shape mismatch! Expected {original_img_shape}, got {predicted_segmentation.shape}")
            # Resize segmentation to match original input shape if needed
            if len(original_img_shape) == 3 and len(predicted_segmentation.shape) == 2:
                predicted_segmentation = np.stack([predicted_segmentation] * original_img_shape[2], axis=-1)
            elif len(original_img_shape) == 2 and len(predicted_segmentation.shape) == 3:
                predicted_segmentation = predicted_segmentation[:, :, 0]  # Take first channel
        
        # Final verification that shapes match
        if predicted_segmentation.shape != original_img_shape:
            raise RuntimeError(f"Failed to match shapes: expected {original_img_shape}, got {predicted_segmentation.shape}")
        else:
            print(f"   ✅ Shape verification passed: {predicted_segmentation.shape}")
        
        # Validate the segmentation (can be skipped for speed)
        if not SKIP_VALIDATION:
            validate_start = time.time()
            # Use the original image for validation, not the preprocessed grayscale image
            original_img = decode_request(request)
            validate_segmentation(original_img, predicted_segmentation)
            validate_time = time.time() - validate_start
            print(f"   ⏱️  Validate segmentation: {validate_time:.3f}s")
        
        # Encode the response
        encode_start = time.time()
        encoded_segmentation = encode_request(predicted_segmentation)
        encode_time = time.time() - encode_start
        print(f"   ⏱️  Encode response: {encode_time:.3f}s")
        
        # Calculate prediction time and update counter
        prediction_end_time = time.time()
        prediction_duration = prediction_end_time - prediction_start_time
        last_prediction_time = prediction_duration
        prediction_counter += 1
        
        print(f"🎯 Prediction #{prediction_counter} completed successfully!")
        print(f"   Total predictions so far: {prediction_counter}")
        print(f"   Previous prediction took: {prediction_duration:.2f} seconds")
        print(f"   📊 Breakdown:")
        print(f"      - Decode: {decode_time:.3f}s")
        print(f"      - Preprocess: {preprocess_time:.3f}s")
        print(f"      - Temp dirs: {temp_time:.3f}s")
        print(f"      - Save image: {save_time:.3f}s")
        print(f"      - nnUNet prediction (single numpy array): {predict_time:.3f}s")
        if not SKIP_VALIDATION:
            print(f"      - Validation: {validate_time:.3f}s")
        print(f"      - Encode: {encode_time:.3f}s")
        print(f"      - Total: {prediction_duration:.3f}s")
        
        response = TumorPredictResponseDto(img=encoded_segmentation)
        return response
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise RuntimeError(f"Prediction failed: {e}")
    finally:
        # Clean up temporary directory
        cleanup_start = time.time()
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        cleanup_time = time.time() - cleanup_start
        print(f"   ⏱️  Cleanup temp dir: {cleanup_time:.3f}s")


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
        "service": "tumor-segmentation-api-nnunetv2-pth",
        "uptime": "{}".format(datetime.timedelta(seconds=time.time() - start_time)),
        "model_folder": MODEL_FOLDER,
        "configuration_name": CONFIGURATION_NAME,
        "use_folds": USE_FOLDS,
        "checkpoint_name": CHECKPOINT_NAME,
        "tile_step_size": TILE_STEP_SIZE,
        "use_mirroring": USE_MIRRORING,
        "use_gaussian": USE_GAUSSIAN,
        "perform_everything_on_device": PERFORM_EVERYTHING_ON_DEVICE,
        "validation_api_url": VALIDATION_API_URL,
        "predict_url": PREDICT_URL,
        "predictor_loaded": predictor_loaded,
        "device": str(device) if device else "None",
        "total_predictions": prediction_counter,
    }


@app.get("/")
def index():
    return "Your nnUNet v2 .pth tumor segmentation endpoint is running!"


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if predictor_loaded else "unhealthy",
        "predictor_loaded": predictor_loaded,
        "device": str(device) if device else "None",
        "model_folder": MODEL_FOLDER,
        "configuration_name": CONFIGURATION_NAME,
        "uptime": "{}".format(datetime.timedelta(seconds=time.time() - start_time)),
        "total_predictions": prediction_counter,
    }


if __name__ == "__main__":
    uvicorn.run("api_nnunetv2_pth:app", host=HOST, port=PORT) 