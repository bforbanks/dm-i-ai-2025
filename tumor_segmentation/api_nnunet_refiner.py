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
import scipy.ndimage as ndi

# Load environment variables from .env file
load_dotenv()

# Add project root to path FIRST (parent of tumor_segmentation directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from dtos import TumorPredictRequestDto, TumorPredictResponseDto
from utils import validate_segmentation, encode_request, decode_request
from post_processing import filter_disconnected_tumors_3d

# Import refiner dataset functions for feature generation
from data.refiner_dataset import compute_entropy, compute_signed_distance, compute_y_coord

# nnUNet v2 and PyTorch imports (required)
try:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
    from batchgenerators.utilities.file_and_folder_operations import join
    import torch
    import torch.nn.functional as F
    from models.RefinerUNet.model import RefinerUNet
except ImportError as e:
    raise ImportError(f"nnUNet v2 and PyTorch are required for this API. Please install dependencies: {e}")

# Load configuration from environment variables
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9052"))  # Use same port as original API

# Performance optimization settings
SKIP_VALIDATION = os.getenv("SKIP_VALIDATION", "false").lower() == "true"

# Connected component filtering settings
ENABLE_CONNECTED_COMPONENT_FILTERING = os.getenv("ENABLE_CC_FILTERING", "true").lower() == "true"  # Filter disconnected tumors

# Validation API configuration
VALIDATION_API_URL = os.getenv("VALIDATION_API_URL", "https://cases.dmiai.dk/api/v1/usecases/tumor-segmentation/validate/queue")
VALIDATION_TOKEN = os.getenv("VALIDATION_TOKEN", "18238e2f472643739573ad26f3680c51")
PREDICT_URL = os.getenv("PREDICT_URL", "https://5ff4ebe3f622.ngrok-free.app/predict")

# nnUNet v2 model configuration
MODEL_FOLDER = os.getenv(
    "MODEL_FOLDER",
    "data_nnUNet/results/Dataset001_TumorSegmentation/nnUNetTrainer__nnUNetResEncUNetMPlans__2d_resenc_optimized"
)

# Ensure MODEL_FOLDER points to the base directory, not a fold directory
if MODEL_FOLDER.endswith('/fold_0') or MODEL_FOLDER.endswith('\\fold_0'):
    MODEL_FOLDER = os.path.dirname(MODEL_FOLDER)
    print(f"⚠️  MODEL_FOLDER was pointing to fold_0 directory, corrected to: {MODEL_FOLDER}")

CONFIGURATION_NAME = os.getenv("CONFIGURATION_NAME", "2d_resenc_optimized")
USE_FOLDS = os.getenv("USE_FOLDS", "0")  # Use fold 0 by default
CHECKPOINT_NAME = os.getenv("CHECKPOINT_NAME", "checkpoint_best.pth")

# Refiner model configuration
REFINER_MODEL_PATH = os.getenv(
    "REFINER_MODEL_PATH",
    "data_nnUNet/results/Dataset001_TumorSegmentation/nnUNetTrainer__nnUNetResEncUNetMPlans__CV-reduced/fold_2/refiner_best.pth"
)

# nnUNet v2 prediction settings
TILE_STEP_SIZE = float(os.getenv("TILE_STEP_SIZE", "0.5"))
USE_MIRRORING = os.getenv("USE_MIRRORING", "true").lower() == "true"
USE_GAUSSIAN = os.getenv("USE_GAUSSIAN", "true").lower() == "true"
PERFORM_EVERYTHING_ON_DEVICE = os.getenv("PERFORM_EVERYTHING_ON_DEVICE", "true").lower() == "true"

print(f"nnUNet + Refiner Combined API Configuration:")
print(f"  MODEL_FOLDER: {MODEL_FOLDER}")
print(f"  CONFIGURATION_NAME: {CONFIGURATION_NAME}")
print(f"  USE_FOLDS: {USE_FOLDS}")
print(f"  CHECKPOINT_NAME: {CHECKPOINT_NAME}")
print(f"  REFINER_MODEL_PATH: {REFINER_MODEL_PATH}")
print(f"  TILE_STEP_SIZE: {TILE_STEP_SIZE}")
print(f"  USE_MIRRORING: {USE_MIRRORING}")
print(f"  USE_GAUSSIAN: {USE_GAUSSIAN}")
print(f"  PERFORM_EVERYTHING_ON_DEVICE: {PERFORM_EVERYTHING_ON_DEVICE}")
print(f"  VALIDATION_API_URL: {VALIDATION_API_URL}")
print(f"  PREDICT_URL: {PREDICT_URL}")
print(f"  SKIP_VALIDATION: {SKIP_VALIDATION}")
print(f"  ENABLE_CONNECTED_COMPONENT_FILTERING: {ENABLE_CONNECTED_COMPONENT_FILTERING}")


def initialize_nnunet_predictor():
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
            use_folds=(int(USE_FOLDS),),
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
        raise RuntimeError(f"nnUNet predictor initialization failed: {e}")


def initialize_refiner_model(device):
    """Initialize the Refiner model and load weights"""
    try:
        print(f"🔄 Loading Refiner model into memory...")
        refiner_start_time = time.time()
        
        # Check if refiner model file exists
        if not Path(REFINER_MODEL_PATH).exists():
            raise FileNotFoundError(f"Refiner model not found at: {REFINER_MODEL_PATH}")
        
        # Determine model configuration from a sample (we'll set reasonable defaults)
        # For the refiner model: softmax (2 classes) + petmr (1 channel) + auxiliary (3 channels) = 6 channels
        in_channels = 6  # softmax (2) + petmr (1) + entropy + distance + y_coord (3)
        num_classes = 2  # binary segmentation (background, tumor)
        
        # Create model instance (using default hyperparameters for inference)
        refiner_model = RefinerUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            lr=1e-3,  # Not used in inference
            weight_decay=1e-5,  # Not used in inference
            warmup_iters=500,  # Not used in inference
            max_epochs=100,  # Not used in inference
            dice_weight=1.0,  # Not used in inference
            bce_weight=0.5,  # Not used in inference
            surface_weight=0.1  # Not used in inference
        )
        
        # Load the trained weights
        state_dict = torch.load(REFINER_MODEL_PATH, map_location=device)
        refiner_model.load_state_dict(state_dict)
        
        # Move to device and set to evaluation mode
        refiner_model.to(device)
        refiner_model.eval()
        
        refiner_load_time = time.time() - refiner_start_time
        print(f"✅ Refiner model loaded in {refiner_load_time:.2f} seconds")
        print(f"   Model path: {REFINER_MODEL_PATH}")
        print(f"   Input channels: {in_channels}")
        print(f"   Number of classes: {num_classes}")
        
        return refiner_model
        
    except Exception as e:
        print(f"❌ Failed to initialize Refiner model: {e}")
        raise RuntimeError(f"Refiner model initialization failed: {e}")


def preprocess_image_for_nnunetv2(img: np.ndarray) -> np.ndarray:
    """
    Preprocess image for nnUNet v2 prediction to match training data format.
    
    Args:
        img: Input image as numpy array (H, W, C) or (H, W)
        
    Returns:
        Preprocessed image ready for nnUNet v2
    """
    # Ensure image is grayscale (single channel) - same as training data
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            # Convert RGB to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img.shape[2] == 1:
            img = img[:, :, 0]
    
    # Ensure image is in uint8 format with 0-255 range
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    # Ensure the image is in the expected range (0-255)
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    return img


def save_image_for_nnunetv2(img: np.ndarray, output_path: str) -> str:
    """Save image in the format expected by nnUNet v2."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path


def generate_refiner_features(nnunet_probabilities: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    """
    Generate features for the refiner model from nnUNet probabilities and original image.
    
    Args:
        nnunet_probabilities: nnUNet output probabilities (H, W) with values [0, 1]
        original_image: Original input image (H, W) normalized to [0, 1]
        
    Returns:
        Combined feature tensor (6, H, W) ready for refiner model
    """
    # 1. Convert probabilities to 2-class softmax format
    background_prob = 1.0 - nnunet_probabilities
    tumor_prob = nnunet_probabilities
    softmax = np.stack([background_prob, tumor_prob], axis=0)  # (2, H, W)
    
    # 2. Compute auxiliary channels
    entropy = compute_entropy(softmax)  # (H, W)
    
    # Create binary mask for distance computation (use tumor probability > 0.5)
    binary_mask = (tumor_prob > 0.5).astype(np.uint8)
    distance = compute_signed_distance(binary_mask)  # (H, W)
    
    # Y-coordinate map
    y_coord = compute_y_coord(entropy.shape)  # (H, W)
    
    # 3. Prepare PET/MR image (original image normalized to [0, 1])
    if original_image.ndim == 2:
        petmr = original_image[None, :, :]  # Add channel dimension (1, H, W)
    else:
        petmr = original_image
    
    # 4. Stack all features following RefinerDataset convention
    # Stack auxiliary channels
    extra = np.stack([entropy, distance, y_coord], axis=0)  # (3, H, W)
    
    # Combine: softmax + petmr + auxiliary (modalities_first=True by default)
    combined_features = np.concatenate([softmax, petmr, extra], axis=0)  # (6, H, W)
    
    return combined_features.astype(np.float32)


# Initialize models at startup
try:
    nnunet_predictor, device = initialize_nnunet_predictor()
    refiner_model = initialize_refiner_model(device)
    models_loaded = True
except Exception as e:
    print(f"❌ Model initialization failed: {e}")
    models_loaded = False
    nnunet_predictor = None
    refiner_model = None
    device = None

app = FastAPI()
start_time = time.time()

# Global prediction counter and timing
prediction_counter = 0
last_prediction_time = 0.0


def queue_validation_attempt(predict_url: str | None = None) -> dict:
    """Queue a validation attempt against the online validation dataset."""
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
    """Combined nnUNet + Refiner prediction endpoint - returns refined binary mask"""
    global prediction_counter, last_prediction_time
    
    # Start timing
    prediction_start_time = time.time()
    
    if not models_loaded or nnunet_predictor is None or refiner_model is None:
        raise RuntimeError("Models are not loaded. Please check model initialization.")
    
    # Decode the request
    img: np.ndarray = decode_request(request)
    original_img_shape = img.shape
    
    print(f"🔍 Processing image with shape: {original_img_shape}")
    
    # Preprocess image for nnUNet v2
    nnunet_img = preprocess_image_for_nnunetv2(img)
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp(prefix="nnunet_refiner_")
    input_image_path = os.path.join(temp_dir, "image_0000.png")
    
    try:
        # ========================================
        # STEP 1: Run nnUNet v2 to get probabilities
        # ========================================
        print("🧠 Step 1: Running nnUNet v2 prediction...")
        
        # Save input image with proper naming for nnUNet v2
        save_image_for_nnunetv2(nnunet_img, input_image_path)
        
        # Load image using the same I/O class as training
        img_loaded, props = NaturalImage2DIO().read_images([input_image_path])
        
        # Get probabilities from nnUNet
        nnunet_segmentation, nnunet_probabilities = nnunet_predictor.predict_single_npy_array(
            img_loaded, props, None, None, True  # save_or_return_probabilities=True
        )
        
        if nnunet_probabilities is None:
            raise RuntimeError("nnUNet prediction returned None probabilities")
        
        print(f"   📊 nnUNet probabilities shape: {nnunet_probabilities.shape}")
        
        # Handle different nnUNet probability shapes
        # nnUNet can return either (2, H, W) or (2, 1, H, W)
        if nnunet_probabilities.ndim == 4 and nnunet_probabilities.shape[1] == 1:
            # Shape: (2, 1, H, W) -> squeeze out the singleton dimension
            nnunet_probabilities = nnunet_probabilities.squeeze(1)  # Shape: (2, H, W)
        
        # Extract tumor probabilities (class 1)
        if nnunet_probabilities.ndim == 3 and nnunet_probabilities.shape[0] == 2:
            tumor_probabilities = nnunet_probabilities[1]  # Shape: (H, W)
        else:
            raise RuntimeError(f"Unexpected nnUNet probability shape after processing: {nnunet_probabilities.shape}")
        
        print(f"   📊 Tumor probabilities range: [{tumor_probabilities.min():.4f}, {tumor_probabilities.max():.4f}]")
        
        # ========================================
        # STEP 2: Generate features for Refiner
        # ========================================
        print("🔧 Step 2: Generating refiner features...")
        
        # Normalize original image to [0, 1] for feature generation
        if len(img.shape) == 3:
            # Convert RGB to grayscale for consistency
            original_normalized = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            original_normalized = img.astype(np.float32) / 255.0
        
        # Generate combined features
        refiner_features = generate_refiner_features(tumor_probabilities, original_normalized)
        print(f"   📊 Refiner features shape: {refiner_features.shape}")
        
        # ========================================
        # STEP 3: Run Refiner model
        # ========================================
        print("🔬 Step 3: Running Refiner model...")
        
        # Convert to tensor and add batch dimension
        refiner_input = torch.from_numpy(refiner_features).unsqueeze(0).to(device)  # (1, 6, H, W)
        
        with torch.no_grad():
            # Run refiner model
            refined_logits, refined_probs = refiner_model(refiner_input)
            
            # Extract refined probabilities (remove batch dimension)
            refined_probs = refined_probs.squeeze(0).cpu().numpy()  # (2, H, W)
            
            # Get tumor probability (class 1)
            refined_tumor_prob = refined_probs[1]  # (H, W)
            
            # Create binary mask (threshold at 0.5)
            refined_mask = (refined_tumor_prob > 0.5).astype(np.uint8)
            
        print(f"   📊 Refined probabilities range: [{refined_tumor_prob.min():.4f}, {refined_tumor_prob.max():.4f}]")
        print(f"   📊 Refined mask: {refined_mask.sum()} tumor pixels out of {refined_mask.size} total")
        
        # ========================================
        # STEP 4: Prepare output
        # ========================================
        print("📦 Step 4: Preparing output...")
        
        # Convert binary mask to 8-bit values (0, 255) for validation
        result_to_encode = (refined_mask * 255).astype(np.uint8)
        
        # Handle shape compatibility with original input
        if len(original_img_shape) == 3 and original_img_shape[2] == 3:
            # Expand (H, W) to (H, W, 3) by repeating across channels
            result_to_encode = np.stack([result_to_encode] * 3, axis=-1)
        
        # Apply connected component filtering to remove disconnected tumors (if enabled)
        if ENABLE_CONNECTED_COMPONENT_FILTERING:
            print("🔧 Step 5: Applying connected component filtering to remove disconnected tumors...")
            original_img_for_filtering = decode_request(request)  # Get original input for body detection
            result_to_encode = filter_disconnected_tumors_3d(original_img_for_filtering, result_to_encode)
            print(f"   ✅ Connected component filtering completed")
        else:
            print("⏭️  Step 5: Connected component filtering disabled")
        
        # Final verification that shapes match
        if result_to_encode.shape != original_img_shape:
            raise RuntimeError(f"Failed to match shapes: expected {original_img_shape}, got {result_to_encode.shape}")
        
        print(f"   ✅ Final output shape: {result_to_encode.shape}")
        
        # Validate the segmentation (if enabled)
        if not SKIP_VALIDATION:
            validate_segmentation(img, result_to_encode)
        
        # Encode the response
        encoded_result = encode_request(result_to_encode)
        
        # Calculate prediction time and update counter
        prediction_end_time = time.time()
        prediction_duration = prediction_end_time - prediction_start_time
        last_prediction_time = prediction_duration
        prediction_counter += 1
        
        print(f"🎯 Combined prediction #{prediction_counter} completed successfully in {prediction_duration:.2f} seconds")
        print(f"   🧠 nnUNet → 🔬 Refiner → 📋 Final binary mask")
        
        response = TumorPredictResponseDto(img=encoded_result)
        return response
        
    except Exception as e:
        print(f"Error during combined prediction: {e}")
        raise RuntimeError(f"Combined prediction failed: {e}")
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


@app.post("/queue-validation")
def queue_validation_endpoint(predict_url: str | None = None):
    """Endpoint to queue a validation attempt against the online validation dataset."""
    result = queue_validation_attempt(predict_url)
    return result


@app.get("/validation-status/{attempt_uuid}")
def check_validation_status_endpoint(attempt_uuid: str):
    """Endpoint to check the status of a validation attempt and get the score if completed."""
    result = check_validation_status(attempt_uuid)
    return result


def check_validation_status(attempt_uuid: str) -> dict:
    """Check the status of a validation attempt and get the score if completed."""
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
        "service": "tumor-segmentation-api-nnunet-refiner",
        "uptime": "{}".format(datetime.timedelta(seconds=time.time() - start_time)),
        "model_folder": MODEL_FOLDER,
        "configuration_name": CONFIGURATION_NAME,
        "use_folds": USE_FOLDS,
        "checkpoint_name": CHECKPOINT_NAME,
        "refiner_model_path": REFINER_MODEL_PATH,
        "tile_step_size": TILE_STEP_SIZE,
        "use_mirroring": USE_MIRRORING,
        "use_gaussian": USE_GAUSSIAN,
        "perform_everything_on_device": PERFORM_EVERYTHING_ON_DEVICE,
        "validation_api_url": VALIDATION_API_URL,
        "predict_url": PREDICT_URL,
        "models_loaded": models_loaded,
        "device": str(device) if device else "None",
        "total_predictions": prediction_counter,
    }


@app.get("/")
def index():
    return "Your nnUNet + Refiner combined tumor segmentation endpoint is running!"


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "device": str(device) if device else "None",
        "model_folder": MODEL_FOLDER,
        "refiner_model_path": REFINER_MODEL_PATH,
        "uptime": "{}".format(datetime.timedelta(seconds=time.time() - start_time)),
        "total_predictions": prediction_counter,
    }


if __name__ == "__main__":
    uvicorn.run("api_nnunet_refiner:app", host=HOST, port=PORT)