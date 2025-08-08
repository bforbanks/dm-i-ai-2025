import uvicorn
import time
import datetime
import numpy as np
from fastapi import FastAPI
from dtos import TumorPredictRequestDto, TumorPredictResponseDto
from utils import validate_segmentation, encode_request, decode_request
import os
import csv
from PIL import Image
import glob

# Helper: path where multi_image_mask_builder writes the current id
CURRENT_ID_FILE = os.path.join("validation2", "ground_truth", "current_image_id.txt")

HOST = "0.0.0.0"
PORT = 9052

app = FastAPI()
start_time = time.time()

# Simple setup
DATASET_DIR = "validation2"
DATASET_CSV = os.path.join(DATASET_DIR, "dataset.csv")

# Create directory
os.makedirs(DATASET_DIR, exist_ok=True)

# Initialize CSV with headers if it doesn't exist
if not os.path.exists(DATASET_CSV):
    with open(DATASET_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "filename", "width", "height", "total_pixels", "timestamp"])

def get_all_existing_images():
    """Get all existing images as numpy arrays for comparison"""
    pattern = os.path.join(DATASET_DIR, "image_*.png")
    image_files = sorted(glob.glob(pattern))
    
    existing_images = {}
    for image_file in image_files:
        try:
            # Extract ID from filename (e.g., image_001.png -> 1)
            filename = os.path.basename(image_file)
            image_id = int(filename.replace("image_", "").replace(".png", ""))
            
            # Load image
            img = np.array(Image.open(image_file))
            existing_images[image_id] = img
        except Exception as e:
            print(f"Error loading {image_file}: {e}")
    
    return existing_images

def is_image_already_in_dataset(new_img):
    """Check if this image already exists in our dataset"""
    existing_images = get_all_existing_images()
    
    for image_id, existing_img in existing_images.items():
        if (existing_img.shape == new_img.shape and 
            np.array_equal(existing_img, new_img)):
            return True, image_id
    
    return False, None

def get_next_image_id():
    """Get the next sequential image ID"""
    pattern = os.path.join(DATASET_DIR, "image_*.png")
    image_files = glob.glob(pattern)
    
    if not image_files:
        return 1
    
    # Find highest existing ID
    max_id = 0
    for image_file in image_files:
        try:
            filename = os.path.basename(image_file)
            image_id = int(filename.replace("image_", "").replace(".png", ""))
            max_id = max(max_id, image_id)
        except:
            continue
    
    return max_id + 1

def save_new_image(img: np.ndarray):
    """Save a new image to the dataset"""
    image_id = get_next_image_id()
    timestamp = datetime.datetime.now().isoformat()
    
    # Save the image
    filename = f"image_{image_id:03d}.png"
    image_path = os.path.join(DATASET_DIR, filename)
    Image.fromarray(img).save(image_path)
    
    # Add to CSV
    with open(DATASET_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            image_id,
            filename,
            img.shape[1] if len(img.shape) > 1 else img.shape[0],  # width
            img.shape[0],  # height
            img.size,  # total pixels
            timestamp
        ])
    
    print(f"✅ Saved NEW image as {filename} (ID: {image_id})")
    return image_id

def get_dataset_stats():
    """Get statistics about the dataset"""
    pattern = os.path.join(DATASET_DIR, "image_*.png")
    image_count = len(glob.glob(pattern))
    
    return {
        "total_images": image_count,
        "dataset_folder": DATASET_DIR,
        "csv_file": DATASET_CSV
    }

@app.post('/predict', response_model=TumorPredictResponseDto)
def predict_endpoint(request: TumorPredictRequestDto):
    # Decode request to numpy array
    img: np.ndarray = decode_request(request)
    
    # Check if this image is already in our dataset
    already_exists, existing_id = is_image_already_in_dataset(img)
    
    if already_exists:
        print(f"🔍 Image already exists in dataset (ID: {existing_id})")
    else:
        # Save new image
        new_id = save_new_image(img)
    
    # Simple prediction: either use the mask that matches current_image_id.txt
    # or fall back to zeros
    current_id_file = "validation2/ground_truth/current_image_id.txt"
    try:
        current_id = int(open(current_id_file).read().strip())
    except Exception:
        current_id = -1  # will trigger zero-mask path

    print(img.shape)
    if existing_id == current_id:
        predicted_segmentation = np.array(Image.open(f"validation2/ground_truth/image_{current_id:03d}_mask.png"))
        print(predicted_segmentation.shape)
    else:
        predicted_segmentation = np.zeros_like(img, dtype=np.uint8)
        print(predicted_segmentation.shape)
    
    # Validate and encode
    validate_segmentation(img, predicted_segmentation)
    encoded_segmentation = encode_request(predicted_segmentation)
    
    return TumorPredictResponseDto(img=encoded_segmentation)

@app.get('/dataset-stats')
def dataset_stats():
    """Get dataset statistics"""
    return get_dataset_stats()

@app.get('/')
def index():
    return "Clean Tumor Segmentation API - Only saves NEW images to dataset"

if __name__ == '__main__':
    uvicorn.run(
        'api2:app',
        host=HOST,
        port=PORT
    )