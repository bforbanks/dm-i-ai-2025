#!/usr/bin/env python3
"""
Convert tumor segmentation dataset to nnUNet v2 format.

This script converts our current data structure to the nnUNet v2 format:
- Renames files to follow nnUNet v2 naming convention
- Creates proper folder structure
- Generates dataset.json with correct metadata
- Creates empty masks for control images

Based on nnUNet v2 documentation requirements:
- Images: {CASE_ID}_{0000}.png
- Labels: {CASE_ID}.png
- Folder structure: imagesTr/, labelsTr/, dataset.json
"""

import os
import json
import shutil
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2


def create_empty_mask(image_path: str) -> np.ndarray:
    """
    Create an empty mask (all zeros) with the same shape as the input image.

    Args:
        image_path: Path to the input image

    Returns:
        Empty mask as numpy array
    """
    # Read the image to get its dimensions
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Create empty mask with same shape
    mask = np.zeros_like(img, dtype=np.uint8)
    return mask


def convert_segmentation_to_binary(seg_path: str) -> np.ndarray:
    """
    Convert segmentation mask to binary format (0 for background, 1 for tumor).

    Args:
        seg_path: Path to segmentation mask

    Returns:
        Binary mask as numpy array
    """
    # Read segmentation mask
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    if seg is None:
        raise ValueError(f"Could not read segmentation: {seg_path}")

    # Convert to binary (any non-zero pixel becomes 1)
    binary_mask = (seg > 0).astype(np.uint8)
    return binary_mask


def convert_dataset_to_nnunetv2_format(
    source_data_dir: str = "data",
    output_dir: str = "data_nnUNet",
    dataset_id: int = 1,
    dataset_name: str = "TumorSegmentation",
):
    """
    Convert the tumor segmentation dataset to nnUNet v2 format.

    Args:
        source_data_dir: Path to source data directory
        output_dir: Path to output directory
        dataset_id: Dataset ID (3-digit number)
        dataset_name: Dataset name
    """

    # Create output directory structure
    dataset_folder = f"Dataset{dataset_id:03d}_{dataset_name}"
    output_path = Path(output_dir) / dataset_folder

    images_tr_dir = output_path / "imagesTr"
    labels_tr_dir = output_path / "labelsTr"

    # Create directories
    images_tr_dir.mkdir(parents=True, exist_ok=True)
    labels_tr_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Converting dataset to nnUNet v2 format...")
    print(f"   Source: {source_data_dir}")
    print(f"   Output: {output_path}")

    # Source directories
    patients_img_dir = Path(source_data_dir) / "patients" / "imgs"
    patients_label_dir = Path(source_data_dir) / "patients" / "labels"
    controls_img_dir = Path(source_data_dir) / "controls" / "imgs"
    controls_tumor_img_dir = Path(source_data_dir) / "controls_tumor" / "imgs"
    controls_tumor_label_dir = Path(source_data_dir) / "controls_tumor" / "labels"

    # Check if source directories exist
    if not patients_img_dir.exists():
        raise ValueError(f"Patients images directory not found: {patients_img_dir}")
    if not patients_label_dir.exists():
        raise ValueError(f"Patients labels directory not found: {patients_label_dir}")
    if not controls_img_dir.exists():
        raise ValueError(f"Controls images directory not found: {controls_img_dir}")
    
    # Check for optional synthetic tumor dataset
    has_synthetic_tumors = controls_tumor_img_dir.exists() and controls_tumor_label_dir.exists()
    if has_synthetic_tumors:
        print(f"   ✅ Found synthetic tumor dataset: {controls_tumor_img_dir}")
    else:
        print(f"   ⚠️  No synthetic tumor dataset found (optional): {controls_tumor_img_dir}")

    # Process patient images and labels
    patient_images = list(patients_img_dir.glob("patient_*.png"))
    patient_labels = list(patients_label_dir.glob("segmentation_*.png"))

    print(f"   Found {len(patient_images)} patient images")
    print(f"   Found {len(patient_labels)} patient labels")

    # Create mapping from patient number to label
    patient_to_label = {}
    for label_path in patient_labels:
        # Extract patient number from segmentation filename
        # segmentation_001.png -> patient_001
        patient_num = "patient_" + label_path.stem.replace("segmentation_", "")
        patient_to_label[patient_num] = label_path

    # Process patient data
    patient_count = 0
    for img_path in patient_images:
        # Extract patient number
        patient_num = img_path.stem  # patient_181

        # Check if we have a corresponding label
        if patient_num not in patient_to_label:
            print(f"   ⚠️  Warning: No label found for {patient_num}, skipping...")
            continue

        # Copy and ensure image is grayscale with nnUNet v2 naming convention
        new_img_name = f"{patient_num}_0000.png"
        new_img_path = images_tr_dir / new_img_name

        # Read and convert to grayscale if needed
        with Image.open(img_path) as img:
            if img.mode != "L":
                img = img.convert("L")
            img.save(new_img_path)

        # Convert and save label
        label_path = patient_to_label[patient_num]
        binary_mask = convert_segmentation_to_binary(str(label_path))

        new_label_name = f"{patient_num}.png"
        new_label_path = labels_tr_dir / new_label_name

        # Save binary mask as PNG - ensure values are exactly 0 and 1
        # Use mode='L' to ensure 8-bit grayscale, then convert back to 0/1
        img = Image.fromarray(binary_mask, mode="L")
        img.save(new_label_path)

        # Verify the saved image has correct values
        saved_mask = cv2.imread(str(new_label_path), cv2.IMREAD_GRAYSCALE)
        if saved_mask is not None:
            # Convert any non-zero values back to 1
            saved_mask = (saved_mask > 0).astype(np.uint8)
            # Save again with correct values
            Image.fromarray(saved_mask, mode="L").save(new_label_path)

        patient_count += 1

    # Process control images (create empty masks)
    # Look for any PNG files in controls directory
    control_images = list(controls_img_dir.glob("*.png"))
    print(f"   Found {len(control_images)} control images")

    control_count = 0
    for img_path in control_images:
        # Extract control number and ensure it starts with "control_"
        control_num = img_path.stem
        if not control_num.startswith("control_"):
            control_num = f"control_{control_num}"

        # Copy and ensure image is grayscale with nnUNet v2 naming convention
        new_img_name = f"{control_num}_0000.png"
        new_img_path = images_tr_dir / new_img_name

        # Read and convert to grayscale if needed
        with Image.open(img_path) as img:
            if img.mode != "L":
                img = img.convert("L")
            img.save(new_img_path)

        # Create empty mask
        empty_mask = create_empty_mask(str(img_path))

        new_label_name = f"{control_num}.png"
        new_label_path = labels_tr_dir / new_label_name

        # Save empty mask as PNG - ensure values are exactly 0
        img = Image.fromarray(empty_mask, mode="L")
        img.save(new_label_path)

        # Verify the saved image has correct values (should be all zeros)
        saved_mask = cv2.imread(str(new_label_path), cv2.IMREAD_GRAYSCALE)
        if saved_mask is not None:
            # Ensure it's all zeros
            saved_mask = np.zeros_like(saved_mask, dtype=np.uint8)
            Image.fromarray(saved_mask, mode="L").save(new_label_path)

        control_count += 1

    # Process synthetic tumor images and labels (if available)
    synthetic_tumor_count = 0
    if has_synthetic_tumors:
        synthetic_tumor_images = list(controls_tumor_img_dir.glob("*.png"))
        print(f"   Found {len(synthetic_tumor_images)} synthetic tumor images")

        for img_path in synthetic_tumor_images:
            # Extract control number and ensure proper naming for synthetic tumors
            control_num = img_path.stem  # control_XXX
            
            # Check if corresponding label exists
            label_path = controls_tumor_label_dir / f"{control_num}.png"
            if not label_path.exists():
                print(f"   ⚠️  Warning: No label found for synthetic tumor {control_num}, skipping...")
                continue

            # Use controls_tumor prefix for nnUNet naming
            synthetic_case_name = f"controls_tumor{control_num.replace('control_', '')}"

            # Copy and ensure image is grayscale with nnUNet v2 naming convention
            new_img_name = f"{synthetic_case_name}_0000.png"
            new_img_path = images_tr_dir / new_img_name

            # Read and convert to grayscale if needed
            with Image.open(img_path) as img:
                if img.mode != "L":
                    img = img.convert("L")
                img.save(new_img_path)

            # Convert and save label (these are real tumor masks, not empty like controls)
            binary_mask = convert_segmentation_to_binary(str(label_path))

            new_label_name = f"{synthetic_case_name}.png"
            new_label_path = labels_tr_dir / new_label_name

            # Save binary mask as PNG - ensure values are exactly 0 and 1
            img = Image.fromarray(binary_mask, mode="L")
            img.save(new_label_path)

            # Verify the saved image has correct values
            saved_mask = cv2.imread(str(new_label_path), cv2.IMREAD_GRAYSCALE)
            if saved_mask is not None:
                # Convert any non-zero values back to 1
                saved_mask = (saved_mask > 0).astype(np.uint8)
                # Save again with correct values
                Image.fromarray(saved_mask, mode="L").save(new_label_path)

            synthetic_tumor_count += 1

    # Create dataset.json
    total_training_cases = patient_count + control_count + synthetic_tumor_count

    dataset_json = {
        "channel_names": {
            "0": "PET"  # Single channel PET images
        },
        "labels": {"background": 0, "tumor": 1},
        "numTraining": total_training_cases,
        "file_ending": ".png",
    }

    # Save dataset.json
    dataset_json_path = output_path / "dataset.json"
    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"✅ Conversion completed successfully!")
    print(f"   Patient cases: {patient_count}")
    print(f"   Control cases: {control_count}")
    if has_synthetic_tumors:
        print(f"   Synthetic tumor cases: {synthetic_tumor_count}")
    print(f"   Total training cases: {total_training_cases}")
    print(f"   Dataset JSON: {dataset_json_path}")

    # Print sample of converted files
    print(f"\n📁 Sample converted files:")
    sample_images = list(images_tr_dir.glob("*_0000.png"))[:5]
    sample_labels = list(labels_tr_dir.glob("*.png"))[:5]

    print(f"   Images:")
    for img in sample_images:
        print(f"     {img.name}")

    print(f"   Labels:")
    for label in sample_labels:
        print(f"     {label.name}")

    print(f"\n🎯 Next steps:")
    print(f"   1. Set nnUNet environment variables:")
    print(f"      export nnUNet_raw='{output_dir}'")
    print(f"      export nnUNet_preprocessed='{output_dir}/preprocessed'")
    print(f"      export nnUNet_results='{output_dir}/results'")
    print(f"   2. Run nnUNet planning and preprocessing:")
    print(
        f"      nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity"
    )
    print(f"   3. Train the model:")
    print(f"      nnUNetv2_train {dataset_id} 2d 0 --npz")

    return output_path


def verify_conversion(output_path: Path):
    """
    Verify that the conversion was successful.

    Args:
        output_path: Path to the converted dataset
    """
    print(f"\n🔍 Verifying conversion...")

    # Check directory structure
    images_tr_dir = output_path / "imagesTr"
    labels_tr_dir = output_path / "labelsTr"
    dataset_json_path = output_path / "dataset.json"

    if not images_tr_dir.exists():
        print(f"   ❌ imagesTr directory not found")
        return False

    if not labels_tr_dir.exists():
        print(f"   ❌ labelsTr directory not found")
        return False

    if not dataset_json_path.exists():
        print(f"   ❌ dataset.json not found")
        return False

    # Count files
    image_files = list(images_tr_dir.glob("*_0000.png"))
    label_files = list(labels_tr_dir.glob("*.png"))

    print(f"   ✅ Images: {len(image_files)}")
    print(f"   ✅ Labels: {len(label_files)}")

    # Check naming convention
    for img_file in image_files:
        if not img_file.name.endswith("_0000.png"):
            print(f"   ❌ Invalid image naming: {img_file.name}")
            return False

    # Check that each image has a corresponding label
    image_ids = {f.stem.replace("_0000", "") for f in image_files}
    label_ids = {f.stem for f in label_files}

    missing_labels = image_ids - label_ids
    if missing_labels:
        print(f"   ❌ Missing labels for: {missing_labels}")
        return False

    extra_labels = label_ids - image_ids
    if extra_labels:
        print(f"   ❌ Extra labels without images: {extra_labels}")
        return False

    # Load and verify dataset.json
    try:
        with open(dataset_json_path, "r") as f:
            dataset_json = json.load(f)

        required_keys = ["channel_names", "labels", "numTraining", "file_ending"]
        for key in required_keys:
            if key not in dataset_json:
                print(f"   ❌ Missing key in dataset.json: {key}")
                return False

        print(f"   ✅ dataset.json is valid")

    except Exception as e:
        print(f"   ❌ Error reading dataset.json: {e}")
        return False

    print(f"   ✅ Conversion verification passed!")
    return True


def main():
    """Main function to run the conversion."""
    print("🚀 Tumor Segmentation Dataset to nnUNet v2 Converter")
    print("=" * 60)

    # CLI arguments
    parser = argparse.ArgumentParser(
        description=(
            "Convert tumor segmentation dataset to nnUNet v2 format. "
            "Use --dataset-id 2 to create Dataset002_* for nnUNet (-d 2)."
        )
    )
    parser.add_argument(
        "--source-data-dir",
        type=str,
        default="data",
        help="Path to the source data directory (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_nnUNet",
        help="Path to the nnUNet raw/output directory (default: data_nnUNet)",
    )
    parser.add_argument(
        "--dataset-id",
        "-d",
        type=int,
        default=1,
        help="Numeric dataset ID (e.g., 1 -> Dataset001_*, 2 -> Dataset002_*)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="TumorSegmentation",
        help="Dataset name suffix (default: TumorSegmentation)",
    )

    args = parser.parse_args()
    source_data_dir = args.source_data_dir
    output_dir = args.output_dir
    dataset_id = args.dataset_id
    dataset_name = args.dataset_name

    try:
        # Convert dataset
        output_path = convert_dataset_to_nnunetv2_format(
            source_data_dir=source_data_dir,
            output_dir=output_dir,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
        )

        # Verify conversion
        verify_conversion(output_path)

        print(f"\n🎉 Dataset conversion completed successfully!")
        print(f"   Output directory: {output_path}")

    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
