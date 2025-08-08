"""
Batch driver that runs InteractiveMaskBuilder over a directory of images
and automatically skips images whose final mask already exists.

It also supports resuming after interruption:  once an image's final mask
has been produced ( <output_dir>/<base>_mask.png ) it is considered done.

Usage:
    python multi_image_mask_builder.py <images_dir> <output_dir>

Example:
    python multi_image_mask_builder.py validation2 validation2/ground_truth
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from interactive_mask_builder import InteractiveMaskBuilder
from remote_validation import TumorSegmentationValidator


MASK_SUFFIX = "_mask.png"  # output mask naming scheme

def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python multi_image_mask_builder.py <images_dir> <output_dir> [--revalidate-existing]")
        print("Example: python multi_image_mask_builder.py validation2 validation2/ground_truth --revalidate-existing")
        print("\nDuring processing the script writes the current image id (e.g. 001) to"
              " <output_dir>/current_image_id.txt so external tools can pick it up.")
        sys.exit(1)

    images_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    revalidate_existing = False
    if len(sys.argv) == 4:
        if sys.argv[3] == "--revalidate-existing":
            revalidate_existing = True
        else:
            print("Unknown third argument. Did you mean --revalidate-existing?")
            sys.exit(1)
    if not images_dir.is_dir():
        print(f"❌ Images dir not found: {images_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gather image files (png)
    image_paths = sorted(glob.glob(str(images_dir / "image_*.png")))
    if not image_paths:
        print(f"No images like 'image_*.png' in {images_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images. Starting processing…")

    # Set up validator once so login is done only a single time
    validator = TumorSegmentationValidator(headless=False)
    validator.navigate_to_site()

    for idx, img_path in enumerate(image_paths, 1):
        base = Path(img_path).stem  # e.g. image_001
        out_mask_path = output_dir / f"{base}{MASK_SUFFIX}"

        ckpt_path = out_mask_path.with_suffix(out_mask_path.suffix + ".ckpt.pkl")
        if out_mask_path.exists() and not ckpt_path.exists() and not revalidate_existing:
            print(f"[{idx}/{len(image_paths)}] ✅ {base} already done – skipping.")
            continue
        if ckpt_path.exists():
            print(f"[{idx}/{len(image_paths)}] 🔄 Resuming unfinished mask (checkpoint found).")
        elif out_mask_path.exists() and revalidate_existing:
            print(f"[{idx}/{len(image_paths)}] ♻️  Revalidating existing mask (no checkpoint).")

        print(f"\n[{idx}/{len(image_paths)}] === Processing {base} ===")
        # Write current image id (e.g. 001) to helper file for API integration
        current_id = base.split("_")[1]  # assumes naming like image_001
        id_file_path = output_dir / "current_image_id.txt"
        with open(id_file_path, "w", encoding="utf-8") as idf:
            idf.write(current_id)
        # Optionally echo for user
        print(f"→ current_image_id.txt updated with {current_id}")
        img = np.array(Image.open(img_path))

        # Provide initial guess mask if exists
        init_mask_path = output_dir / f"{base}_initial.png"
        init_mask_str = str(init_mask_path) if init_mask_path.exists() else None

        builder = InteractiveMaskBuilder(img, str(out_mask_path), init_mask_str, validator=validator)
        try:
            builder.run()
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting…")
            sys.exit(0)
        except Exception as e:
            print(f"⚠️  Error while processing {base}: {e}")
            print("Continuing with next image…")

    print("\nAll images processed.")
    validator.close()


if __name__ == "__main__":
    main()
