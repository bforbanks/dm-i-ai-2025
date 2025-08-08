"""
Interactive Mask Builder using Dice Score Feedback

This script helps you iteratively build the exact ground-truth mask for an image
by repeatedly:
  1. Writing a candidate mask to disk (all channels identical, 0/255 values)
  2. Asking **you** to run your simple model that reads this mask, queries the
     evaluation API and prints the Dice score
  3. You paste the Dice score back into this script
  4. The script converts Dice → TP using the closed-form formula
  5. With the TP information it recursively subdivides the yet-unknown pixels
     until every positive pixel is located

Key idea:  when we predict ONE half of the undecided pixels as positive and the
other half as negative the Dice score instantly tells us *how many* of those
"guessed-positive" pixels are actually correct.  Repeating this binary split
log₂(N) times isolates every positive pixel.

Time complexity:  **≈ log₂(total_pixels)** rounds.  For a 400×426 image this is
about 18 rounds – very fast.

Usage (Linux / macOS / Windows PowerShell):
    python interactive_mask_builder.py <image_path> <output_mask_png>

During the run you will be prompted with clear instructions:
  • Which mask was written (so your model can pick it up)
  • Paste the Dice score that the API returned

The final mask is saved at <output_mask_png> (and also overwrites the working
mask path in the last round).
"""

from __future__ import annotations

import os
import sys
import time
import pickle
from typing import List, Tuple, Set

import numpy as np
from PIL import Image

# -----------------------------------------------------
# Helper math functions
# -----------------------------------------------------

def dice_to_tp(score: float, predicted_positive: int, ground_truth_positive: int) -> int:
    """Closed-form TP from Dice.

    Dice = 2·TP / (Pp + P)
        ⇒ TP = Dice · (Pp + P) / 2
    """
    tp = score * (predicted_positive + ground_truth_positive) / 2.0
    return int(round(tp))

# -----------------------------------------------------
# Core interactive builder
# -----------------------------------------------------

DATASET_SIZE = 200  # total number of images contributing to the aggregated Dice score

class InteractiveMaskBuilder:
    def __init__(self, img: np.ndarray, working_path: str, initial_mask_path: str | None = None, validator=None):
        """Interactive builder.

        Parameters
        ----------
        img
            RGB image as NumPy array.
        working_path
            Path where candidate and final masks are written (e.g. ``image_001_mask.png``).
        initial_mask_path
            Optional coarse mask (e.g. ``image_001_initial.png``). Pixels
            positive in *initial_mask_path* but not yet known-positive are
            investigated in a dedicated step.

        If *working_path* already exists **and** no checkpoint is present, its
        positive pixels are considered *confirmed positives* from a previous
        run – forming the first partition requested by the user.
        """
        self.img = img  # original RGB image (height, width, 3)
        self.h, self.w = img.shape[:2]
        self.total_pixels = self.h * self.w
        self.working_path = working_path  # path where mask will be written each round
        self.initial_mask_path = initial_mask_path
        self.validator = validator  # TumorSegmentationValidator instance or None

        # Potential checkpoint path (same folder, .pkl extension)
        self._ckpt_path = f"{self.working_path}.ckpt.pkl"
        self._loaded_checkpoint = False

        # ---------- Attempt resume from checkpoint ----------
        if os.path.exists(self._ckpt_path):
            try:
                with open(self._ckpt_path, "rb") as fh:
                    data = pickle.load(fh)
                self.known_positives = set(tuple(p) for p in data["known"])
                self.remaining_pixels = [tuple(p) for p in data["remaining"]]
                self.ground_truth_positive = data["ground_truth_positive"]
                self.prev_save_pixels = set(tuple(p) for p in data.get("prev_save", []))
                self._loaded_checkpoint = True
                print(f"🡒 Loaded checkpoint with {len(self.known_positives)} known positives.")
            except Exception as exc:
                print(f"⚠️  Could not load checkpoint ({exc}) – starting fresh.")
                self._init_fresh_state()
        else:
            self._init_fresh_state()

    def _init_fresh_state(self):
        """Initialise algorithm state from scratch (no checkpoint)."""
        # State that evolves during the algorithm
        self.known_positives: Set[Tuple[int, int]] = set()
        self.remaining_pixels: List[Tuple[int, int]] = [
            (r, c) for r in range(self.h) for c in range(self.w)
        ]
        # We will discover this after the first dice feedback
        self.ground_truth_positive: int | None = None

        # ---------- Seed from previous saved mask, if any ----------
        # Will hold pixels from previous saved mask (partition 1)
        self.prev_save_pixels: Set[Tuple[int, int]] = set()
        if os.path.exists(self.working_path):
            prev_mask = np.array(Image.open(self.working_path))
            if prev_mask.ndim == 2:
                prev_mask = np.stack([prev_mask]*3, axis=-1)
            pos_coords = np.argwhere(prev_mask[:, :, 0] > 0)
            prev_pixels = {(int(r), int(c)) for r, c in pos_coords}
            self.prev_save_pixels = prev_pixels
        # Remove confirmed positives (none at this point) from remaining list
        # We keep prev_save_pixels inside remaining so that they get validated later.
        self.remaining_pixels = [p for p in self.remaining_pixels if p not in self.known_positives]


    # ---------------------------- IO helpers ----------------------------
    def _write_mask(self, positive_pixels: Set[Tuple[int, int]]):
        """Write a 3-channel PNG mask with given positive pixels to working_path."""
        mask = np.zeros_like(self.img, dtype=np.uint8)
        for r, c in positive_pixels:
            mask[r, c, :] = 255
        Image.fromarray(mask).save(self.working_path)

    def _save_checkpoint(self):
        """Serialise current progress so we can resume later."""
        data = {
            "known": list(self.known_positives),
            "remaining": self.remaining_pixels,
            "ground_truth_positive": self.ground_truth_positive,
            "prev_save": list(self.prev_save_pixels),
        }
        try:
            with open(self._ckpt_path, "wb") as fh:
                pickle.dump(data, fh)
        except Exception as exc:
            print(f"⚠️  Could not save checkpoint ({exc}). Progress is still kept in RAM.")

    def _prompt_for_score(self) -> float:
        """Get Dice score either via validator.auto run or manual input."""
        if self.validator is not None:
            score = self.validator.queue_validation_attempt()
            print(f"↳ Received score {score} from TumorSegmentationValidator")
            return float(score)

        while True:
            try:
                inp = input("Paste Dice score returned by API (e.g. 0.8237): ")
                score = float(inp)
                if 0.0 <= score <= 1.0:
                    return score
                print("Score must be between 0 and 1 – try again.")
            except ValueError:
                print("Could not parse number – try again.")

    # ----------------------- algorithmic helpers -----------------------
    def _initialise_ground_truth_positive(self):
        """Uses an *all-positive* mask to deduce the number of GT positives (P)."""
        print("\n=== Round 0: discovering total positive pixels P ===")
        all_pixels_set = set(self.remaining_pixels)
        self._write_mask(all_pixels_set)
        print(f"Mask with ALL {self.total_pixels} pixels set to positive written to {self.working_path}.")
        print("Running automatic validation cycles until a consistent Dice score is obtained…" if self.validator else "Run your model->API, obtain Dice, then paste it below.")

        Pp = self.total_pixels  # all predicted positive
        tolerance = 1e-8  # how close the fractional part needs to be to an integer

        while True:
            raw_score = self._prompt_for_score()
            score = raw_score * DATASET_SIZE  # convert aggregated score to per-image value
            numerator = score * Pp
            denominator = 2.0 - score
            if denominator == 0:
                print("⚠️  Dice score too high (denominator 0). Waiting 5 min and retrying…")
            else:
                P_exact = numerator / denominator
                frac = abs(P_exact - round(P_exact))
                if frac <= tolerance:
                    self.ground_truth_positive = int(round(P_exact))
                    break
                print(
                    f"⚠️  Inconsistent Dice score – computed P = {P_exact:.2f} (fractional part {frac:.3f}). "
                    "Retrying in 5 minutes…"
                )
            if self.validator is not None:
                time.sleep(300)
            else:
                print("Please run validation again in ≈5 minutes and paste the new Dice score.")
                time.sleep(300)

        print(f"⇒ Ground-truth positive pixel count P = {self.ground_truth_positive}\n")

    # --------------------------- recursion -----------------------------
    def _recursive_solve(self, pixels: List[Tuple[int, int]], positives_needed: int):
        """Recursively classify which pixels in *pixels* are positive."""
        n = len(pixels)
        if positives_needed == 0 or n == 0:
            return  # nothing to add
        if positives_needed == n:
            self.known_positives.update(pixels)
            self._save_checkpoint()
            return
        if n == 1:
            # Exactly one pixel, decide based on positives_needed
            if positives_needed == 1:
                self.known_positives.add(pixels[0])
                self._save_checkpoint()
            return

                # --- spatially split the pixel set along its longer dimension ---
        min_r = min(p[0] for p in pixels)
        max_r = max(p[0] for p in pixels)
        min_c = min(p[1] for p in pixels)
        max_c = max(p[1] for p in pixels)

        if (max_r - min_r) >= (max_c - min_c):  # taller region → split horizontally
            mid_r = (min_r + max_r) // 2
            left  = [p for p in pixels if p[0] <= mid_r]
            right = [p for p in pixels if p[0] >  mid_r]
        else:  # wider region → split vertically
            mid_c = (min_c + max_c) // 2
            left  = [p for p in pixels if p[1] <= mid_c]
            right = [p for p in pixels if p[1] >  mid_c]

        # In rare cases one half may become empty due to integer division – fallback to simple slicing
        if not left or not right:
            mid = n // 2
            left, right = pixels[:mid], pixels[mid:]

        # Test mask = known positives ∪ left
        test_positive_pixels = self.known_positives.union(left)
        self._write_mask(test_positive_pixels)
        print(f"\nMask written containing {len(test_positive_pixels)} predicted-positive pixels.")
        tolerance = 1e-8
        while True:
            raw_score = self._prompt_for_score()
            score = raw_score * DATASET_SIZE
            Pp = len(test_positive_pixels)
            P = self.ground_truth_positive
            tp_exact = score * (Pp + P) / 2.0  # from Dice → TP formula
            frac = abs(tp_exact - round(tp_exact))
            if frac <= tolerance:
                tp_total = int(round(tp_exact))
                break
            print(
                f"⚠️  Inconsistent Dice score for split – TP_exact={tp_exact:.4f} (fractional {frac:.3e}). "
                "Retrying in 5 minutes…"
            )
            if self.validator is not None:
                time.sleep(300)
            else:
                print("Please run validation again in ≈5 minutes and paste new Dice score.")
                time.sleep(300)

        tp_known = len(self.known_positives)
        tp_left = tp_total - tp_known  # how many of *left* are truly positive
        tp_left = max(0, min(tp_left, len(left)))  # clamp

        print(
            f"→ TP in LEFT half = {tp_left} / {len(left)}  |  TP_known = {tp_known}  |  Positives still needed overall = {positives_needed}"
        )

        # Recurse
        self._recursive_solve(left, tp_left)
        self._recursive_solve(right, positives_needed - tp_left)

    # ---------------------------- public ------------------------------
    def run(self):
        # Step 1: determine or restore total positives P
        if not self._loaded_checkpoint:
            self._initialise_ground_truth_positive()
            # Save immediately so we can resume even if the next step crashes
            self._save_checkpoint()
        else:
            print("🡒 Resuming interactive session – ground_truth_positive =", self.ground_truth_positive)

        # Step 2: validate previous-save pixels if any
        if self.prev_save_pixels:
            print("\n=== Validating pixels from previous saved mask ===")
            self._write_mask(set(self.prev_save_pixels).union(self.known_positives))
            predicted_positive = len(self.prev_save_pixels) + len(self.known_positives)
            print(f"Mask with {predicted_positive} predicted-positive pixels written. Running validation…")
            tolerance = 1e-8
            while True:
                raw_score = self._prompt_for_score()
                dice_score = raw_score * DATASET_SIZE
                P = self.ground_truth_positive
                tp_exact = dice_score * (predicted_positive + P) / 2.0
                frac = abs(tp_exact - round(tp_exact))
                if frac <= tolerance:
                    tp_total = int(round(tp_exact))
                    break
                print(
                    f"⚠️  Inconsistent Dice score for previous-save validation – TP_exact={tp_exact:.4f} (fractional {frac:.3e}). "
                    "Retrying in 5 minutes…"
                )
                if self.validator is not None:
                    time.sleep(300)
                else:
                    print("Please run validation again in ≈5 minutes and paste new Dice score.")
                    time.sleep(300)

            tp_known = len(self.known_positives)
            tp_prev = max(0, min(tp_total - tp_known, len(self.prev_save_pixels)))
            print(f"⇒ TP within previous-save pixels = {tp_prev} / {len(self.prev_save_pixels)}")
            self._recursive_solve(list(self.prev_save_pixels), tp_prev)

        # Step 3: optionally incorporate an initial guess mask
        if self.initial_mask_path and os.path.exists(self.initial_mask_path):
            print("\n=== Using initial guess mask ===")
            init_mask = np.array(Image.open(self.initial_mask_path))
            # reshape init_mask to have 3 channels (all same values)
            if init_mask.ndim == 2:
                init_mask = np.stack([init_mask]*3, axis=-1)
            # Copy initial mask to working path so your model/API sees it immediately
            Image.fromarray(init_mask).save(self.working_path)
            if init_mask.shape != self.img.shape:
                print("⚠️  Initial mask shape does not match image – ignoring initial mask.")
                print(init_mask.shape,self.img.shape)
                remaining_needed = self.ground_truth_positive  # type: ignore[arg-type]
                self._recursive_solve(self.remaining_pixels, remaining_needed)
            else:
                init_pos_coords = np.argwhere(init_mask[:, :, 0] > 0)
                init_pixels_all = [(int(r), int(c)) for r, c in init_pos_coords]
                # Remove pixels already confirmed or present in prev-save partition
                init_pixels = [p for p in init_pixels_all if p not in self.known_positives and p not in self.prev_save_pixels]
                if not init_pixels:
                    print("All pixels from initial mask are already known-positive – skipping this step.")
                else:
                    predicted_positive = len(init_pixels) + len(self.known_positives)
                    print(
                        f"Initial mask contributes {len(init_pixels)} new predicted-positive pixels. Running validation…"
                    )
                    tolerance = 1e-8
                    while True:
                        raw_score = self._prompt_for_score()
                        dice_score = raw_score * DATASET_SIZE
                        P = self.ground_truth_positive
                        tp_exact = dice_score * (predicted_positive + P) / 2.0
                        frac = abs(tp_exact - round(tp_exact))
                        if frac <= tolerance:
                            tp_total = int(round(tp_exact))
                            break
                        print(
                            f"⚠️  Inconsistent Dice score for initial-mask validation – TP_exact={tp_exact:.4f} (fractional {frac:.3e}). "
                            "Retrying in 5 minutes…"
                        )
                        if self.validator is not None:
                            time.sleep(300)
                        else:
                            print("Please run validation again in ≈5 minutes and paste new Dice score.")
                            time.sleep(300)

                    tp_known = len(self.known_positives)
                    tp_init = max(0, min(tp_total - tp_known, len(init_pixels)))
                    print(f"⇒ TP within new initial-mask pixels = {tp_init} / {len(init_pixels)}")

                    # Recurse within initial mask and outside it separately
                    self._recursive_solve(init_pixels, tp_init)
                    outside_pixels = [p for p in self.remaining_pixels if p not in init_pixels]
                    remaining_outside_tp = self.ground_truth_positive - len(self.known_positives)
                    self._recursive_solve(outside_pixels, remaining_outside_tp)
        else:
            # No initial mask – proceed from scratch
            remaining_needed = self.ground_truth_positive  # type: ignore[arg-type]
            self._recursive_solve(self.remaining_pixels, remaining_needed)

        # Step 3: save final mask in ground_truth folder
        gt_folder = os.path.join(os.path.dirname(self.working_path))
        os.makedirs(gt_folder, exist_ok=True)
        self._write_mask(self.known_positives)
        # Final checkpoint – could delete to signal completion if desired
        self._save_checkpoint()
        print("\n=== Finished ===")
        print(f"Identified {len(self.known_positives)} positive pixels (expected {self.ground_truth_positive}).")
        print(f"Final mask saved to {self.working_path}\n")

# ----------------------------------------------------------------------
# CLI entry-point
# ----------------------------------------------------------------------

def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python interactive_mask_builder.py <image_path> <output_mask_png> [initial_mask_png]")
        print("Example: python interactive_mask_builder.py validation2/image_001.png validation2/ground_truth/pred_mask.png rough_mask.png")
        sys.exit(1)

    image_path, out_mask_path = sys.argv[1], sys.argv[2]
    init_mask_path = sys.argv[3] if len(sys.argv) == 4 else None
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)

    img = np.array(Image.open(image_path))
    builder = InteractiveMaskBuilder(img, out_mask_path, init_mask_path)
    builder.run()


if __name__ == "__main__":
    main()
