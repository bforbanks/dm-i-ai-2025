import pygame
import importlib
import random
import matplotlib.pyplot as plt
from typing import List

# Import game functions
from src.game.core import initialize_game_state, game_loop


def run_lane_shift_tests(num_tests: int = 500, seeds: List[int] | None = None) -> List[float]:
    """Run LaneShift model for a given number of tests and return list of distances."""

    # Dynamic import of LaneShift model
    module = importlib.import_module("models.LaneShift")
    LaneShift = getattr(module, "LaneShift")

    # Prepare seeds
    if seeds is None:
        seeds = [random.randint(0, 10000) for _ in range(num_tests)]
    else:
        num_tests = len(seeds)

    distances = []

    pygame.init()
    try:
        for idx, seed_value in enumerate(seeds):
            print(f"Test {idx + 1}/{num_tests} (seed={seed_value})")

            # Initialize model and game state
            model = LaneShift()
            initialize_game_state(api_url="http://example.com/api/predict", seed_value=seed_value)

            # Run the game loop
            try:
                game_loop(verbose=False, model=model)
            except Exception as e:
                print(f"  Error during game loop: {e}")
                # If an error occurs, treat distance as 0
                import src.game.core as game_core
                distances.append(0)
                continue

            # Retrieve distance from global state
            import src.game.core as game_core
            distances.append(game_core.STATE.distance)
            print(f"  Distance: {game_core.STATE.distance:.1f}")
    finally:
        pygame.quit()

    return distances


def plot_histogram(distances: List[float], bins: int = 10, output_file: str | None = None):
    """Plot a histogram of the distances."""
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=bins, color="skyblue", edgecolor="black")
    plt.title("LaneShift Model Distance Distribution")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)

    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
        print(f"Histogram saved to {output_file}")
    else:
        plt.show()


if __name__ == "__main__":
    NUM_TESTS = 500  # Change this number for more/less tests

    distances = run_lane_shift_tests(num_tests=NUM_TESTS)
    print(f"\nAverage distance over {len(distances)} runs: {sum(distances) / len(distances):.1f}")

    plot_histogram(distances, bins=30, output_file="lane_shift_histogram.png")
