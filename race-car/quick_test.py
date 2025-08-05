import pygame
import importlib
from src.game.core import initialize_game_state, game_loop


def quick_test_model(model_name: str, num_tests: int = 3):
    """Quick test of a model without complex analysis"""
    pygame.init()

    print(f"🧪 Quick testing {model_name}...")

    try:
        # Import the model
        if model_name == "SimpleExpertSystem":
            from models.ImprovedExpertSystem import SimpleExpertSystem

            model = SimpleExpertSystem()
        else:
            module = importlib.import_module(f"models.{model_name}")
            ModelClass = getattr(module, model_name)
            model = ModelClass()

        results = []

        for i in range(num_tests):
            print(f"  Test {i + 1}/{num_tests}...", end=" ")

            # Initialize game
            initialize_game_state("http://test.com", None)

            # Use the existing game_loop with our model
            try:
                game_loop(verbose=False, model=model)

                # Get results from the game state
                import src.game.core as game_core

                result = {
                    "distance": game_core.STATE.distance,
                    "ticks": game_core.STATE.ticks,
                    "crashed": game_core.STATE.crashed,
                }
                results.append(result)

                print(
                    f"Distance: {result['distance']:.1f}, Crashed: {result['crashed']}"
                )

            except Exception as e:
                print(f"Error: {e}")
                results.append({"distance": 0, "ticks": 0, "crashed": True})

        # Quick analysis
        if results:
            distances = [r["distance"] for r in results]
            crashes = [r["crashed"] for r in results]

            avg_distance = sum(distances) / len(distances)
            crash_rate = sum(crashes) / len(crashes) * 100
            max_distance = max(distances)

            print(f"\n📊 {model_name} Results:")
            print(f"  Average Distance: {avg_distance:.1f}")
            print(f"  Max Distance: {max_distance:.1f}")
            print(f"  Crash Rate: {crash_rate:.1f}%")

            return avg_distance
        else:
            print(f"❌ No valid results for {model_name}")
            return 0

    except Exception as e:
        print(f"❌ Failed to test {model_name}: {e}")
        return 0

    finally:
        pygame.quit()


def compare_quick():
    """Quick comparison of all models"""
    models = [
        "BaselineV",
        "BaselineL",
        "SimpleExpertSystem",
        "ImprovedExpertSystem",
    ]

    results = {}

    print("🏁 QUICK MODEL COMPARISON")
    print("=" * 40)

    for model in models:
        results[model] = quick_test_model(model, 3)
        print()

    # Show winner
    winner = max(results.items(), key=lambda x: x[1])
    print(f"🏆 Winner: {winner[0]} with {winner[1]:.1f} average distance")


if __name__ == "__main__":
    compare_quick()
