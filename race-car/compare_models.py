import pygame
import importlib
from test_expert_system import ExpertSystemTester
from models.ImprovedExpertSystem import SimpleExpertSystem


def compare_all_models():
    """Compare all available models quickly"""
    pygame.init()

    models_to_test = [
        ("BaselineV", "BaselineV"),
        ("BaselineL", "BaselineL"),
        ("SimpleExpertSystem", None),  # Special case
        ("ImprovedExpertSystem", "ImprovedExpertSystem"),
        ("OptimalExpertSystem", "OptimalExpertSystem"),
    ]

    results = {}

    print("🏁 RACING MODEL COMPARISON")
    print("=" * 50)

    for model_name, module_name in models_to_test:
        print(f"\n🧪 Testing {model_name}...")

        try:
            if model_name == "SimpleExpertSystem":
                # Special handling for SimpleExpertSystem
                tester = ExpertSystemTester("ImprovedExpertSystem")
                tester.model = SimpleExpertSystem()
                tester.model_name = "SimpleExpertSystem"
            else:
                tester = ExpertSystemTester(module_name)

            # Run quick test (5 runs, no visual)
            tester.run_test_suite(num_tests=5, verbose=False)

            # Extract key metrics
            distances = [r["distance"] for r in tester.test_results]
            crashes = [r["crashed"] for r in tester.test_results]
            velocities = [r["avg_velocity"] for r in tester.test_results]

            results[model_name] = {
                "avg_distance": sum(distances) / len(distances),
                "max_distance": max(distances),
                "crash_rate": sum(crashes) / len(crashes) * 100,
                "avg_velocity": sum(velocities) / len(velocities),
                "raw_results": tester.test_results,
            }

        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            results[model_name] = None

    # Display comparison
    print(f"\n🏆 FINAL COMPARISON")
    print("=" * 80)
    print(
        f"{'Model':<20} {'Avg Dist':<10} {'Max Dist':<10} {'Crash %':<10} {'Avg Vel':<10}"
    )
    print("-" * 80)

    # Sort by average distance
    sorted_results = sorted(
        [(name, data) for name, data in results.items() if data is not None],
        key=lambda x: x[1]["avg_distance"],
        reverse=True,
    )

    for model_name, data in sorted_results:
        print(
            f"{model_name:<20} {data['avg_distance']:<10.1f} {data['max_distance']:<10.1f} "
            f"{data['crash_rate']:<10.1f} {data['avg_velocity']:<10.2f}"
        )

    # Identify winner
    if sorted_results:
        winner_name, winner_data = sorted_results[0]
        print(f"\n🥇 WINNER: {winner_name}")
        print(f"   Average Distance: {winner_data['avg_distance']:.1f}")
        print(f"   Crash Rate: {winner_data['crash_rate']:.1f}%")
        print(f"   Average Velocity: {winner_data['avg_velocity']:.2f}")

    pygame.quit()


if __name__ == "__main__":
    compare_all_models()
