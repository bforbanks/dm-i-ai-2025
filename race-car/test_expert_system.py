import pygame
import importlib
import json
import time
from typing import Dict, List
from src.game.core import initialize_game_state, game_loop
import src.game.core as game_core


class ExpertSystemTester:
    """Test and analyze the OptimalExpertSystem performance"""

    def __init__(self, model_name: str = "OptimalExpertSystem"):
        self.model_name = model_name
        self.test_results = []
        self.detailed_logs = []

        # Import the model
        module = importlib.import_module(f"models.{model_name}")
        ModelClass = getattr(module, model_name)
        self.model = ModelClass()

    def run_test_suite(self, num_tests: int = 10, verbose: bool = True):
        """Run multiple tests and collect performance data"""
        print(f"🧪 Testing {self.model_name} with {num_tests} runs...")

        for i in range(num_tests):
            print(f"\n--- Test {i + 1}/{num_tests} ---")
            result = self.run_single_test(test_id=i, verbose=verbose)
            self.test_results.append(result)

            # Print quick summary
            print(
                f"Result: Distance={result['distance']:.1f}, Ticks={result['ticks']}, Crashed={result['crashed']}"
            )

        self.analyze_results()

    def run_single_test(
        self, test_id: int, verbose: bool = False, seed_value=None
    ) -> Dict:
        """Run a single test and collect detailed data"""

        # Initialize game
        initialize_game_state("http://test.com", seed_value)

        # Track performance
        start_time = time.time()
        action_log = []
        state_log = []

        # Custom game loop with logging
        clock = pygame.time.Clock()
        screen = None
        if verbose:
            screen = pygame.display.set_mode((1600, 1200))
            pygame.display.set_caption(f"Testing {self.model_name}")

        tick_count = 0

        while True:
            if verbose:
                delta = clock.tick(60)
            else:
                delta = clock.tick(100000)  # Fast mode

            game_core.STATE.elapsed_game_time += delta
            game_core.STATE.ticks += 1
            tick_count += 1

            # Check end conditions
            if (
                game_core.STATE.crashed
                or game_core.STATE.ticks > 3600
                or game_core.STATE.elapsed_game_time > 60000
            ):
                break

            # Get action from model
            state_dict = self.state_to_dict()
            action_list = self.model.return_action(state_dict)
            action = action_list[0] if action_list else "NOTHING"

            # Log the action and state
            buffer_size = 0
            if hasattr(self.model, "action_buffer"):
                buffer_size = len(self.model.action_buffer)

            action_log.append(
                {
                    "tick": game_core.STATE.ticks,
                    "action": action,
                    "velocity_x": game_core.STATE.ego.velocity.x,
                    "distance": game_core.STATE.distance,
                    "front_sensor": state_dict["sensors"].get("front"),
                    "model_buffer_size": buffer_size,
                }
            )

            # Execute action and update game
            self.handle_action(action)
            game_core.STATE.distance += game_core.STATE.ego.velocity.x
            self.update_game_state()

            # Visual rendering
            if verbose and screen:
                self.render_debug_info(screen, state_dict, action)

        end_time = time.time()

        if screen:
            pygame.display.quit()

        # Compile results
        result = {
            "test_id": test_id,
            "distance": game_core.STATE.distance,
            "ticks": game_core.STATE.ticks,
            "crashed": game_core.STATE.crashed,
            "final_velocity": game_core.STATE.ego.velocity.x,
            "test_time": end_time - start_time,
            "avg_velocity": game_core.STATE.distance / max(game_core.STATE.ticks, 1),
            "actions": action_log[-100:],  # Last 100 actions
        }

        return result

    def state_to_dict(self):
        """Convert game state to dictionary format"""
        sensors_dict = {}
        for sensor in game_core.STATE.sensors:
            sensors_dict[sensor.name] = sensor.reading

        return {
            "did_crash": game_core.STATE.crashed,
            "elapsed_ticks": game_core.STATE.ticks,
            "distance": game_core.STATE.distance,
            "velocity": {
                "x": game_core.STATE.ego.velocity.x,
                "y": game_core.STATE.ego.velocity.y,
            },
            "sensors": sensors_dict,
        }

    def handle_action(self, action: str):
        """Execute a single action"""
        if action == "ACCELERATE":
            game_core.STATE.ego.speed_up()
        elif action == "DECELERATE":
            game_core.STATE.ego.slow_down()
        elif action == "STEER_LEFT":
            game_core.STATE.ego.turn(-0.1)
        elif action == "STEER_RIGHT":
            game_core.STATE.ego.turn(0.1)

    def update_game_state(self):
        """Update game state (cars, sensors, collisions)"""
        from src.game.core import update_cars, remove_passed_cars, place_car, intersects

        update_cars()
        remove_passed_cars()
        place_car()

        # Update sensors
        for sensor in game_core.STATE.sensors:
            sensor.update()

        # Check collisions
        for car in game_core.STATE.cars:
            if car != game_core.STATE.ego and intersects(
                game_core.STATE.ego.rect, car.rect
            ):
                game_core.STATE.crashed = True

        # Check wall collisions
        for wall in game_core.STATE.road.walls:
            if intersects(game_core.STATE.ego.rect, wall.rect):
                game_core.STATE.crashed = True

    def render_debug_info(self, screen, state_dict, current_action):
        """Render debug information on screen"""
        screen.fill((0, 0, 0))

        # Draw game elements
        screen.blit(game_core.STATE.road.surface, (0, 0))

        for wall in game_core.STATE.road.walls:
            wall.draw(screen)

        for car in game_core.STATE.cars:
            if car == game_core.STATE.ego:
                if car.sprite:
                    screen.blit(car.sprite, (car.x, car.y))
                    bounds = car.get_bounds()
                    pygame.draw.rect(screen, (255, 0, 0), bounds, width=2)

        # Draw sensors
        for sensor in game_core.STATE.sensors:
            sensor.draw(screen)

        # Debug text
        font = pygame.font.SysFont("monospace", 20)
        buffer_size = 0
        if hasattr(self.model, "action_buffer"):
            buffer_size = len(self.model.action_buffer)

        debug_lines = [
            f"Action: {current_action}",
            f"Velocity: {game_core.STATE.ego.velocity.x:.1f}",
            f"Distance: {game_core.STATE.distance:.1f}",
            f"Ticks: {game_core.STATE.ticks}",
            f"Front Sensor: {state_dict['sensors'].get('front', 'None')}",
            f"Buffer Size: {buffer_size}",
        ]

        for i, line in enumerate(debug_lines):
            text = font.render(line, True, (255, 255, 255))
            screen.blit(text, (10, 10 + i * 25))

        pygame.display.flip()

    def analyze_results(self):
        """Analyze test results and provide insights"""
        if not self.test_results:
            print("No test results to analyze!")
            return

        # Calculate statistics
        distances = [r["distance"] for r in self.test_results]
        ticks = [r["ticks"] for r in self.test_results]
        crashes = [r["crashed"] for r in self.test_results]
        velocities = [r["avg_velocity"] for r in self.test_results]

        crash_rate = sum(crashes) / len(crashes) * 100
        avg_distance = sum(distances) / len(distances)
        max_distance = max(distances)
        avg_ticks = sum(ticks) / len(ticks)
        avg_velocity = sum(velocities) / len(velocities)

        print(f"\n{'=' * 50}")
        print(f"📊 PERFORMANCE ANALYSIS - {self.model_name}")
        print(f"{'=' * 50}")
        print(f"Tests Run: {len(self.test_results)}")
        print(f"Crash Rate: {crash_rate:.1f}%")
        print(f"Average Distance: {avg_distance:.1f}")
        print(f"Max Distance: {max_distance:.1f}")
        print(f"Average Survival: {avg_ticks:.1f} ticks")
        print(f"Average Velocity: {avg_velocity:.2f}")

        # Identify issues
        print(f"\n🔍 ISSUE ANALYSIS:")

        if crash_rate > 50:
            print("❌ HIGH CRASH RATE - Poor collision avoidance")

        if avg_velocity < 10:
            print("❌ LOW VELOCITY - Not aggressive enough")

        if avg_distance < 1000:
            print("❌ LOW DISTANCE - Strategy needs improvement")

        # Action analysis
        all_actions = []
        for result in self.test_results:
            all_actions.extend([a["action"] for a in result["actions"]])

        action_counts = {}
        for action in all_actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        print(f"\n📈 ACTION DISTRIBUTION:")
        for action, count in sorted(
            action_counts.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = count / len(all_actions) * 100
            print(f"{action}: {percentage:.1f}%")

        # Save detailed results
        with open(f"test_results_{self.model_name}.json", "w") as f:
            json.dump(self.test_results, f, indent=2)

        print(f"\n💾 Detailed results saved to test_results_{self.model_name}.json")


def main():
    """Run the testing suite"""
    pygame.init()

    print("Choose model to test:")
    print("1. OptimalExpertSystem (original complex)")
    print("2. ImprovedExpertSystem (simplified)")
    print("3. SimpleExpertSystem (ultra-simple)")
    print("4. BaselineV (existing baseline)")
    print("5. BaselineL (existing baseline)")

    model_choice = input("Enter model choice (1-5): ").strip()

    model_map = {
        "1": "OptimalExpertSystem",
        "2": "ImprovedExpertSystem",
        "3": "SimpleExpertSystem",
        "4": "BaselineV",
        "5": "BaselineL",
    }

    model_name = model_map.get(model_choice, "ImprovedExpertSystem")

    # Handle SimpleExpertSystem which is in ImprovedExpertSystem file
    if model_name == "SimpleExpertSystem":
        tester = ExpertSystemTester("ImprovedExpertSystem")
        # Manually set the model class
        from models.ImprovedExpertSystem import SimpleExpertSystem

        tester.model = SimpleExpertSystem()
        tester.model_name = "SimpleExpertSystem"
    else:
        tester = ExpertSystemTester(model_name)

    print(f"\nTesting {tester.model_name}...")
    print("Choose test mode:")
    print("1. Quick test (5 runs, no visual)")
    print("2. Detailed test (10 runs, no visual)")
    print("3. Visual test (3 runs, with pygame window)")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        tester.run_test_suite(num_tests=5, verbose=False)
    elif choice == "2":
        tester.run_test_suite(num_tests=10, verbose=False)
    elif choice == "3":
        tester.run_test_suite(num_tests=3, verbose=True)
    else:
        print("Invalid choice, running quick test...")
        tester.run_test_suite(num_tests=5, verbose=False)

    pygame.quit()


if __name__ == "__main__":
    main()
