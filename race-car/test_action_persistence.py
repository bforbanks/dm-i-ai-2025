#!/usr/bin/env python3
"""
Test the action persistence system in GPU race environment
"""

import torch
from gpu_race_env import GPURaceEnvironment
import time


def test_action_persistence():
    """Test that actions persist for the correct number of ticks"""
    print("🧪 Testing Action Persistence System")
    print("=" * 50)

    # Create environment with action persistence
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4  # Small for easy tracking
    action_repeat = 3  # Actions persist for 3 ticks

    env = GPURaceEnvironment(
        batch_size=batch_size,
        device=device,
        action_repeat=action_repeat,
        max_ticks=50,  # Short test
    )

    print(f"✅ Environment created:")
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}")
    print(f"   Action repeat: {action_repeat}")

    # Reset environment
    state = env.reset()
    print(f"✅ Reset complete, state shape: {state.shape}")

    # Test action persistence
    step_count = 0
    action_call_count = 0

    print(f"\n🎮 Testing action persistence...")
    print(f"Expected: Actions called every {action_repeat} ticks")

    while not env.done.all() and step_count < 30:
        # Step environment
        if step_count == 0:
            # First step needs actions
            actions = torch.randint(0, 5, (batch_size,), device=device)
            action_call_count += 1
            print(f"Tick {step_count:2d}: Providing actions {actions.tolist()}")
        else:
            actions = None

        next_state, rewards, done, info = env.step(actions)

        # Check if actions are needed next tick
        needs_action = info.get("needs_action", torch.zeros_like(done))
        remaining_ticks = info.get("action_ticks_remaining", torch.zeros_like(done))

        if needs_action.any():
            action_call_count += 1
            print(f"Tick {step_count:2d}: Actions will be needed next tick")
            # Provide actions for next step
            actions = torch.randint(0, 5, (batch_size,), device=device)
        else:
            print(
                f"Tick {step_count:2d}: Continuing previous actions, {remaining_ticks[0].item()} ticks remaining"
            )

        step_count += 1

    # Calculate efficiency
    expected_calls = (step_count + action_repeat - 1) // action_repeat
    actual_efficiency = (1 - action_call_count / step_count) * 100
    expected_efficiency = (1 - expected_calls / step_count) * 100

    print(f"\n📊 Results:")
    print(f"   Total ticks: {step_count}")
    print(f"   Action calls: {action_call_count}")
    print(f"   Expected calls: {expected_calls}")
    print(
        f"   Actual efficiency: {actual_efficiency:.1f}% reduction in neural network calls"
    )
    print(f"   Expected efficiency: {expected_efficiency:.1f}% reduction")

    success = abs(action_call_count - expected_calls) <= 1  # Allow 1 call difference

    if success:
        print("✅ Action persistence working correctly!")
    else:
        print("❌ Action persistence not working as expected")

    return success


def test_neat_integration():
    """Test that the action persistence works with NEAT training"""
    print(f"\n🧬 Testing NEAT Integration...")

    try:
        from neat_gpu_interface import GPUNEATTrainer, create_neat_config

        # Create tiny config for test
        config_path = "test_persistence_config.txt"
        create_neat_config(config_path, population_size=8)

        # Create trainer with action persistence
        trainer = GPUNEATTrainer(
            config_path=config_path,
            population_size=8,
            max_generations=1,  # Just test one generation
        )

        # Check that the environment has action persistence
        action_repeat = trainer.env.action_repeat
        print(f"✅ NEAT trainer created with action_repeat={action_repeat}")

        # Run a quick test
        print("🎯 Running one generation test...")

        # This should work without errors and show decision efficiency
        winner, stats = trainer.run_evolution()

        print("✅ NEAT integration test successful!")
        return True

    except Exception as e:
        print(f"❌ NEAT integration test failed: {e}")
        return False


def benchmark_performance():
    """Compare performance with and without action persistence"""
    print(f"\n⚡ Performance Benchmark...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    test_steps = 100

    results = {}

    for action_repeat in [1, 3, 5]:
        print(f"\n  Testing action_repeat={action_repeat}...")

        env = GPURaceEnvironment(
            batch_size=batch_size,
            device=device,
            action_repeat=action_repeat,
            max_ticks=test_steps,
        )

        state = env.reset()

        start_time = time.time()
        action_calls = 0

        for step in range(test_steps):
            if step == 0:
                actions = torch.randint(0, 5, (batch_size,), device=device)
                action_calls += 1
            else:
                actions = None

            next_state, rewards, done, info = env.step(actions)

            if info.get("needs_action", torch.zeros_like(done)).any():
                actions = torch.randint(0, 5, (batch_size,), device=device)
                action_calls += 1

        end_time = time.time()

        efficiency = (1 - action_calls / test_steps) * 100

        results[action_repeat] = {
            "time": end_time - start_time,
            "action_calls": action_calls,
            "efficiency": efficiency,
        }

        print(f"    Time: {results[action_repeat]['time']:.3f}s")
        print(f"    Action calls: {action_calls}/{test_steps}")
        print(f"    Efficiency: {efficiency:.1f}% fewer neural network calls")

    print(f"\n📈 Performance Summary:")
    baseline_time = results[1]["time"]
    for action_repeat, data in results.items():
        speedup = baseline_time / data["time"]
        print(
            f"   action_repeat={action_repeat}: {speedup:.1f}x speedup, {data['efficiency']:.1f}% efficiency"
        )

    return results


if __name__ == "__main__":
    print("🚀 Action Persistence Test Suite")
    print("=" * 60)

    success = True

    # Test 1: Basic action persistence
    success &= test_action_persistence()

    # Test 2: NEAT integration
    success &= test_neat_integration()

    # Test 3: Performance benchmark
    benchmark_performance()

    if success:
        print(f"\n🎉 All tests passed! Action persistence is working correctly.")
        print(f"💡 NEAT models now make decisions every 3 ticks instead of every tick!")
        print(f"💡 This should provide ~3x speedup and more impactful actions.")
    else:
        print(f"\n❌ Some tests failed. Check the errors above.")
