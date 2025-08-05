import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from gpu_race_env import GPURaceEnvironment


def test_basic_functionality():
    """Test basic GPU environment functionality"""
    print("🧪 Testing GPU Race Environment...")

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create environment
    batch_size = 128
    env = GPURaceEnvironment(batch_size=batch_size, device=device)

    print(f"✅ Environment created with batch size: {batch_size}")
    print(f"   State space size: {env.get_state_space_size()}")
    print(f"   Action space size: {env.get_action_space_size()}")

    # Test reset
    state = env.reset()
    print(f"✅ Reset successful, state shape: {state.shape}")

    # Test random actions
    print("\n🎮 Running random action test...")
    steps = 0
    start_time = time.time()

    while not env.done.all() and steps < 1000:
        # Random actions
        actions = torch.randint(
            0, env.get_action_space_size(), (batch_size,), device=device
        )

        state, rewards, done, info = env.step(actions)
        steps += 1

        if steps % 100 == 0:
            stats = env.get_batch_stats()
            print(
                f"  Step {steps}: Active envs: {stats['active_envs']}, "
                f"Mean distance: {stats['mean_distance']:.1f}"
            )

    end_time = time.time()

    # Final statistics
    final_stats = env.get_batch_stats()
    print(f"\n📊 Final Results:")
    print(f"   Total steps: {steps}")
    print(f"   Simulation time: {end_time - start_time:.2f}s")
    print(f"   Steps per second: {steps * batch_size / (end_time - start_time):.0f}")
    print(f"   Mean distance: {final_stats['mean_distance']:.1f}")
    print(f"   Max distance: {final_stats['max_distance']:.1f}")
    print(f"   Crash rate: {final_stats['crash_rate'] * 100:.1f}%")

    return final_stats


def test_simple_policies():
    """Test some simple hardcoded policies"""
    print("\n🧠 Testing Simple Policies...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64

    policies = {
        "always_accelerate": lambda state: torch.ones(
            batch_size, device=device, dtype=torch.long
        ),
        "front_sensor_reactive": lambda state: front_sensor_policy(state),
        "velocity_controller": lambda state: velocity_policy(state),
    }

    results = {}

    for policy_name, policy_func in policies.items():
        print(f"\n  Testing {policy_name}...")

        env = GPURaceEnvironment(batch_size=batch_size, device=device)
        state = env.reset()

        steps = 0
        start_time = time.time()

        while not env.done.all() and steps < 2000:
            actions = policy_func(state)
            state, rewards, done, info = env.step(actions)
            steps += 1

        end_time = time.time()
        stats = env.get_batch_stats()

        results[policy_name] = {
            "mean_distance": stats["mean_distance"],
            "max_distance": stats["max_distance"],
            "crash_rate": stats["crash_rate"],
            "time": end_time - start_time,
        }

        print(
            f"    Distance: {stats['mean_distance']:.1f}, "
            f"Crash rate: {stats['crash_rate'] * 100:.1f}%"
        )

    # Show comparison
    print(f"\n🏆 Policy Comparison:")
    for name, stats in results.items():
        print(
            f"  {name:<20}: Dist={stats['mean_distance']:.1f}, "
            f"Crash={stats['crash_rate'] * 100:.1f}%"
        )

    return results


def front_sensor_policy(state):
    """Simple policy based on front sensor"""
    batch_size = state.shape[0]
    device = state.device

    # Extract front sensor (sensor index 4)
    front_sensor = state[:, 6]  # 2 velocity + 4 sensors = index 6 for front

    # Simple logic: brake if obstacle close, accelerate otherwise
    actions = torch.ones(
        batch_size, device=device, dtype=torch.long
    )  # Default: accelerate

    # If front obstacle close, brake
    close_obstacle = front_sensor < 0.3  # Normalized sensor reading
    actions[close_obstacle] = 2  # Decelerate

    return actions


def velocity_policy(state):
    """Policy that tries to maintain optimal velocity"""
    batch_size = state.shape[0]
    device = state.device

    # Extract velocity
    velocity_x = state[:, 0]  # First component is normalized velocity_x

    actions = torch.zeros(
        batch_size, device=device, dtype=torch.long
    )  # Default: nothing

    # Accelerate if too slow
    too_slow = velocity_x < 0.6  # Target ~60% of max velocity
    actions[too_slow] = 1  # Accelerate

    # Decelerate if too fast
    too_fast = velocity_x > 0.8
    actions[too_fast] = 2  # Decelerate

    return actions


def benchmark_performance():
    """Benchmark GPU vs CPU performance"""
    print("\n⚡ Performance Benchmark...")

    batch_sizes = [64, 128, 256, 512]
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    results = {}

    for device in devices:
        print(f"\n  Testing on {device.upper()}:")
        device_results = {}

        for batch_size in batch_sizes:
            try:
                env = GPURaceEnvironment(batch_size=batch_size, device=device)
                state = env.reset()

                # Time 100 steps
                start_time = time.time()
                for _ in range(100):
                    actions = torch.randint(0, 5, (batch_size,), device=device)
                    state, _, _, _ = env.step(actions)

                end_time = time.time()
                steps_per_sec = 100 * batch_size / (end_time - start_time)
                device_results[batch_size] = steps_per_sec

                print(f"    Batch {batch_size}: {steps_per_sec:.0f} steps/sec")

            except Exception as e:
                print(f"    Batch {batch_size}: Failed ({e})")
                device_results[batch_size] = 0

        results[device] = device_results

    return results


if __name__ == "__main__":
    print("🏎️ GPU Race Car Environment Test Suite")
    print("=" * 50)

    try:
        # Basic functionality test
        basic_stats = test_basic_functionality()

        # Simple policies test
        policy_results = test_simple_policies()

        # Performance benchmark
        perf_results = benchmark_performance()

        print("\n✅ All tests completed successfully!")

        # Summary
        print(f"\n📈 Summary:")
        print(f"   GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU name: {torch.cuda.get_device_name()}")
        print(
            f"   Best policy: {max(policy_results.keys(), key=lambda k: policy_results[k]['mean_distance'])}"
        )

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
