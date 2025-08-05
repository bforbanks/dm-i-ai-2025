#!/usr/bin/env python3
"""
Quick demo showing enhanced progress reporting during NEAT evolution
"""

from neat_gpu_interface import GPUNEATTrainer, create_neat_config


def run_demo():
    """Run a quick NEAT demo with enhanced progress reporting"""

    print("🚀 GPU NEAT Progress Demo")
    print("This demo shows detailed progress information during evolution")
    print("=" * 60)

    # Configuration
    config_path = "demo_neat_config.txt"
    population_size = 64  # Small for quick demo
    max_generations = 5  # Just a few generations to see progress

    # Create config
    create_neat_config(config_path, population_size)
    print(
        f"📝 Config created for {population_size} genomes, {max_generations} generations"
    )

    # Create trainer
    trainer = GPUNEATTrainer(
        config_path=config_path,
        population_size=population_size,
        max_generations=max_generations,
    )

    print("\n🎬 Starting evolution with enhanced progress reporting...")
    print("You should see:")
    print("  • Neural network building progress")
    print("  • Step-by-step simulation updates")
    print("  • Detailed generation statistics")
    print("  • ETA predictions")
    print("  • Fitness improvements")

    # Run evolution
    winner, stats = trainer.run_evolution()

    # Quick test of the winner
    trainer.test_best_genome(winner, num_tests=2)

    print("\n🎉 Demo complete!")
    print("💡 This same progress reporting works for longer evolutions.")
    print("💡 Increase population_size and max_generations for serious training.")


if __name__ == "__main__":
    run_demo()
