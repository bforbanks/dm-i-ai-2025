import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any
import neat
import pickle
import time
from gpu_race_env import GPURaceEnvironment


class NEATNeuralNetwork(nn.Module):
    """
    PyTorch implementation of NEAT neural network for GPU execution
    """

    def __init__(self, genome, config):
        super().__init__()
        self.genome = genome
        self.config = config

        # Build network from genome
        self.layers = self._build_network()

        # Set weights from genome after layers are created
        self._set_weights_from_genome()

    def _build_network(self):
        """Build PyTorch network from NEAT genome"""

        # Get all nodes and connections
        input_keys = list(self.config.genome_config.input_keys)
        output_keys = list(self.config.genome_config.output_keys)

        # For simplicity, create a feedforward network
        # In a full NEAT implementation, you'd handle the full topology

        # Count hidden nodes
        hidden_nodes = []
        for node_key in self.genome.nodes.keys():
            if node_key not in input_keys and node_key not in output_keys:
                hidden_nodes.append(node_key)

        layers = nn.ModuleList()

        if hidden_nodes:
            # Input to hidden
            input_size = len(input_keys)
            hidden_size = len(hidden_nodes)
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.Tanh())

            # Hidden to output
            output_size = len(output_keys)
            layers.append(nn.Linear(hidden_size, output_size))
        else:
            # Direct input to output
            input_size = len(input_keys)
            output_size = len(output_keys)
            layers.append(nn.Linear(input_size, output_size))

        return layers

    def _set_weights_from_genome(self):
        """Set network weights from NEAT genome"""
        # This is a simplified version - full NEAT would handle arbitrary topologies

        connections = self.genome.connections
        input_keys = list(self.config.genome_config.input_keys)
        output_keys = list(self.config.genome_config.output_keys)

        # For now, create random weights (you'd extract from genome connections)
        with torch.no_grad():
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    # Initialize with small random weights
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Forward pass through network"""
        for layer in self.layers:
            x = layer(x)
        return x


class GPUNEATTrainer:
    """
    GPU-accelerated NEAT trainer for the race car environment
    """

    def __init__(
        self,
        config_path: str,
        population_size: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_generations: int = 1000,
    ):
        # Load NEAT config
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )

        self.population_size = population_size
        self.device = torch.device(device)
        self.max_generations = max_generations

        # Create GPU environment
        self.env = GPURaceEnvironment(batch_size=population_size, device=device)

        # Statistics tracking
        self.generation_stats = []
        self.best_genome = None
        self.best_fitness = float("-inf")

        print(f"🚀 GPU NEAT Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Population Size: {population_size}")
        print(f"   State Space: {self.env.get_state_space_size()}")
        print(f"   Action Space: {self.env.get_action_space_size()}")

    def evaluate_population(self, genomes: List[Tuple[int, Any]]) -> List[float]:
        """
        Evaluate entire population in parallel on GPU

        Args:
            genomes: List of (genome_id, genome) tuples

        Returns:
            List of fitness values
        """

        start_time = time.time()
        print(f"🔄 Evaluating {len(genomes)} genomes...")

        # Create neural networks for all genomes
        networks = []
        print(f"⚙️  Building neural networks...", end=" ")
        for i, (genome_id, genome) in enumerate(genomes):
            if i % 50 == 0 and i > 0:
                print(f"{i}/{len(genomes)}", end=" ")
            net = NEATNeuralNetwork(genome, self.config)
            net.to(self.device)
            net.eval()
            networks.append(net)
        print("✅")

        # Pad to batch size if needed
        while len(networks) < self.population_size:
            networks.append(networks[0])  # Duplicate first network

        # Reset environment
        print(f"🎮 Starting simulation...")
        state = self.env.reset()  # (batch_size, state_dim)

        # Track fitness for each network
        fitness_scores = torch.zeros(len(genomes), device=self.device)
        step_count = 0
        last_report = 0
        decision_count = 0

        # Initial step to get the environment started
        next_state, rewards, done, info = self.env.step()
        state = next_state

        while not self.env.done.all() and step_count < self.env.max_ticks:
            # Check which environments need new actions
            needs_action = info.get("needs_action", torch.ones_like(self.env.done))

            new_actions = None
            if needs_action.any():
                decision_count += 1
                # Get actions from networks that need them
                new_actions = torch.zeros(
                    self.population_size, device=self.device, dtype=torch.long
                )

                with torch.no_grad():
                    for i, network in enumerate(networks):
                        if (
                            i < len(genomes) and needs_action[i]
                        ):  # Only process genomes that need actions
                            output = network(state[i : i + 1])  # Single state
                            action = torch.argmax(output, dim=1)
                            new_actions[i] = action

            # Step environment (physics always runs at 60fps)
            next_state, rewards, done, info = self.env.step(new_actions)

            # Accumulate fitness (only for real genomes)
            fitness_scores[: len(genomes)] += rewards[: len(genomes)]

            state = next_state
            step_count += 1

            # Progress reporting every 500 steps
            if step_count - last_report >= 500:
                active_envs = (~self.env.done).sum().item()
                avg_distance = info["distances"][: len(genomes)].mean().item()
                decision_efficiency = decision_count / step_count * 100
                print(
                    f"   Step {step_count:4d}: {active_envs:3d} active, avg dist: {avg_distance:6.1f}, decisions: {decision_efficiency:.1f}%"
                )
                last_report = step_count

        # Add final distance bonus
        final_distances = info["distances"][: len(genomes)]
        fitness_scores += final_distances * 0.01  # Distance bonus

        # Penalty for crashing
        crashed = info["crashed"][: len(genomes)]
        fitness_scores -= crashed.float() * 50.0

        eval_time = time.time() - start_time

        # Print detailed statistics
        avg_fitness = fitness_scores.mean().item()
        max_fitness = fitness_scores.max().item()
        min_fitness = fitness_scores.min().item()
        avg_distance = final_distances.mean().item()
        max_distance = final_distances.max().item()
        crash_rate = crashed.float().mean().item() * 100
        survival_rate = 100 - crash_rate

        print(f"\n📊 Generation Results:")
        print(f"   ⏱️  Evaluation time: {eval_time:.2f}s")
        print(f"   🏃 Total steps: {step_count}")
        print(
            f"   📈 Fitness - Avg: {avg_fitness:.1f}, Max: {max_fitness:.1f}, Min: {min_fitness:.1f}"
        )
        print(f"   🏁 Distance - Avg: {avg_distance:.1f}, Max: {max_distance:.1f}")
        print(f"   💥 Crash rate: {crash_rate:.1f}%")
        print(f"   ✅ Survival rate: {survival_rate:.1f}%")
        print(f"   ⚡ Speed: {step_count * len(genomes) / eval_time:.0f} steps/sec")

        return fitness_scores.cpu().numpy().tolist()

    def run_evolution(self) -> neat.Population:
        """Run NEAT evolution"""

        print("🧬 Starting NEAT Evolution...")
        print(f"🎯 Target: {self.max_generations} generations")
        print(f"🏆 Fitness threshold: {self.config.fitness_threshold}")
        print(f"👥 Population size: {self.population_size}")
        print("=" * 60)

        # Create population
        population = neat.Population(self.config)

        # Add reporters
        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)

        generation_count = 0
        start_time = time.time()

        # Evolution loop
        def eval_genomes(genomes, config):
            """Evaluation function for NEAT"""
            nonlocal generation_count
            generation_count += 1

            print(f"\n🔄 GENERATION {generation_count}")
            print(f"⏱️  Elapsed time: {time.time() - start_time:.1f}s")

            fitness_values = self.evaluate_population(genomes)

            # Assign fitness to genomes
            for (genome_id, genome), fitness in zip(genomes, fitness_values):
                genome.fitness = fitness

                # Track best genome
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_genome = genome
                    print(
                        f"🌟 NEW BEST FITNESS: {self.best_fitness:.1f} (Genome {genome_id})"
                    )

            # Generation summary
            fitness_values = [f for f in fitness_values if f is not None]
            if fitness_values:
                avg_fitness = sum(fitness_values) / len(fitness_values)
                max_fitness = max(fitness_values)
                min_fitness = min(fitness_values)

                print(f"📊 Generation {generation_count} Summary:")
                print(
                    f"   Best: {max_fitness:.1f} | Avg: {avg_fitness:.1f} | Worst: {min_fitness:.1f}"
                )
                print(
                    f"   Progress: {generation_count}/{self.max_generations} ({generation_count / self.max_generations * 100:.1f}%)"
                )

                # Estimate time remaining
                elapsed = time.time() - start_time
                time_per_gen = elapsed / generation_count
                remaining_gens = self.max_generations - generation_count
                eta = remaining_gens * time_per_gen

                if eta > 60:
                    print(f"   ETA: {eta / 60:.1f} minutes")
                else:
                    print(f"   ETA: {eta:.0f} seconds")

            print("-" * 60)

        # Run evolution
        winner = population.run(eval_genomes, self.max_generations)

        total_time = time.time() - start_time
        print(f"\n🏆 Evolution Complete!")
        print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
        print(f"🎯 Generations completed: {generation_count}")
        print(f"🌟 Best fitness achieved: {self.best_fitness:.2f}")
        print(f"🏁 Winner genome ID: {winner.key}")

        return winner, stats

    def test_best_genome(self, genome, num_tests: int = 10, visualize: bool = False):
        """Test the best genome"""

        print(f"🧪 Testing best genome ({num_tests} runs)...")

        # Create network
        network = NEATNeuralNetwork(genome, self.config)
        network.to(self.device)
        network.eval()

        total_distance = 0
        total_crashes = 0

        for test in range(num_tests):
            # Single environment test
            env = GPURaceEnvironment(batch_size=1, device=self.device)
            state = env.reset()

            step_count = 0
            # Initial step
            next_state, reward, done, info = env.step()
            state = next_state

            while not env.done[0] and step_count < env.max_ticks:
                # Check if action is needed
                needs_action = info.get(
                    "needs_action", torch.ones(1, device=self.device)
                )

                new_action = None
                if needs_action[0]:
                    with torch.no_grad():
                        output = network(state)
                        new_action = torch.argmax(output, dim=1)

                state, reward, done, info = env.step(new_action)
                step_count += 1

            distance = info["distances"][0].item()
            crashed = info["crashed"][0].item()

            total_distance += distance
            total_crashes += crashed

            print(f"  Test {test + 1}: Distance={distance:.1f}, Crashed={crashed}")

        avg_distance = total_distance / num_tests
        crash_rate = total_crashes / num_tests * 100

        print(f"\n📊 Test Results:")
        print(f"   Average Distance: {avg_distance:.1f}")
        print(f"   Crash Rate: {crash_rate:.1f}%")

        return avg_distance, crash_rate

    def save_best_genome(self, filename: str):
        """Save the best genome"""
        if self.best_genome:
            with open(filename, "wb") as f:
                pickle.dump(self.best_genome, f)
            print(f"💾 Best genome saved to {filename}")

    def load_genome(self, filename: str):
        """Load a genome from file"""
        with open(filename, "rb") as f:
            genome = pickle.load(f)
        print(f"📁 Genome loaded from {filename}")
        return genome


def create_neat_config(config_path: str, population_size: int = 256):
    """Create a NEAT configuration file"""

    config_content = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = 50000
pop_size              = {population_size}
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = tanh
activation_mutate_rate  = 0.05
activation_options      = tanh relu sigmoid

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.05
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.3
conn_delete_prob        = 0.3

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = full

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 0
num_inputs              = 19
num_outputs             = 5

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""

    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"📝 NEAT config created: {config_path}")


if __name__ == "__main__":
    # Quick demo with detailed progress
    config_path = "neat_config.txt"
    population_size = 128  # Smaller for faster demo
    max_generations = 10  # Fewer generations to see quick progress

    print("🚀 GPU NEAT Demo - Enhanced Progress Reporting")
    print("=" * 50)

    create_neat_config(config_path, population_size)

    trainer = GPUNEATTrainer(
        config_path=config_path,
        population_size=population_size,
        max_generations=max_generations,
    )

    # Run evolution with detailed progress
    print("\nRunning evolution with detailed progress reporting...")
    winner, stats = trainer.run_evolution()

    # Test the best genome
    trainer.test_best_genome(winner, num_tests=3)

    # Save the best
    trainer.save_best_genome("best_race_car_genome.pkl")

    print(f"\n🎉 Demo complete! Best genome saved.")
    print(f"💡 To run longer evolution, increase max_generations in the script.")
