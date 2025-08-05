#!/usr/bin/env python3
"""
Quick test to verify NEAT neural network creation works
"""
import torch
from neat_gpu_interface import NEATNeuralNetwork, create_neat_config, GPUNEATTrainer
import neat

def test_neat_network_creation():
    """Test if we can create NEAT neural networks without errors"""
    print("🧪 Testing NEAT Neural Network Creation...")
    
    # Create config
    config_path = "test_neat_config.txt"
    create_neat_config(config_path, population_size=4)  # Small test
    
    # Load NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    # Create a test population
    population = neat.Population(config)
    
    print("✅ NEAT config and population created successfully")
    
    # Get a few genomes to test
    genomes = list(population.population.items())[:3]
    print(f"✅ Got {len(genomes)} test genomes")
    
    # Test creating neural networks
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for i, (genome_id, genome) in enumerate(genomes):
        try:
            print(f"  Testing genome {i+1}/{len(genomes)}...", end=" ")
            
            # Create network
            net = NEATNeuralNetwork(genome, config)
            net.to(device)
            net.eval()
            
            # Test forward pass
            with torch.no_grad():
                test_input = torch.randn(1, 19, device=device)  # 19 input features
                output = net(test_input)
                
            assert output.shape == (1, 5), f"Expected output shape (1, 5), got {output.shape}"
            print("✅ Success!")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
    
    print("🎉 All NEAT neural networks created successfully!")
    return True

def test_trainer_creation():
    """Test if we can create the GPU trainer"""
    print("\n🧪 Testing GPU NEAT Trainer Creation...")
    
    try:
        config_path = "test_neat_config.txt"
        
        trainer = GPUNEATTrainer(
            config_path=config_path,
            population_size=4,  # Very small for test
            max_generations=2
        )
        
        print("✅ GPU NEAT Trainer created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create trainer: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing NEAT GPU Interface Fixes")
    print("=" * 40)
    
    success = True
    
    # Test 1: Neural network creation
    success &= test_neat_network_creation()
    
    # Test 2: Trainer creation  
    success &= test_trainer_creation()
    
    if success:
        print("\n🎉 All tests passed! NEAT GPU interface is working.")
        print("You can now run: python neat_gpu_interface.py")
    else:
        print("\n❌ Some tests failed. Check the errors above.")