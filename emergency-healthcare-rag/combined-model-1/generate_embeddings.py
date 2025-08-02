import sys
import os
sys.path.append('..')
from search import create_embeddings

if __name__ == "__main__":
    print("Generating embeddings for medical documents...")
    create_embeddings()
    print("Done! You can now run the model.")