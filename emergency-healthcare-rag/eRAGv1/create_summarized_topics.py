#!/usr/bin/env python3
"""
Summarize all condensed topics to create summarized_topics using ChatGPT API
"""

from pathlib import Path
import json
import openai
import time
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONDENSED_DIR = Path("data/condensed_topics")
SUMMARIZED_DIR = Path("data/summarized_topics")
TOPIC_MAP = json.loads(Path("data/topics.json").read_text())

# Create output directory
SUMMARIZED_DIR.mkdir(exist_ok=True)

# Load the summarization prompt
with open("eRAGv1/summarize_prompt.md", "r") as f:
    PROMPT = f.read()

# OpenAI configuration
# Set your API key in environment variable: export OPENAI_API_KEY="your-key-here"
# Or uncomment and set it here:
# openai.api_key = "your-api-key-here"

def summarize_with_chatgpt(content: str, topic_name: str) -> str:
    """Summarize content using ChatGPT API with the medical prompt."""
    
    # Construct the full prompt
    full_prompt = f"""{PROMPT}

Topic: {topic_name}

Content:
{content}

Summarized version:"""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper
            messages=[
                {"role": "system", "content": "You are a medical expert specializing in emergency medicine. Your task is to summarize medical topics while preserving all critical diagnostic and treatment information."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=1500,  # Adjust based on expected summary length
            temperature=0.1,  # Low temperature for consistent, factual output
            top_p=0.9
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error summarizing {topic_name}: {e}")
        return content  # Return original if API call fails


def process_all_topics():
    """Process all topics in condensed_topics."""
    
    # Get all markdown files
    md_files = list(CONDENSED_DIR.rglob("*.md"))
    
    print(f"Found {len(md_files)} topics to process")
    print("Using ChatGPT API for summarization...")
    print("-" * 50)
    
    for md_file in tqdm(md_files, desc="Processing topics"):
        topic_name = md_file.parent.name
        
        # Create topic directory
        topic_dir = SUMMARIZED_DIR / topic_name
        topic_dir.mkdir(exist_ok=True)
        
        # Read original content
        content = md_file.read_text(encoding="utf-8")
        
        # Skip if already processed (for resume capability)
        output_file = topic_dir / md_file.name
        if output_file.exists():
            print(f"Skipping {topic_name}/{md_file.name} (already exists)")
            continue
        
        # Summarize with ChatGPT
        summarized = summarize_with_chatgpt(content, topic_name)
        
        # Save summarized version
        output_file.write_text(summarized, encoding="utf-8")
        
        # Rate limiting - be nice to the API
        time.sleep(1)  # 1 second delay between requests
        
        print(f"✓ Processed: {topic_name}/{md_file.name}")
    
    print("\n" + "=" * 50)
    print("Summarization complete!")
    print(f"Output directory: {SUMMARIZED_DIR}")


def test_single_topic():
    """Test summarization on a single topic first."""
    print("=== TESTING SINGLE TOPIC ===")
    
    # Find first topic
    md_files = list(CONDENSED_DIR.rglob("*.md"))
    if not md_files:
        print("No topics found!")
        return
    
    test_file = md_files[0]
    topic_name = test_file.parent.name
    
    print(f"Testing with: {topic_name}/{test_file.name}")
    
    # Read content
    content = test_file.read_text(encoding="utf-8")
    original_length = len(content.split())
    
    print(f"Original length: {original_length} words")
    print("\nOriginal content (first 200 chars):")
    print(content[:200] + "...")
    
    # Summarize
    summarized = summarize_with_chatgpt(content, topic_name)
    summarized_length = len(summarized.split())
    
    print(f"\nSummarized length: {summarized_length} words")
    print(f"Reduction: {((original_length - summarized_length) / original_length * 100):.1f}%")
    
    print("\nSummarized content:")
    print(summarized)
    
    # Ask user if they want to proceed
    response = input("\nDoes this look good? Proceed with all topics? (y/n): ")
    if response.lower() == 'y':
        process_all_topics()
    else:
        print("Aborted. You can modify the prompt and try again.")


if __name__ == "__main__":
    # Check if API key is set
    try:
        # Test API connection
        openai.Model.list()
        print("✓ OpenAI API connection successful")
    except Exception as e:
        print("❌ OpenAI API connection failed!")
        print("Please set your API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("Or set it in the script: openai.api_key = 'your-key-here'")
        exit(1)
    
    # Ask user what to do
    print("Choose an option:")
    print("1. Test with single topic first")
    print("2. Process all topics")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        test_single_topic()
    elif choice == "2":
        process_all_topics()
    else:
        print("Invalid choice. Exiting.") 