#!/usr/bin/env python3
"""
Improved Summarization Script - Create summarized_topics using GPT-4 Turbo
"""

import os
import logging
from pathlib import Path
import json
from openai import OpenAI
import time
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv('eRAGv1/.env')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORIGINAL_DIR = Path("data/condensed_topics")  # Use condensed topics (already cleaned up)
SUMMARIZED_DIR = Path("data/summarized_topics")
TOPIC_MAP = json.loads(Path("data/topics.json").read_text())
PROGRESS_FILE = Path("eRAGv1/summarization_progress.json")

# Create output directory
SUMMARIZED_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eRAGv1/summarization.log'),
        logging.StreamHandler()
    ]
)

# OpenAI configuration
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Enhanced prompt for better summarization
ENHANCED_PROMPT = """You are a medical expert specializing in emergency medicine. Your task is to summarize complex clinical topics for emergency physicians and AI-based retrieval systems. The summaries must be concise yet **fully preserve all diagnostically and therapeutically relevant medical content** — including rare variants, edge-case presentations, and mechanistic insights.

## CRITICAL REQUIREMENTS

### MUST PRESERVE:
- All **diagnostic criteria** (signs, symptoms, lab thresholds, imaging findings, QRS morphology, ECG progression)
- All **treatment protocols** (medications, procedures, doses, timelines)
- All **differential diagnoses** and distinguishing features
- All **terminology** (clinical terms, synonyms, acronyms — do not simplify or omit)
- All **pathophysiology** (causal mechanisms, disease evolution, collateral systems, cellular/molecular theories)
- All **contraindications**, warnings, and edge-case findings
- All **anatomical references**, numbers, values, and classifications (e.g., Killip, TIMI)
- All **epidemiology** and **rare presentations** (e.g., uncommon vessels, transient ECG patterns)

### MUST MAINTAIN:
- **Medical accuracy** — no omissions, no hallucinated treatments or facts
- **Clinical relevance** — exclude general background unless needed for triage/diagnosis
- **Terminology precision** — use exact terms, not lay explanations
- **Logical flow** — preserve symptom-to-diagnosis-to-treatment chain
- **Structural clarity** — outputs must be consistently formatted

### CAN REMOVE:
- Repetitive or verbose background information
- Historical anecdotes, reference citations
- Multiple examples of the same concept

---

## OUTPUT FORMAT

1. **Definition/Overview** — 1–2 sentence summary
2. **Key Symptoms & Signs** — bullet list
3. **Diagnostic Criteria** — key thresholds, ECG/lab/imaging, morphology
4. **Differential Diagnosis** — relevant alternatives only
5. **Treatment Options** — immediate and definitive steps
6. **Critical Considerations** — ECG progression, complications, warnings
7. **Epidemiology & Pathophysiology** — prevalence, mechanisms, rare variants

---

## LENGTH STRATEGY

Prioritize **preservation over compression**. Do not shorten at the cost of detail. Only compress where repetition or background bloat is present. Summaries will typically land between 50–90% of original length.

---

## FORMATTING RULES

- Use **bolded section headers** exactly as listed
- Use **bullet points** under all sections except the overview
- Do not invent or infer facts not present in the original material
- Maintain clinical tone with no conversational language

---

Now summarize the following medical topic:"""

def load_progress():
    """Load progress from JSON file."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load progress file: {e}")
    return {"completed": [], "failed": [], "total_processed": 0}

def save_progress(progress):
    """Save progress to JSON file."""
    try:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
    except Exception as e:
        logging.error(f"Could not save progress: {e}")

def update_progress(topic_name, status="completed", error=None):
    """Update progress and save immediately."""
    progress = load_progress()
    
    if status == "completed":
        if topic_name not in progress["completed"]:
            progress["completed"].append(topic_name)
        if topic_name in progress["failed"]:
            progress["failed"].remove(topic_name)
    elif status == "failed":
        if topic_name not in progress["failed"]:
            progress["failed"].append(topic_name)
        if topic_name in progress["completed"]:
            progress["completed"].remove(topic_name)
    
    progress["total_processed"] = len(progress["completed"]) + len(progress["failed"])
    
    if error:
        progress.setdefault("errors", {})[topic_name] = str(error)
    
    save_progress(progress)
    return progress

def combine_topic_files(topic_dir: Path) -> str:
    """Combine all .md files in a topic directory into one comprehensive document."""
    combined_content = []
    
    md_files = list(topic_dir.glob("*.md"))
    logging.info(f"Found {len(md_files)} .md files in {topic_dir.name}")
    
    for md_file in sorted(md_files):
        content = md_file.read_text(encoding="utf-8").strip()
        if content:
            combined_content.append(f"## {md_file.stem}\n{content}\n")
            logging.info(f"Added {md_file.name} ({len(content)} chars)")
        else:
            logging.warning(f"Skipped {md_file.name} (empty content)")
    
    result = "\n".join(combined_content)
    logging.info(f"Combined content length: {len(result)} characters")
    return result

def summarize_with_gpt4_turbo(content: str, topic_name: str) -> str:
    """Summarize content using GPT-4 Turbo with enhanced prompt."""
    
    # Content length check - truncate if over 100,000 characters (much higher limit)
    if len(content) > 100000:
        logging.warning(f"Content for {topic_name} truncated from {len(content)} to 100000 characters")
        content = content[:100000] + "\n\n[Content truncated due to length]"
    
    # Construct the full prompt
    full_prompt = f"""{ENHANCED_PROMPT}

TOPIC: {topic_name}

ORIGINAL CONTENT:
{content}

SUMMARIZED VERSION:"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",  # GPT-4 Turbo
            messages=[
                {"role": "system", "content": "You are a medical expert specializing in emergency medicine."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=2000,  # Increased for comprehensive summaries
            temperature=0.1,  # Low temperature for consistent, factual output
            top_p=0.9
        )
        
        result = response.choices[0].message.content.strip()
        
        # Add topic name at the beginning
        result = f"# {topic_name}\n\n{result}"
        
        logging.info(f"Successfully summarized {topic_name}")
        return result
        
    except Exception as e:
        if "rate_limit" in str(e).lower():
            logging.warning(f"Rate limit hit for {topic_name}, waiting 60 seconds...")
            time.sleep(60)
            return summarize_with_gpt4_turbo(content, topic_name)  # Retry
        else:
            logging.error(f"API error for {topic_name}: {e}")
            raise e  # Re-raise to handle in calling function

def save_summarized_topic(topic_name: str, summarized_content: str, original_words: int, summarized_words: int):
    """Save a single summarized topic immediately."""
    try:
        # Create topic directory in summarized_topics
        output_dir = SUMMARIZED_DIR / topic_name
        output_dir.mkdir(exist_ok=True)
        
        # Save the file
        output_file = output_dir / f"{topic_name}.md"
        output_file.write_text(summarized_content, encoding="utf-8")
        
        # Calculate compression ratio
        compression = ((original_words - summarized_words) / original_words * 100) if original_words > 0 else 0
        
        # Update progress immediately
        update_progress(topic_name, "completed")
        
        logging.info(f"✓ {topic_name}: {original_words} → {summarized_words} words ({compression:.1f}% reduction)")
        print(f"✓ {topic_name}: {original_words} → {summarized_words} words ({compression:.1f}% reduction)")
        print(f"  Saved: {output_file}")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to save {topic_name}: {e}")
        update_progress(topic_name, "failed", e)
        print(f"✗ Failed to save {topic_name}: {e}")
        return False

def process_all_topics():
    """Process all topics in the original topics directory with immediate saving."""
    
    # Load existing progress
    progress = load_progress()
    print(f"Resuming from previous run: {len(progress['completed'])} completed, {len(progress['failed'])} failed")
    
    # Get all topic directories
    topic_dirs = [d for d in ORIGINAL_DIR.iterdir() if d.is_dir()]
    
    logging.info(f"Found {len(topic_dirs)} topics to process")
    print(f"Found {len(topic_dirs)} topics to process")
    print("Using GPT-4 Turbo for summarization...")
    print("-" * 50)
    
    successful = len(progress['completed'])
    failed = len(progress['failed'])
    
    # Filter out already completed topics
    remaining_topics = [d for d in topic_dirs if d.name not in progress['completed']]
    
    print(f"Topics to process: {len(remaining_topics)}")
    print(f"Already completed: {len(progress['completed'])}")
    
    for topic_dir in tqdm(remaining_topics, desc="Processing topics"):
        topic_name = topic_dir.name
        
        # Skip if already processed (double-check)
        output_file = SUMMARIZED_DIR / topic_name / f"{topic_name}.md"
        if output_file.exists():
            logging.info(f"Skipping {topic_name} (already exists)")
            print(f"Skipping {topic_name} (already exists)")
            successful += 1
            continue
        
        try:
            # Combine all .md files in the topic
            combined_content = combine_topic_files(topic_dir)
            
            if not combined_content.strip():
                logging.warning(f"No content found in {topic_name}")
                print(f"Warning: No content found in {topic_name}")
                update_progress(topic_name, "failed", "No content found")
                failed += 1
                continue
            
            # Summarize with GPT-4 Turbo
            summarized = summarize_with_gpt4_turbo(combined_content, topic_name)
            
            # Save immediately after successful summarization
            original_words = len(combined_content.split())
            summarized_words = len(summarized.split())
            
            if save_summarized_topic(topic_name, summarized, original_words, summarized_words):
                successful += 1
            else:
                failed += 1
            
        except Exception as e:
            logging.error(f"✗ Failed to process {topic_name}: {e}")
            print(f"✗ Failed to process {topic_name}: {e}")
            update_progress(topic_name, "failed", str(e))
            failed += 1
        
        # Rate limiting - be nice to the API
        time.sleep(2)  # 2 second delay between requests
        
        # Print progress every 10 topics
        if (successful + failed) % 10 == 0:
            print(f"\n--- Progress: {successful + failed}/{len(topic_dirs)} topics processed ---")
    
    logging.info(f"Summarization complete! Successful: {successful}, Failed: {failed}")
    print("\n" + "=" * 50)
    print("Summarization complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {SUMMARIZED_DIR}")
    print(f"Progress saved to: {PROGRESS_FILE}")

def test_single_topic():
    """Test summarization on a single topic first."""
    print("=== TESTING SINGLE TOPIC ===")
    
    # Find first topic
    topic_dirs = [d for d in ORIGINAL_DIR.iterdir() if d.is_dir()]
    if not topic_dirs:
        print("No topics found!")
        return
    
    test_dir = topic_dirs[0]
    topic_name = test_dir.name
    
    print(f"Testing with: {topic_name}")
    
    # Combine content
    combined_content = combine_topic_files(test_dir)
    original_length = len(combined_content.split())
    
    print(f"Original length: {original_length} words")
    print(f"Number of files: {len(list(test_dir.glob('*.md')))}")
    print("\nOriginal content (first 300 chars):")
    print(combined_content[:300] + "...")
    
    # Summarize
    summarized = summarize_with_gpt4_turbo(combined_content, topic_name)
    summarized_length = len(summarized.split())
    
    print(f"\nSummarized length: {summarized_length} words")
    print(f"Reduction: {((original_length - summarized_length) / original_length * 100):.1f}%")
    
    # Save the test file
    if save_summarized_topic(topic_name, summarized, original_length, summarized_length):
        print(f"\n✓ Test file saved successfully!")
    else:
        print(f"\n✗ Failed to save test file!")
    
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
    if not os.getenv('OPENAI_API_KEY'):
        logging.error("OpenAI API key not found!")
        print("❌ OpenAI API key not found!")
        print("Please set OPENAI_API_KEY in eRAGv1/.env file")
        exit(1)
    
    try:
        # Test API connection
        client.models.list()
        logging.info("OpenAI API connection successful")
        print("✓ OpenAI API connection successful")
    except Exception as e:
        logging.error(f"OpenAI API connection failed: {e}")
        print("❌ OpenAI API connection failed!")
        print(f"Error: {e}")
        exit(1)
    
    # Ask user what to do
    print("Choose an option:")
    print("1. Test with single topic first")
    print("2. Process all topics")
    print("3. Show progress")
    
    choice = input("Enter choice (1, 2, or 3): ")
    
    if choice == "1":
        test_single_topic()
    elif choice == "2":
        process_all_topics()
    elif choice == "3":
        progress = load_progress()
        print(f"\nProgress Summary:")
        print(f"Completed: {len(progress['completed'])} topics")
        print(f"Failed: {len(progress['failed'])} topics")
        if progress['completed']:
            print(f"Completed topics: {', '.join(progress['completed'][:5])}{'...' if len(progress['completed']) > 5 else ''}")
        if progress['failed']:
            print(f"Failed topics: {', '.join(progress['failed'][:5])}{'...' if len(progress['failed']) > 5 else ''}")
    else:
        print("Invalid choice. Exiting.") 