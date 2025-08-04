#!/usr/bin/env python3
"""
Script to automatically queue validation attempts and check their status/scores
against the online validation dataset.
"""

import requests
import json
import time
import os
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
VALIDATION_API_URL = os.getenv("VALIDATION_API_URL", "https://cases.dmiai.dk/api/v1/usecases/tumor-segmentation/validate/queue")
VALIDATION_TOKEN = os.getenv("VALIDATION_TOKEN", "18238e2f472643739573ad26f3680c51")
PREDICT_URL = os.getenv("PREDICT_URL", "https://2d4ae66d0358.ngrok-free.app/predict")

def queue_validation_attempt(predict_url: str | None = None) -> dict:
    """
    Queue a validation attempt against the online validation dataset.
    
    Args:
        predict_url: URL for the prediction endpoint (defaults to PREDICT_URL env var)
    
    Returns:
        dict: Response from the validation API containing queue information
    """
    if predict_url is None:
        predict_url = PREDICT_URL
    
    headers = {
        "x-token": VALIDATION_TOKEN,
        "Content-Type": "application/json"
    }
    
    data = {
        "url": predict_url
    }
    
    try:
        print(f"Queueing validation attempt with predict URL: {predict_url}")
        response = requests.post(
            VALIDATION_API_URL,
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        print(f"Successfully queued validation attempt:")
        print(json.dumps(result, indent=2))
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error queueing validation attempt: {e}")
        return {"error": str(e)}

def check_validation_status(attempt_uuid: str) -> dict:
    """
    Check the status of a validation attempt and get the score if completed.
    
    Args:
        attempt_uuid: UUID of the validation attempt to check
    
    Returns:
        dict: Status information including score if completed
    """
    headers = {
        "x-token": VALIDATION_TOKEN
    }
    
    status_url = f"{VALIDATION_API_URL}/{attempt_uuid}"
    
    try:
        print(f"Checking status for attempt: {attempt_uuid}")
        response = requests.get(
            status_url,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        print(f"Status check result:")
        print(json.dumps(result, indent=2))
        
        # Extract and display score if available
        if result.get("status") == "done" and "attempt" in result:
            score = result["attempt"].get("score")
            if score is not None:
                print(f"\n🎯 VALIDATION SCORE: {score:.6f}")
            else:
                print("\n⚠️  Validation completed but no score available")
        elif result.get("status") == "queued":
            position = result.get("position_in_queue", "unknown")
            print(f"\n⏳ Still in queue at position: {position}")
        elif result.get("status") in ["processing", "in_progress"]:
            print("\n🔄 Validation is currently processing...")
        else:
            print(f"\n❓ Unknown status: {result.get('status')}")
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error checking validation status: {e}")
        return {"error": str(e)}

def wait_for_validation_completion(attempt_uuid: str, max_wait_time: int = 300, check_interval: int = 10) -> dict:
    """
    Wait for a validation attempt to complete and return the final result.
    
    Args:
        attempt_uuid: UUID of the validation attempt to monitor
        max_wait_time: Maximum time to wait in seconds (default: 5 minutes)
        check_interval: Interval between status checks in seconds (default: 10 seconds)
    
    Returns:
        dict: Final validation result with score
    """
    print(f"Waiting for validation completion (max {max_wait_time}s, checking every {check_interval}s)...")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        result = check_validation_status(attempt_uuid)
        
        if "error" in result:
            print(f"Error during status check: {result['error']}")
            return result
        
        status = result.get("status")
        
        if status == "done":
            print("✅ Validation completed!")
            return result
        elif status == "failed":
            print("❌ Validation failed!")
            return result
        elif status in ["queued", "processing", "in_progress"]:
            elapsed = time.time() - start_time
            remaining = max_wait_time - elapsed
            print(f"⏳ Still waiting... (elapsed: {elapsed:.0f}s, remaining: {remaining:.0f}s)")
            time.sleep(check_interval)
        else:
            print(f"❓ Unknown status: {status}")
            return result
    
    print(f"⏰ Timeout reached after {max_wait_time}s")
    return {"error": "Timeout waiting for validation completion"}

def queue_multiple_validations(num_attempts: int = 1, delay_seconds: float = 1.0):
    """
    Queue multiple validation attempts with a delay between each.
    
    Args:
        num_attempts: Number of validation attempts to queue
        delay_seconds: Delay between attempts in seconds
    """
    print(f"Queueing {num_attempts} validation attempts with {delay_seconds}s delay between each...")
    
    attempt_uuids = []
    
    for i in range(num_attempts):
        print(f"\n--- Attempt {i+1}/{num_attempts} ---")
        result = queue_validation_attempt()
        
        if "error" in result:
            print(f"Failed to queue attempt {i+1}: {result['error']}")
        else:
            print(f"Successfully queued attempt {i+1}")
            attempt_uuid = result.get("queued_attempt_uuid")
            if attempt_uuid:
                attempt_uuids.append(attempt_uuid)
                print(f"Attempt UUID: {attempt_uuid}")
            if "position_in_queue" in result:
                print(f"Position in queue: {result['position_in_queue']}")
        
        if i < num_attempts - 1:  # Don't sleep after the last attempt
            print(f"Waiting {delay_seconds} seconds before next attempt...")
            time.sleep(delay_seconds)
    
    return attempt_uuids

def main():
    parser = argparse.ArgumentParser(description="Queue validation attempts and check their status/scores")
    parser.add_argument("--action", choices=["queue", "status", "wait"], default="queue",
                       help="Action to perform (default: queue)")
    parser.add_argument("--num-attempts", "-n", type=int, default=1, 
                       help="Number of validation attempts to queue (default: 1)")
    parser.add_argument("--delay", "-d", type=float, default=1.0,
                       help="Delay between attempts in seconds (default: 1.0)")
    parser.add_argument("--predict-url", "-u", type=str, default=None,
                       help="Custom predict URL (defaults to PREDICT_URL env var)")
    parser.add_argument("--attempt-uuid", "-a", type=str, default=None,
                       help="UUID of validation attempt to check (for status/wait actions)")
    parser.add_argument("--max-wait", "-w", type=int, default=300,
                       help="Maximum wait time in seconds for completion (default: 300)")
    parser.add_argument("--check-interval", "-i", type=int, default=10,
                       help="Interval between status checks in seconds (default: 10)")
    
    args = parser.parse_args()
    
    if args.action == "queue":
        if args.num_attempts == 1:
            # Single attempt
            result = queue_validation_attempt(args.predict_url)
            if "queued_attempt_uuid" in result:
                print(f"\n💡 To check status later, use:")
                print(f"python queue_validation.py --action status --attempt-uuid {result['queued_attempt_uuid']}")
        else:
            # Multiple attempts
            attempt_uuids = queue_multiple_validations(args.num_attempts, args.delay)
            if attempt_uuids:
                print(f"\n💡 To check status later, use:")
                for uuid in attempt_uuids:
                    print(f"python queue_validation.py --action status --attempt-uuid {uuid}")
    
    elif args.action == "status":
        if not args.attempt_uuid:
            print("❌ Error: --attempt-uuid is required for status action")
            return
        check_validation_status(args.attempt_uuid)
    
    elif args.action == "wait":
        if not args.attempt_uuid:
            print("❌ Error: --attempt-uuid is required for wait action")
            return
        result = wait_for_validation_completion(
            args.attempt_uuid, 
            args.max_wait, 
            args.check_interval
        )
        
        if "error" not in result and result.get("status") == "done":
            score = result.get("attempt", {}).get("score")
            if score is not None:
                print(f"\n🎉 FINAL SCORE: {score:.6f}")
            else:
                print("\n⚠️  Validation completed but no score available")

if __name__ == "__main__":
    main() 