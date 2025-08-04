#!/usr/bin/env python3
"""
Example script demonstrating how to queue a validation attempt and get the score.
"""

from queue_validation import queue_validation_attempt, wait_for_validation_completion
import time

def main():
    print("🚀 Starting validation example...")
    
    # Step 1: Queue a validation attempt
    print("\n1️⃣  Queueing validation attempt...")
    result = queue_validation_attempt()
    
    if "error" in result:
        print(f"❌ Failed to queue validation: {result['error']}")
        return
    
    attempt_uuid = result.get("queued_attempt_uuid")
    if not attempt_uuid:
        print("❌ No attempt UUID received")
        return
    
    print(f"✅ Successfully queued validation attempt: {attempt_uuid}")
    
    # Step 2: Wait for completion and get score
    print("\n2️⃣  Waiting for validation to complete...")
    final_result = wait_for_validation_completion(
        attempt_uuid, 
        max_wait_time=600,  # 10 minutes
        check_interval=15   # Check every 15 seconds
    )
    
    if "error" in final_result:
        print(f"❌ Error during validation: {final_result['error']}")
        return
    
    if final_result.get("status") == "done":
        score = final_result.get("attempt", {}).get("score")
        if score is not None:
            print(f"\n🎉 VALIDATION COMPLETED!")
            print(f"🎯 FINAL SCORE: {score:.6f}")
            
            # Additional details
            attempt_info = final_result.get("attempt", {})
            if attempt_info:
                print(f"📊 Details:")
                print(f"   Submitted: {attempt_info.get('submitted_at', 'N/A')}")
                print(f"   Started: {attempt_info.get('started_at', 'N/A')}")
                print(f"   Finished: {attempt_info.get('finished_at', 'N/A')}")
                print(f"   Service URL: {attempt_info.get('service_url', 'N/A')}")
                
                errors = attempt_info.get('errors', [])
                if errors:
                    print(f"   Errors: {errors}")
                else:
                    print(f"   Errors: None")
        else:
            print("⚠️  Validation completed but no score available")
    elif final_result.get("status") in ["queued", "processing", "in_progress"]:
        print(f"⏳ Validation is still {final_result.get('status')} - you may need to wait longer or check manually")
    else:
        print(f"❓ Unexpected final status: {final_result.get('status')}")

if __name__ == "__main__":
    main() 