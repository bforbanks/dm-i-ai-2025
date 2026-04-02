#!/bin/bash
# Submit all 6 WorldModel training jobs concurrently.
# Run from the root of dm-i-ai-2025/ after SSHing into the HPC.
#
# Usage:
#   cd ~/Desktop/dm-i-ai-2025
#   bash race-car/WorldModel/train_run_1/submit_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in \
    submit_model_a.sh \
    submit_model_a_lg.sh \
    submit_model_b.sh \
    submit_model_b_lg.sh \
    submit_model_c.sh \
    submit_model_d.sh
do
    echo "Submitting $script …"
    bsub < "$SCRIPT_DIR/$script"
done

echo ""
echo "All jobs submitted. Check status with: bstat"
