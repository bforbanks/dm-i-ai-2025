# W&B Setup on DTU HPC

## 1. Install wandb in your venv (do once on the HPC)

```bash
ssh sXXXXXX@login1.gbar.dtu.dk
cd ~/dm-i-ai-2025
source venv/bin/activate
pip install wandb
```

## 2. Log in interactively (do once — stores API key in ~/.netrc)

```bash
wandb login
# Paste your API key from https://wandb.ai/settings (copy "API keys" section)
```

This writes your key to `~/.netrc` and persists across sessions and batch jobs.

**Alternative — environment variable (more portable):**

Add to `~/.bashrc`:
```bash
export WANDB_API_KEY="your_key_here"
```

Or set it directly in your job `.sh` script before the Python call:
```bash
export WANDB_API_KEY="your_key_here"
python race-car/WorldModel/train.py ...
```

## 3. Verify it works on an interactive node

```bash
voltash       # open interactive GPU node
source ~/dm-i-ai-2025/venv/bin/activate
python -c "import wandb; wandb.login(); print('wandb OK')"
```

## 4. Check internet connectivity from compute nodes

DTU HPC compute nodes DO have outbound internet access, so wandb can sync
in real time. If you hit firewall issues, fall back to offline mode:

```bash
export WANDB_MODE=offline
# Sync later with: wandb sync <run_dir>
```

## 5. View results

After jobs start, go to https://wandb.ai/ → project `laneshift-worldmodel`.
You should see all 6 runs appear within a minute of each job starting.

## 6. Troubleshooting

- `wandb: Network error` — set `WANDB_MODE=offline` in job script, sync after
- `wandb: Error 401` — API key not set; run `wandb login` or set `WANDB_API_KEY`
- Runs not appearing — check the `.err` file from the job for Python tracebacks
