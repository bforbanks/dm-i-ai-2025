# W&B Setup on DTU HPC

## 1. Install wandb in your venv (do once on the HPC)

```bash
ssh sXXXXXX@login1.gbar.dtu.dk
cd ~/Desktop/dm-i-ai-2025
source venv/bin/activate
pip install wandb
```

## 2. API key for batch jobs (pick one)

### Option A — `wandb.env` (recommended for submit scripts)

The job scripts in `train_run_1/` source `wandb.env` if it exists (file is gitignored).

```bash
cd ~/Desktop/dm-i-ai-2025/race-car/WorldModel/train_run_1
cp wandb.env.example wandb.env
# Edit wandb.env: set export WANDB_API_KEY="..."
```

### Option B — Log in interactively (stores key in ~/.netrc)

```bash
wandb login
# Paste your API key from https://wandb.ai/settings (copy "API keys" section)
```

This writes your key to `~/.netrc` and persists across sessions and batch jobs. LSF jobs still see it if the cluster reads `~/.netrc` for your user (usually yes).

### Option C — `~/.bashrc`

```bash
export WANDB_API_KEY="your_key_here"
```

Note: non-interactive batch jobs may not source `~/.bashrc` unless your site does; prefer A or B.

## 3. Verify it works on an interactive node

```bash
voltash       # open interactive GPU node
source ~/Desktop/dm-i-ai-2025/venv/bin/activate
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
