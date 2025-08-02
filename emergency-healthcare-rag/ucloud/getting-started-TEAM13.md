# UCloud Setup Guide - TEAM13

**Important**: GPU resources are shared with limited hours. Stop GPU jobs when not in use and limit your team to **1 GPU instance** at a time.

## Overview
This tutorial covers setting up FastAPI and Ollama (for local LLMs) on UCloud for TEAM13.

## 1. Accept Team Invite
1. Accept your team's UCloud invite link and login with university credentials
2. Share the invite link only with your teammates (not other participants)
3. You may need to click the invite link again after signup to join the team
4. Note your team name: **TEAM13**

## 2. Start GPU Instance

1. Navigate to https://cloud.sdu.dk/app/applications
2. If you get message stating that you need to reconnect to DeiC, press it and press "Reconnect"
3. Click **Terminal**
4. Enter a job name
5. Set hours to 1-4
6. Keep nodes at 1
7. Choose machine type: **uc1-l4-1**  (do not select multi-GPU nodes)
8. Under "Select folders to use":
   - Click **Add folder**
   - Choose your team drive (TEAM13)
   - Files in the team drive persist between runs (50GB storage). All files outside the drive folder will be deleted when the job finishes.
9. Click **Submit** and wait for job to start
10. Click **Open terminal**
11. Team drive location: `/work/TEAM13`
12. Verify GPU: `nvidia-smi`

## 3. Setup Ollama

**Note**: Save all files, programs, and data in the team drive to persist between jobs. Files outside the team drive are deleted when jobs finish.

1. (On the GPU instance) Navigate to team directory:
   ```bash
   cd /work/TEAM13
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/bforbanks/dm-i-ai-2025.git
   cd dm-i-ai-2025
   git checkout emergency-healthcare-rag
   ```

3. Install Ollama (this modified ollama installation script installs ollama in the working directory as opposed to /usr/bin):
   ```bash
   sh emergency-healthcare-rag/ucloud/ollama-install.sh
   ```

4. **Open a new terminal tab** for Ollama server

5. In the new terminal tab, start Ollama server (this will start ollama on port 11434):
   ```bash
   cd /work/TEAM13
   OLLAMA_MODELS=/work/TEAM13/models /work/TEAM13/ollama/bin/ollama serve
   ```

6. **Open another terminal tab** for model management

7. In the new terminal tab, download the model:
   ```bash
   cd /work/TEAM13
   /work/TEAM13/ollama/bin/ollama pull gemma3:27b
   ```

8. Test Ollama in the terminal:
   ```bash
   /work/TEAM13/ollama/bin/ollama run gemma3:27b
   ```
   (Press `Ctrl+D` to exit chat)

9. Verify server is running:
   ```bash
   curl -L localhost:11434
   ```

## 4. Setup FastAPI Server

1. **Open a new terminal tab** for FastAPI

2. In the new terminal tab, navigate to project directory:
   ```bash
   cd /work/TEAM13/dm-i-ai-2025/emergency-healthcare-rag
   ```

3. Create and activate virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

5. Generate embeddings for combined-model-2:
   ```bash
   python combined-model-2/setup.py
   ```

6. Start the FastAPI server:
   ```bash
   python ucloud/api.py
   ```

7. Test the API:
   ```bash
   curl -X POST localhost:8000/predict   -H "Content-Type: application/json"   -d '{"statement": "constipation is a disease"}'
   ```
 
## 5. Setup Nginx (Public Access)

To expose the FastAPI endpoint to the internet:

1. Navigate to https://cloud.sdu.dk/app/applications
2. Search for "nginx" in the upper right corner
3. Select the nginx application
4. Configure the job:
   - Give it a name
   - Machine type: **uc1-gc1-1** (no GPU needed)
   - Duration: 1-4 hours

5. Optional parameters:
   - NGINX configuration: Select `TEAM13/dm-i-ai-2025/emergency-healthcare-rag/ucloud/nginx.conf`

6. Configure custom links to your application: 
   - Click "Add public link" (save this URL - it's your public endpoint)

7. Connect to other jobs: 
   - Click "Connect to job"
   - Hostname: `my-api`
   - Select the GPU job

8. Click **Submit** and wait for startup

9. Verify setup:
   - Check logs for any errors
   - Test the public link in your browser
   - Test from a different machine:
   ```bash
   curl -X POST https://public-link.cloud.aau.dk/predict  -H "Content-Type: application/json"   -d '{"statement": "constipation is a disease"}
   ```

## 6. Terminal Tab Organization

Keep these terminal tabs open:
- **Tab 1**: Main terminal for setup and management
- **Tab 2**: Ollama server (`/work/TEAM13/ollama/bin/ollama serve`)
- **Tab 3**: FastAPI server (`python ucloud/api.py`)
- **Tab 4**: Model management and testing

## 7. Restarting Jobs
To restart nginx or GPU terminal applications:
1. Go to https://cloud.sdu.dk/app/jobs
2. Double-click a finished job
3. Click "Run application again" 

## 8. Model-Agnostic Configuration

The system is designed to work with any Ollama model. You can easily switch between models:

### List Available Models:
```bash
cd /work/TEAM13/dm-i-ai-2025/emergency-healthcare-rag
python combined-model-2/model_switcher.py
```

### Switch to a Different Model:
```bash
# Switch to DeepSeek R1 32B (excellent for reasoning)
python combined-model-2/model_switcher.py deepseek-r1:32b

# Switch to Llama 3.1 70B (maximum performance)
python combined-model-2/model_switcher.py llama3.1:70b

# Switch to Gemma 3 12B (good balance)
python combined-model-2/model_switcher.py gemma3:12b
```

### Download the New Model:
```bash
# After switching, download the new model
/work/TEAM13/ollama/bin/ollama pull <model_name>
```

### Set Model via Environment Variable:
```bash
# Set model directly via environment variable
export LLM_MODEL=deepseek-r1:32b
python combined-model-2/evaluate.py
```

## 9. TEAM13 Specific Notes

- **Repository**: https://github.com/bforbanks/dm-i-ai-2025/tree/emergency-healthcare-rag
- **Team Drive**: `/work/TEAM13`
- **Default Model**: `gemma3:27b` (current, most capable model for single GPU)
- **Active Model**: `combined-model-2` (hybrid search + improved LLM)
- **Model-Agnostic**: Easy switching between different LLM models
- **No screen sessions**: Use separate terminal tabs instead 