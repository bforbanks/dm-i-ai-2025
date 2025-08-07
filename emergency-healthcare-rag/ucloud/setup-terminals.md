# Terminal Setup

## 1st Ubuntu terminal:
```bash
tmux new -s ollama_server
cd ..
OLLAMA_MODELS=/work/TEAM13/models /work/TEAM13/ollama/bin/ollama serve
```

## 2nd Ubuntu terminal:
```bash
tmux new -s main
cd TEAM13/dm-i-ai-2025/emergency-healthcare-rag/
git config --global credential.helper store
git fetch
git checkout emergency-healthcare-rag
git pull
```
