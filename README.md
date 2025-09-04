## docker command
docker compose build
docker compose up -d
docker exec -it ft bash

## GPU check
nvidia-smi

## hugging face login
hf auth login

## torch test
```bash
docker exec -it ft bash -lc 'pwd; which python; python - <<PY
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("capability:", torch.cuda.get_device_capability(0))
    print("device:", torch.cuda.get_device_name(0))
PY'
```