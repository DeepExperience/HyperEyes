## How to contribute

1. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run training (see `scripts/` for example launch scripts):

```bash
# Example: 8-GPU Qwen3-4B training
bash scripts/training/text/run-qwen3-4B-8xgpu.sh
```

   Or run the training entry point directly:

```bash
python relax/entrypoints/train.py [ARGS...]
```
