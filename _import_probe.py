import sys

print("BEFORE", sys.version, flush=True)
import torch

print("AFTER", torch.__version__, flush=True)
print("CUDA", torch.cuda.is_available(), flush=True)
print("END", flush=True)
