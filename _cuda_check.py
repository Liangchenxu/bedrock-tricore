import torch

print("TORCH", torch.__version__, flush=True)
print("CUDA_AVAILABLE", torch.cuda.is_available(), flush=True)
print("DEVICE_COUNT", torch.cuda.device_count(), flush=True)
if torch.cuda.is_available():
    print("DEVICE", torch.cuda.get_device_name(0), flush=True)
print("DONE", flush=True)
