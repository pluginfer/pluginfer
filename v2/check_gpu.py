import torch
import sys
try:
    import torch_directml
    dml = True
except ImportError:
    dml = False

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
print(f"DirectML Available: {dml}")
