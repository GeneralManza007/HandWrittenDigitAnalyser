import torch

# Check if PyTorch is installed
print("PyTorch version:", torch.__version__)

# Check for GPU
print("CUDA available:", torch.cuda.is_available())

# Simple tensor test
x = torch.rand(3, 3)
print("Random tensor:\n", x)
