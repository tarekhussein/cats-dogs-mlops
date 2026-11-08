
import torch

# Select Apple Silicon GPU if available
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device: {device}")

# Create two random matrices on the selected device
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)

# Perform matrix multiplication
c = torch.matmul(a, b)

print(f"Result shape: {c.shape}")
print(f"Result device: {c.device}")
print("GPU computation successful!")
