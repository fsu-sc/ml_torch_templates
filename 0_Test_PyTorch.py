# %%
import torch

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# Create a tensor on the selected device
x = torch.rand(3, 3).to(device)

# Perform a simple operation on the tensor
y = x + x

# Print the result
print(y)
# %%
