"""
PyTorch Tutorial - Part 1: Tensor Basics
Essential tensor operations for PyTorch interviews
"""

import torch
import numpy as np

print("=== PyTorch Tensor Basics ===")

# 1. Creating tensors
print("\n1. Creating Tensors:")
tensor_from_data = torch.tensor([1, 2, 3, 4])
print(f"From list: {tensor_from_data}")

tensor_zeros = torch.zeros(2, 3)
print(f"Zeros tensor:\n{tensor_zeros}")

tensor_ones = torch.ones(2, 3)
print(f"Ones tensor:\n{tensor_ones}")

tensor_random = torch.rand(2, 3)
print(f"Random tensor:\n{tensor_random}")

tensor_randn = torch.randn(2, 3)  # Normal distribution
print(f"Normal random tensor:\n{tensor_randn}")

# 2. Tensor properties
print("\n2. Tensor Properties:")
x = torch.randn(3, 4, 5)
print(f"Shape: {x.shape}")
print(f"Size: {x.size()}")
print(f"Data type: {x.dtype}")
print(f"Device: {x.device}")
print(f"Number of dimensions: {x.dim()}")
print(f"Number of elements: {x.numel()}")

# 3. Basic operations
print("\n3. Basic Operations:")
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(f"a: {a}")
print(f"b: {b}")
print(f"Addition: {a + b}")
print(f"Element-wise multiplication: {a * b}")
print(f"Matrix multiplication: {torch.dot(a, b)}")

# 4. Reshaping tensors
print("\n4. Reshaping Tensors:")
x = torch.randn(12)
print(f"Original shape: {x.shape}")

reshaped = x.view(3, 4)
print(f"Reshaped (3,4):\n{reshaped}")

reshaped2 = x.view(-1, 2)  # -1 means infer this dimension
print(f"Reshaped (-1,2):\n{reshaped2}")

# 5. Indexing and slicing
print("\n5. Indexing and Slicing:")
x = torch.randn(3, 4)
print(f"Original tensor:\n{x}")
print(f"First row: {x[0]}")
print(f"First column: {x[:, 0]}")
print(f"Element at (1,2): {x[1, 2]}")
print(f"Last 2 rows, last 2 cols:\n{x[-2:, -2:]}")

# 6. Common tensor operations for interviews
print("\n6. Common Interview Operations:")

# Broadcasting
a = torch.tensor([[1], [2], [3]])  # 3x1
b = torch.tensor([10, 20, 30])     # 1x3
result = a + b  # Broadcasting to 3x3
print(f"Broadcasting example:\n{result}")

# Concatenation
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[5, 6], [7, 8]])
concat_dim0 = torch.cat([x, y], dim=0)  # Along rows
concat_dim1 = torch.cat([x, y], dim=1)  # Along columns
print(f"Concatenate along dim=0:\n{concat_dim0}")
print(f"Concatenate along dim=1:\n{concat_dim1}")

# Stacking
stack_dim0 = torch.stack([x, y], dim=0)  # New dimension at index 0
print(f"Stack along dim=0 shape: {stack_dim0.shape}")
print(f"Stack along dim=0:\n{stack_dim0}")

# 7. Converting between NumPy and PyTorch
print("\n7. NumPy <-> PyTorch Conversion:")
numpy_array = np.array([1, 2, 3, 4])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"NumPy to Tensor: {tensor_from_numpy}")

tensor = torch.tensor([1, 2, 3, 4])
numpy_from_tensor = tensor.numpy()
print(f"Tensor to NumPy: {numpy_from_tensor}")

# 8. GPU operations (if available)
print("\n8. GPU Operations:")
if torch.cuda.is_available():
    device = torch.device("cuda")
    tensor_gpu = torch.randn(3, 4).to(device)
    print(f"Tensor on GPU: {tensor_gpu.device}")
    tensor_cpu = tensor_gpu.cpu()
    print("Moved back to CPU")
else:
    print("CUDA not available, using CPU")

# 9. Gradient computation basics
print("\n9. Gradient Computation:")
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
print(f"x: {x}")
print(f"y = x² + 3x + 1: {y}")

y.backward()  # Compute gradients
print(f"dy/dx at x=2: {x.grad}")  # Should be 2*2 + 3 = 7

print("\n=== End of Tensor Basics ===")