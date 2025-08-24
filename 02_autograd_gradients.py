"""
PyTorch Tutorial - Part 2: Autograd and Gradients
Understanding automatic differentiation - crucial for interviews
"""

import torch
import torch.nn as nn

print("=== PyTorch Autograd and Gradients ===")

# 1. Basic gradient computation
print("\n1. Basic Gradient Computation:")
x = torch.tensor([3.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)

z = x * y + x ** 2
print(f"x: {x}, y: {y}")
print(f"z = x*y + x²: {z}")

z.backward()
print(f"dz/dx: {x.grad}")  # Should be y + 2*x = 2 + 6 = 8
print(f"dz/dy: {y.grad}")  # Should be x = 3

# 2. Multiple backward passes (need to zero gradients)
print("\n2. Multiple Backward Passes:")
x = torch.tensor([1.0], requires_grad=True)
for i in range(3):
    y = x ** 2
    y.backward()
    print(f"Iteration {i+1}, x.grad: {x.grad}")
    x.grad.zero_()  # Reset gradients

# 3. Vector gradients
print("\n3. Vector Gradients:")
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
print(f"x: {x}")
print(f"y = x²: {y}")

# For vector outputs, need to provide gradient argument
gradient = torch.tensor([1.0, 1.0, 1.0])
y.backward(gradient)
print(f"dy/dx: {x.grad}")  # Should be 2*x = [2, 4, 6]

# 4. Matrix gradients
print("\n4. Matrix Gradients:")
x = torch.randn(2, 2, requires_grad=True)
y = x ** 2
z = y.mean()  # Scalar output
print(f"x shape: {x.shape}")
print(f"z (mean of x²): {z}")

z.backward()
print(f"dz/dx:\n{x.grad}")  # Should be 2*x/4 (mean divides by 4)

# 5. Higher order gradients
print("\n5. Higher Order Gradients:")
x = torch.tensor([2.0], requires_grad=True)
y = x ** 3
dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"y = x³: {y}")
print(f"dy/dx: {dy_dx}")  # Should be 3*2² = 12

d2y_dx2 = torch.autograd.grad(dy_dx, x)[0]
print(f"d²y/dx²: {d2y_dx2}")  # Should be 6*2 = 12

# 6. Gradient flow control
print("\n6. Gradient Flow Control:")

# detach() stops gradient computation
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y.detach() + x  # y.detach() has no gradient
z.backward()
print(f"With detach, dx: {x.grad}")  # Should be 1 (only from +x term)

# with torch.no_grad() context
x = torch.tensor([2.0], requires_grad=True)
with torch.no_grad():
    y = x ** 2  # No gradient tracked
print(f"y.requires_grad after no_grad: {y.requires_grad}")

# 7. Custom gradient functions (advanced)
print("\n7. Custom Gradient Function:")
class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return 2 * input * grad_output

square = SquareFunction.apply
x = torch.tensor([3.0], requires_grad=True)
y = square(x)
y.backward()
print(f"Custom square function gradient: {x.grad}")  # Should be 6

# 8. Common interview scenario: Loss function gradients
print("\n8. Loss Function Gradients:")
# Simulate a simple linear model: y = wx + b
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.1], requires_grad=True)
x_data = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

# Forward pass
y_pred = w * x_data + b
loss = ((y_pred - y_true) ** 2).mean()  # MSE loss

print(f"Predictions: {y_pred}")
print(f"Loss: {loss}")

# Backward pass
loss.backward()
print(f"dL/dw: {w.grad}")
print(f"dL/db: {b.grad}")

# 9. Gradient accumulation pattern
print("\n9. Gradient Accumulation:")
w = torch.tensor([1.0], requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.01)

for i in range(3):
    # Forward pass
    loss = (w - 5) ** 2
    
    # Backward pass (accumulates gradients)
    loss.backward()
    
    print(f"Step {i+1}: w.grad = {w.grad}")
    
    # Optimizer step and zero gradients
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"Step {i+1}: w = {w}")

print("\n=== End of Autograd and Gradients ===")