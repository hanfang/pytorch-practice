"""
PyTorch Tutorial - Part 3: Neural Networks
Building and training neural networks - essential for interviews
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

print("=== PyTorch Neural Networks ===")

# 1. Simple linear layer
print("\n1. Simple Linear Layer:")
linear = nn.Linear(3, 2)  # 3 inputs, 2 outputs
print(f"Linear layer: {linear}")
print(f"Weight shape: {linear.weight.shape}")
print(f"Bias shape: {linear.bias.shape}")

x = torch.randn(5, 3)  # Batch size 5, 3 features
output = linear(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# 2. Simple feedforward network
print("\n2. Simple Feedforward Network:")
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = SimpleNet(10, 5, 1)
print(f"Network: {net}")

x = torch.randn(32, 10)  # Batch size 32
output = net(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")

# 3. Common activation functions
print("\n3. Activation Functions:")
x = torch.linspace(-3, 3, 100)
activations = {
    'ReLU': F.relu(x),
    'Sigmoid': F.sigmoid(x), 
    'Tanh': F.tanh(x),
    'LeakyReLU': F.leaky_relu(x, negative_slope=0.1),
    'Softmax': F.softmax(x.unsqueeze(0), dim=1).squeeze(0)
}

for name, activation in activations.items():
    print(f"{name} output range: [{activation.min():.3f}, {activation.max():.3f}]")

# 4. Loss functions
print("\n4. Loss Functions:")
# Binary classification
pred_binary = torch.randn(10, 1)
target_binary = torch.randint(0, 2, (10, 1)).float()
bce_loss = F.binary_cross_entropy_with_logits(pred_binary, target_binary)
print(f"Binary Cross Entropy Loss: {bce_loss:.4f}")

# Multi-class classification
pred_multi = torch.randn(10, 5)  # 10 samples, 5 classes
target_multi = torch.randint(0, 5, (10,))  # Class indices
ce_loss = F.cross_entropy(pred_multi, target_multi)
print(f"Cross Entropy Loss: {ce_loss:.4f}")

# Regression
pred_reg = torch.randn(10, 1)
target_reg = torch.randn(10, 1)
mse_loss = F.mse_loss(pred_reg, target_reg)
print(f"MSE Loss: {mse_loss:.4f}")

# 5. Complete training example - XOR problem
print("\n5. Complete Training Example - XOR Problem:")

class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

model = XORNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

print("Training XOR network...")
losses = []
for epoch in range(1000):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# Test the trained model
with torch.no_grad():
    test_outputs = model(X)
    predictions = (test_outputs > 0.5).float()
    accuracy = (predictions == y).float().mean()
    print(f"\nFinal predictions vs targets:")
    for i in range(len(X)):
        print(f"Input: {X[i].numpy()}, Predicted: {test_outputs[i].item():.4f}, Target: {y[i].item()}")
    print(f"Accuracy: {accuracy.item():.4f}")

# 6. Model saving and loading
print("\n6. Model Saving and Loading:")
# Save the model
torch.save(model.state_dict(), '/Users/hanfang/Coding/pytorch-practice/xor_model.pth')
print("Model saved to xor_model.pth")

# Load the model
new_model = XORNet()
new_model.load_state_dict(torch.load('/Users/hanfang/Coding/pytorch-practice/xor_model.pth'))
new_model.eval()

# Verify loaded model works
with torch.no_grad():
    loaded_outputs = new_model(X)
    print("Loaded model predictions:", loaded_outputs.flatten())

# 7. Common neural network patterns
print("\n7. Common Neural Network Patterns:")

# Dropout for regularization
class NetWithDropout(nn.Module):
    def __init__(self):
        super(NetWithDropout, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Only active during training
        x = self.fc2(x)
        return x

dropout_net = NetWithDropout()
print("Network with dropout:", dropout_net)

# Batch normalization
class NetWithBatchNorm(nn.Module):
    def __init__(self):
        super(NetWithBatchNorm, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

bn_net = NetWithBatchNorm()
print("Network with batch norm:", bn_net)

# 8. Parameter counting and model info
print("\n8. Model Information:")
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Simple net parameters: {count_parameters(net)}")
print(f"XOR net parameters: {count_parameters(model)}")

# Print model architecture
print("\nXOR model architecture:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

print("\n=== End of Neural Networks ===")