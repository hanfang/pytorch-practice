"""
PyTorch Tutorial - Part 5: Advanced Topics and Best Practices
Advanced concepts frequently asked in senior ML engineer interviews
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math

print("=== Advanced PyTorch Topics ===")

# 1. Custom Optimizers
print("\n1. Custom Optimizer Implementation:")
class SGDWithMomentum:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in self.parameters]
    
    def step(self):
        for param, velocity in zip(self.parameters, self.velocities):
            if param.grad is not None:
                velocity.mul_(self.momentum).add_(param.grad, alpha=1)
                param.data.add_(velocity, alpha=-self.lr)
    
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

# Test custom optimizer
model = nn.Linear(2, 1)
custom_optimizer = SGDWithMomentum(model.parameters(), lr=0.01)

x = torch.randn(10, 2)
y = torch.randn(10, 1)

for epoch in range(5):
    pred = model(x)
    loss = F.mse_loss(pred, y)
    
    custom_optimizer.zero_grad()
    loss.backward()
    custom_optimizer.step()
    
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# 2. Attention Mechanism
print("\n2. Attention Mechanism Implementation:")
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_length, d_model = query.size()
        
        # Linear transformations and reshape
        Q = self.W_q(query).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, d_model)
        
        # Final linear transformation
        output = self.W_o(attention_output)
        
        return output, attention_weights

# Test attention
d_model, n_heads, seq_len, batch_size = 512, 8, 10, 2
attention = MultiHeadAttention(d_model, n_heads)

x = torch.randn(batch_size, seq_len, d_model)
output, weights = attention(x, x, x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")

# 3. Residual Connections and Layer Normalization
print("\n3. Residual Connections and Layer Normalization:")
class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        return self.layer_norm(x + self.net(x))  # Residual connection + LayerNorm

residual_block = ResidualBlock(256, 512)
x = torch.randn(32, 256)
output = residual_block(x)
print(f"Residual block - Input: {x.shape}, Output: {output.shape}")

# 4. Learning Rate Warmup and Cosine Annealing
print("\n4. Learning Rate Scheduling with Warmup:")
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_count = 0
    
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Warmup phase
            lr = self.max_lr * (self.step_count / self.warmup_steps)
        else:
            # Cosine annealing phase
            progress = (self.step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

model = nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters())
scheduler = WarmupCosineScheduler(optimizer, warmup_steps=100, max_steps=1000, max_lr=0.001)

print("Learning rate schedule (first 20 steps):")
for step in range(20):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print(f"Step {step+1}: LR = {lr:.6f}")

# 5. Mixed Precision Training
print("\n5. Mixed Precision Training:")
if torch.cuda.is_available():
    print("CUDA available - demonstrating mixed precision")
    
    model = nn.Linear(10, 1).cuda()
    optimizer = optim.Adam(model.parameters())
    scaler = torch.cuda.amp.GradScaler()
    
    x = torch.randn(32, 10).cuda()
    y = torch.randn(32, 1).cuda()
    
    # Mixed precision training loop
    for epoch in range(3):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            pred = model(x)
            loss = F.mse_loss(pred, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"Mixed precision epoch {epoch+1}: Loss = {loss.item():.4f}")
else:
    print("CUDA not available - skipping mixed precision example")

# 6. Custom Backward Hook
print("\n6. Custom Backward Hook:")
def gradient_hook(grad):
    print(f"Gradient shape: {grad.shape}, norm: {grad.norm().item():.4f}")
    return grad  # Can modify gradient here

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Register hook on the first layer
handle = model[0].weight.register_hook(gradient_hook)

x = torch.randn(32, 10)
y = torch.randn(32, 1)

pred = model(x)
loss = F.mse_loss(pred, y)
loss.backward()

# Remove hook
handle.remove()

# 7. Model Profiling and Debugging
print("\n7. Model Profiling:")
model = nn.Sequential(
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

x = torch.randn(64, 1000)

# Memory profiling
def profile_model_memory():
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
    
    # Forward pass
    output = model(x)
    
    if torch.cuda.is_available():
        forward_memory = torch.cuda.memory_allocated()
        print(f"Memory after forward: {(forward_memory - start_memory) / 1024**2:.2f} MB")
    else:
        print("CUDA not available - memory profiling skipped")

profile_model_memory()

# 8. Dynamic Computation Graphs
print("\n8. Dynamic Computation Graphs:")
class DynamicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        # Dynamic: random number of layers
        for _ in range(torch.randint(1, 4, (1,)).item()):
            x = F.relu(self.fc2(x))
        
        return self.fc3(x)

dynamic_net = DynamicNet()
x = torch.randn(32, 10)

print("Dynamic computation graph outputs:")
for i in range(3):
    output = dynamic_net(x)
    print(f"Run {i+1}: Output shape = {output.shape}, mean = {output.mean().item():.4f}")

# 9. Model Quantization
print("\n9. Model Quantization:")
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# Post-training quantization
model.eval()
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

x = torch.randn(32, 10)
original_output = model(x)
quantized_output = quantized_model(x)

print(f"Original model size: ~{sum(p.numel() * 4 for p in model.parameters()) / 1024:.1f} KB")
print(f"Quantized model parameters: {sum(p.numel() for p in quantized_model.parameters() if hasattr(p, 'numel'))}")
print(f"Output difference: {torch.mean(torch.abs(original_output - quantized_output)).item():.6f}")

# 10. Best Practices Summary
print("\n10. Best Practices Summary:")

best_practices = {
    "Data Loading": "Use DataLoader with num_workers > 0 for faster loading",
    "Memory": "Use torch.cuda.empty_cache() and del unnecessary variables",
    "Gradients": "Always call optimizer.zero_grad() before backward()",
    "Evaluation": "Use model.eval() and torch.no_grad() during inference",
    "Reproducibility": "Set random seeds: torch.manual_seed(42)",
    "Debugging": "Use torch.autograd.set_detect_anomaly(True) for gradient issues",
    "Performance": "Use appropriate data types (float16 for inference if possible)",
    "Initialization": "Use proper weight initialization schemes",
    "Regularization": "Apply dropout, batch norm, weight decay as needed",
    "Monitoring": "Track gradient norms, learning rates, and loss curves"
}

print("PyTorch Best Practices:")
for category, practice in best_practices.items():
    print(f"• {category}: {practice}")

print("\n=== End of Advanced Topics ===")