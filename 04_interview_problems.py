"""
PyTorch Tutorial - Part 4: Common Interview Problems
Real interview questions with PyTorch implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

print("=== Common PyTorch Interview Problems ===")

# Problem 1: Implement Softmax from scratch
print("\n1. Implement Softmax from Scratch:")
def softmax_from_scratch(x):
    """Numerically stable softmax implementation"""
    exp_x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)

x = torch.randn(3, 5)
our_softmax = softmax_from_scratch(x)
pytorch_softmax = F.softmax(x, dim=-1)

print(f"Our softmax:\n{our_softmax}")
print(f"PyTorch softmax:\n{pytorch_softmax}")
print(f"Difference: {torch.max(torch.abs(our_softmax - pytorch_softmax))}")

# Problem 2: Custom Dataset and DataLoader
print("\n2. Custom Dataset Implementation:")
class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Create sample data
data = torch.randn(100, 10)
targets = torch.randint(0, 3, (100,))

dataset = CustomDataset(data, targets)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

print(f"Dataset length: {len(dataset)}")
print(f"Number of batches: {len(dataloader)}")

for batch_idx, (batch_data, batch_targets) in enumerate(dataloader):
    print(f"Batch {batch_idx}: data shape {batch_data.shape}, targets shape {batch_targets.shape}")
    if batch_idx == 2:  # Show first 3 batches
        break

# Problem 3: Implement Batch Normalization
print("\n3. Batch Normalization from Scratch:")
class CustomBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics (not learnable)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        if self.training:
            # Training mode: use batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            mean, var = batch_mean, batch_var
        else:
            # Evaluation mode: use running statistics
            mean, var = self.running_mean, self.running_var
        
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_normalized + self.bias

# Test custom batch norm
x = torch.randn(32, 10)
custom_bn = CustomBatchNorm1d(10)
pytorch_bn = nn.BatchNorm1d(10)

# Initialize with same parameters
pytorch_bn.weight.data = custom_bn.weight.data.clone()
pytorch_bn.bias.data = custom_bn.bias.data.clone()

custom_output = custom_bn(x)
pytorch_output = pytorch_bn(x)

print(f"Custom BN output mean: {custom_output.mean(dim=0)}")
print(f"PyTorch BN output mean: {pytorch_output.mean(dim=0)}")
print(f"Difference: {torch.max(torch.abs(custom_output - pytorch_output))}")

# Problem 4: Learning Rate Scheduling
print("\n4. Learning Rate Scheduling:")
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Different schedulers
step_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
exp_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

print("Learning rate changes with StepLR:")
for epoch in range(10):
    print(f"Epoch {epoch}: LR = {optimizer.param_groups[0]['lr']:.6f}")
    step_scheduler.step()

# Problem 5: Gradient Clipping
print("\n5. Gradient Clipping Implementation:")
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

x = torch.randn(32, 10)
y = torch.randn(32, 1)

def gradient_clipping_example():
    for epoch in range(3):
        pred = model(x)
        loss = F.mse_loss(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients before clipping
        total_norm_before = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_before += param_norm.item() ** 2
        total_norm_before = total_norm_before ** (1. / 2)
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Check gradients after clipping
        total_norm_after = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_after += param_norm.item() ** 2
        total_norm_after = total_norm_after ** (1. / 2)
        
        print(f"Epoch {epoch}: Grad norm before: {total_norm_before:.4f}, after: {total_norm_after:.4f}")
        optimizer.step()

gradient_clipping_example()

# Problem 6: Multi-GPU Training Setup
print("\n6. Multi-GPU Training Setup:")
if torch.cuda.device_count() > 1:
    print(f"Found {torch.cuda.device_count()} GPUs")
    
    model = nn.Linear(10, 1)
    model = nn.DataParallel(model)  # Wrap model for multi-GPU
    
    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randn(32, 10).cuda()
        y = model(x)
        print(f"Multi-GPU output shape: {y.shape}")
else:
    print("Single GPU or CPU training")

# Problem 7: Custom Loss Function
print("\n7. Custom Loss Function:")
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Test focal loss
pred = torch.randn(10, 1)
target = torch.randint(0, 2, (10, 1)).float()

focal_loss = FocalLoss()
bce_loss = nn.BCEWithLogitsLoss()

focal_result = focal_loss(pred, target)
bce_result = bce_loss(pred, target)

print(f"Focal Loss: {focal_result:.4f}")
print(f"BCE Loss: {bce_result:.4f}")

# Problem 8: Weight Initialization
print("\n8. Weight Initialization Strategies:")
def init_weights(module):
    if isinstance(module, nn.Linear):
        print(f"Initializing Linear layer with shape {module.weight.shape}")
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        print(f"Initializing Conv2d layer with shape {module.weight.shape}")
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
        self.conv = nn.Conv2d(3, 16, 3)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

net = TestNet()
print("Before initialization:")
print(f"fc1 weight mean: {net.fc1.weight.mean():.4f}, std: {net.fc1.weight.std():.4f}")

net.apply(init_weights)
print("\nAfter initialization:")
print(f"fc1 weight mean: {net.fc1.weight.mean():.4f}, std: {net.fc1.weight.std():.4f}")

# Problem 9: Memory Optimization
print("\n9. Memory Optimization Techniques:")
def memory_efficient_training():
    model = nn.Sequential(
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )
    
    x = torch.randn(64, 1000)
    y = torch.randn(64, 1)
    
    optimizer = optim.Adam(model.parameters())
    
    # Technique 1: Gradient accumulation
    accumulation_steps = 4
    
    for step in range(accumulation_steps):
        # Smaller batch
        batch_x = x[step * 16:(step + 1) * 16]
        batch_y = y[step * 16:(step + 1) * 16]
        
        pred = model(batch_x)
        loss = F.mse_loss(pred, batch_y) / accumulation_steps  # Scale loss
        loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    print("Gradient accumulation completed")
    
    # Technique 2: Checkpointing (saves memory during backward pass)
    from torch.utils.checkpoint import checkpoint
    
    def checkpoint_function(x):
        return model(x)
    
    x_checkpoint = torch.randn(32, 1000, requires_grad=True)
    # y_checkpoint = checkpoint(checkpoint_function, x_checkpoint)  # Uncomment if needed
    print("Checkpointing example prepared")

memory_efficient_training()

# Problem 10: Model Ensemble
print("\n10. Model Ensemble:")
class SimpleEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

# Create ensemble of 3 models
models = [nn.Linear(10, 1) for _ in range(3)]
ensemble = SimpleEnsemble(models)

x = torch.randn(32, 10)
ensemble_output = ensemble(x)
print(f"Ensemble output shape: {ensemble_output.shape}")

# Individual model predictions
individual_outputs = [model(x) for model in models]
manual_average = torch.mean(torch.stack(individual_outputs), dim=0)
print(f"Manual average matches ensemble: {torch.allclose(ensemble_output, manual_average)}")

print("\n=== End of Interview Problems ===")