# PyTorch Learning Tutorial

A comprehensive PyTorch tutorial designed for learning and interview preparation.

## Overview

This tutorial covers PyTorch fundamentals through advanced topics commonly asked in machine learning engineer interviews.

## Tutorial Structure

### Part 1: Tensor Basics (`01_tensor_basics.py`)
- Creating and manipulating tensors
- Basic operations and broadcasting
- Indexing and slicing
- NumPy integration
- GPU operations
- Essential tensor operations for interviews

### Part 2: Autograd and Gradients (`02_autograd_gradients.py`)
- Automatic differentiation
- Gradient computation
- Higher-order gradients
- Gradient flow control
- Custom gradient functions
- Common gradient scenarios in interviews

### Part 3: Neural Networks (`03_neural_networks.py`)
- Building neural networks with `nn.Module`
- Activation functions and loss functions
- Training loops and optimization
- Model saving/loading
- Regularization techniques (Dropout, BatchNorm)
- Complete XOR problem implementation

### Part 4: Interview Problems (`04_interview_problems.py`)
- Softmax implementation from scratch
- Custom Dataset and DataLoader
- Batch Normalization implementation
- Learning rate scheduling
- Gradient clipping
- Multi-GPU training setup
- Custom loss functions
- Weight initialization strategies
- Memory optimization techniques
- Model ensembles

### Part 5: Advanced Topics (`05_advanced_topics.py`)
- Custom optimizers
- Attention mechanisms (Multi-Head Attention)
- Residual connections and Layer Normalization
- Learning rate warmup and cosine annealing
- Mixed precision training
- Model profiling and debugging
- Dynamic computation graphs
- Model quantization
- Best practices summary

## Quick Start

### Option 1: Run Individual Parts
```bash
python 01_tensor_basics.py
python 02_autograd_gradients.py
python 03_neural_networks.py
python 04_interview_problems.py
python 05_advanced_topics.py
```

### Option 2: Use the Tutorial Runner
```bash
python run_tutorial.py
```

The runner provides an interactive menu to:
- Run all tutorials sequentially
- Run individual tutorial parts

## Requirements

```bash
pip install torch torchvision numpy matplotlib
```

For CUDA support (optional):
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Interview Preparation Focus

This tutorial emphasizes concepts frequently tested in ML engineering interviews:

**Fundamental Concepts:**
- Tensor operations and broadcasting
- Gradient computation and backpropagation
- Neural network architecture design

**Implementation Skills:**
- Custom layers and loss functions
- Training loops and optimization
- Data loading and preprocessing

**Advanced Topics:**
- Attention mechanisms
- Mixed precision training
- Model optimization and quantization
- Memory management

**Best Practices:**
- Code organization with `nn.Module`
- Proper gradient handling
- Efficient data loading
- Model evaluation patterns

## Key Learning Outcomes

After completing this tutorial, you should be able to:

1. **Manipulate tensors** efficiently and understand broadcasting
2. **Implement neural networks** from scratch using PyTorch primitives
3. **Debug gradient flow** and understand autograd mechanics
4. **Optimize training** with proper learning rate scheduling and regularization
5. **Handle advanced scenarios** like multi-GPU training and mixed precision
6. **Answer common interview questions** with practical implementations

## Tips for Interview Success

1. **Practice implementations**: Don't just read the code—type it out and experiment
2. **Understand the math**: Know why operations work, not just how to use them
3. **Optimize for readability**: In interviews, clean code is as important as correct code
4. **Know the trade-offs**: Be prepared to discuss memory vs. speed, accuracy vs. efficiency
5. **Stay current**: PyTorch evolves rapidly; be aware of newer features and best practices

## Common Interview Topics Covered

- ✅ Tensor operations and broadcasting rules
- ✅ Gradient computation and backpropagation
- ✅ Custom loss functions and optimizers
- ✅ Batch normalization and layer normalization
- ✅ Attention mechanisms and transformers
- ✅ Model quantization and optimization
- ✅ Multi-GPU training strategies
- ✅ Memory optimization techniques
- ✅ Debugging and profiling models

## Next Steps

After mastering these concepts:
1. Implement larger projects (image classification, NLP tasks)
2. Study specific architectures (ResNet, Transformer, etc.)
3. Practice on real datasets (ImageNet, GLUE, etc.)
4. Contribute to open-source PyTorch projects
5. Stay updated with PyTorch documentation and releases

---

*This tutorial is designed to provide a solid foundation for both learning PyTorch and succeeding in machine learning engineer interviews.*