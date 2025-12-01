# Project 4: Convolutional Neural Network

**Course:** CSCI 5561 - Computer Vision  
**Institution:** University of Minnesota  
**Student:** Apurv Kushwaha  
**Due Date:** December 2, 2025

---

## 📋 Project Overview

Implementation of neural networks for hand-written digit recognition using the MNIST dataset. The project involves building four different neural network architectures from scratch using NumPy.

### Dataset
- **Source:** MNIST hand-written digits (reduced from 28×28 to 14×14)
- **Training Data:** `mnist_train.mat`
- **Testing Data:** `mnist_test.mat`
- **Format:** Vectorized images (196 × n) and labels (1 × n)

---

## 🎯 Implemented Models

### 1. Single-Layer Linear Perceptron
- **Architecture:** Input (196) → FC → Output (10)
- **Loss:** Euclidean distance
- **Target Accuracy:** >25%
- **Achieved Accuracy:** ✓

### 2. Single-Layer Perceptron with Softmax
- **Architecture:** Input (196) → FC → Softmax → Output (10)
- **Loss:** Cross-entropy
- **Target Accuracy:** >85%
- **Achieved Accuracy:** ✓

### 3. Multi-Layer Perceptron
- **Architecture:** Input (196) → FC (30 units) → ReLU → FC → Softmax → Output (10)
- **Loss:** Cross-entropy
- **Target Accuracy:** >90%
- **Achieved Accuracy:** ✓

### 4. Convolutional Neural Network
- **Architecture:**
  - Input (14×14×1)
  - Conv (3×3, 3 channels, stride 1)
  - ReLU
  - MaxPool (2×2, stride 2)
  - Flatten (147 units)
  - FC (10 units)
  - Softmax
- **Target Accuracy:** >92%
- **Status:** In progress

---

## 📁 Project Structure

```
P4:Convolutional Neural Network/
├── p4.py                    # Main implementation file (17 functions)
├── main_functions.py        # Testing and evaluation functions
├── ReducedMNIST/           # Dataset directory
│   ├── mnist_train.mat
│   └── mnist_test.mat
├── images/                 # Confusion matrix visualizations
│   ├── Figure_1.png        # SLP Linear confusion matrix
│   ├── Figure_2.png        # SLP confusion matrix
│   ├── Figure_3.png        # MLP confusion matrix
│   └── Figure_4.png        # CNN confusion matrix
├── slp_linear.npz          # Trained weights - Linear Perceptron
├── slp.npz                 # Trained weights - Softmax Perceptron
├── mlp.npz                 # Trained weights - Multi-Layer Perceptron
├── cnn.npz                 # Trained weights - CNN
└── README.md               # This file
```

---

## 🔧 Implemented Functions

### Data Handling
1. **`get_mini_batch()`** - Create mini-batches with one-hot encoding

### Fully Connected Layer
2. **`fc()`** - Forward pass: y = wx + b
3. **`fc_backward()`** - Backward pass with gradients

### Loss Functions
4. **`loss_euclidean()`** - L2 loss for linear perceptron
5. **`loss_cross_entropy_softmax()`** - Cross-entropy with softmax

### Activation Functions
6. **`relu()`** - ReLU activation (forward)
7. **`relu_backward()`** - ReLU gradient (backward)

### Convolutional Operations
8. **`conv()`** - 2D convolution with padding
9. **`conv_backward()`** - Convolution backward pass
10. **`pool2x2()`** - 2×2 max pooling
11. **`pool2x2_backward()`** - Pooling backward pass
12. **`flattening()`** - Flatten tensor to vector
13. **`flattening_backward()`** - Flattening backward pass

### Training Functions
14. **`train_slp_linear()`** - Train linear perceptron (8000 iterations)
15. **`train_slp()`** - Train softmax perceptron (5000 iterations)
16. **`train_mlp()`** - Train multi-layer perceptron (10000 iterations)
17. **`train_cnn()`** - Train CNN (10000 iterations)

---

## ⚙️ Training Configuration

### Stochastic Gradient Descent Parameters

| Model | Learning Rate | Decay Rate | Iterations | Weight Init |
|-------|--------------|------------|------------|-------------|
| **SLP Linear** | 0.01 | 0.9 | 8,000 | N(0,1) |
| **SLP** | 0.1 | 0.9 | 5,000 | N(0,1) |
| **MLP** | 0.2 | 0.9 | 10,000 | N(0,0.1), zero bias |
| **CNN** | 0.1 | 0.9 | 10,000 | N(0,0.1), zero bias |

- **Batch Size:** 32
- **Decay Schedule:** Every 1000 iterations
- **Update Rule:** `w ← w - (γ/batch_size) * ∂L/∂w`

---

## 🚀 Usage

### Running the Code

```bash
# Run all models
python p4.py

# Or run specific models in main_functions.py
python -c "import main_functions as main; main.main_slp_linear()"
```

### Loading Pre-trained Weights

```python
import main_functions as main

# Load and test with pre-trained weights
main.main_slp_linear(load_weights=True)
main.main_slp(load_weights=True)
main.main_mlp(load_weights=True)
main.main_cnn(load_weights=True)
```

---

## 💻 Development Environment

### Local Development
- **OS:** Ubuntu/macOS
- **Python:** 3.10+
- **Dependencies:** NumPy, Matplotlib, SciPy

### MSI OnDemand (High-Performance Computing)
- **Platform:** Minnesota Supercomputing Institute
- **Resources:** Interactive GPU - 16 cores, 60 GB RAM
- **Environment:** Conda (seam2025)
- **Directory:** `/users/2/kushw022/cv5561-p4/`

### Installation

```bash
# Install dependencies
pip install numpy matplotlib scipy

# Or using conda
conda install numpy matplotlib scipy
```

---

## 📊 Results

### Model Performance

| Model | Test Accuracy | Status |
|-------|--------------|--------|
| **SLP Linear** | ~30% | ✅ Passed |
| **SLP** | ~90% | ✅ Passed |
| **MLP** | ~91%+ | ✅ Passed |
| **CNN** | Target: >92% | 🔄 In Progress |

### Confusion Matrices

#### Single-Layer Linear Perceptron
![SLP Linear Confusion Matrix](images/Figure_1.png)

*Accuracy: ~30% - Random initialization with Euclidean loss*

#### Single-Layer Perceptron with Softmax
![SLP Confusion Matrix](images/Figure_2.png)

*Accuracy: ~90% - Softmax activation with cross-entropy loss*

#### Multi-Layer Perceptron
![MLP Confusion Matrix](images/Figure_3.png)

*Accuracy: ~91% - 30 hidden units with ReLU activation*

#### Convolutional Neural Network
![CNN Confusion Matrix](images/Figure_4.png)

*Target: >92% - 3×3 convolution with max pooling*

### Training Time (on MSI)
- **SLP Linear:** ~10 seconds
- **SLP:** ~10 seconds
- **MLP:** ~20 seconds
- **CNN:** ~30-60 minutes

---

## 🐛 Known Issues & Solutions

### Issue 1: Flattening Function
**Problem:** Autograder expects C-order (row-major) flattening  
**Solution:** Use `x.reshape((-1, 1))` without `order='F'`

### Issue 2: CNN Low Accuracy
**Problem:** Poor weight initialization causing vanishing gradients  
**Solution:** Use `* 0.1` scaling and zero bias initialization

### Issue 3: Slow CNN Training
**Problem:** Triple nested loops in convolution  
**Solution:** Use MSI HPC resources or implement vectorized operations (optional im2col)

---

## 📝 Implementation Notes

### Key Design Decisions

1. **Mini-batch Processing:** Random permutation ensures data diversity
2. **Learning Rate Decay:** Exponential decay every 1000 iterations for stability
3. **Weight Initialization:** Gaussian with small variance prevents saturation
4. **Numerical Stability:** Max subtraction in softmax prevents overflow

### Optimization Techniques

- **Vectorized Operations:** NumPy broadcasting for efficiency
- **Fortran-order Input:** Consistent with MATLAB data format
- **Gradient Accumulation:** Sum gradients over batch before update

---

## 🎓 Learning Outcomes

- ✅ Implemented backpropagation from scratch
- ✅ Understood gradient flow in deep networks
- ✅ Debugged numerical stability issues
- ✅ Optimized hyperparameters empirically
- ✅ Used high-performance computing resources

---

## 📚 References

1. **Course Materials:** CSCI 5561 Lecture Notes
2. **Assignment PDF:** `CSCI5561_P4__1_.pdf`
3. **NumPy Documentation:** https://numpy.org/doc/
4. **Convolution Guide:** [im2col technique](https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster)

---

## 📤 Submission

### Gradescope Deliverables
1. `p4.py` - Complete implementation
2. `slp_linear.npz` - Linear perceptron weights
3. `slp.npz` - Softmax perceptron weights
4. `mlp.npz` - Multi-layer perceptron weights
5. `cnn.npz` - CNN weights

### Autograder Tests
- 16 function tests (0.25-2 points each)
- Total: 10 points
- Passing threshold: Meet accuracy requirements

---

## 🔗 Repository

**GitHub:** https://github.com/ApurvK032/CV-Assignments-and-Projects

```bash
# Clone repository
git clone https://github.com/ApurvK032/CV-Assignments-and-Projects.git

# Navigate to P4
cd "Computer-Vision-CSCI-5561/P4:Convolutional Neural Network"
```

---

## 👤 Author

**Apurv Kushwaha**  
University of Minnesota  
Computer Science & Engineering  

---

## 📅 Timeline

- **Project Assigned:** November 13, 2025
- **Implementation:** November 25-30, 2025
- **Testing & Debugging:** November 29-30, 2025
- **Due Date:** December 2, 2025, 11:59 PM
- **Status:** 14/16 tests passing

---

## 🙏 Acknowledgments

- MSI OnDemand for high-performance computing resources
- CSCI 5561 teaching staff for assignment design
- Claude AI for implementation assistance and debugging support

---

*Last Updated: November 30, 2025*
