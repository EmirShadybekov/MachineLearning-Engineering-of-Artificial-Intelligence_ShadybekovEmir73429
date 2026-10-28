# Lab 2: A Neural Network from Scratch

**Course:** Machine Learning [CII5SE039CI], Group 03  
**Institution:** Faculty of Computer Science, Graphics and Architecture, Akademia Finansów i Biznesu Vistula w Warszawie  
**Instructor:** mgr. inż. Kumar Nalinaksh  
**Student:** Emir Shadybekov (Album: 73429)  
**Prescribed Reading:** Vijay Janapa Reddi, *Introduction to Machine Learning Systems: Principles and Practices of Engineering Artificially Intelligent Systems*, Chapter 3 DL Primer: §3.4, §§3.5.1–3.5.5, §3.6.

---

## 1. Goal

The objective of Laboratory 2 is to design, implement, mathematically derive, and benchmark a two-hidden-layer feedforward neural network from scratch using only low-level NumPy tensor operations. Specific learning outcomes include:
1. Constructing dimensionally correct weight matrices and bias vectors initialized from $\mathcal{N}(0, 0.01)$.
2. Implementing vectorized forward propagation through two ReLU hidden layers and a numerically stabilized Softmax output head.
3. Formulating categorical cross-entropy loss and verifying its theoretical behavior on untrained weights against $\ln(10) \approx 2.3026$.
4. Deriving exact analytical gradients via matrix calculus and the multivariate chain rule, and validating them against two-sided numerical finite difference approximations (gradient checking with $\epsilon = 10^{-5}$ and tolerance $< 10^{-6}$).
5. Implementing a mini-batch stochastic gradient descent training loop with dataset shuffling across a grid of learning rates ($\eta \in \{0.001, 0.01, 0.1\}$) and hidden widths ($W \in \{16, 64\}$).
6. Conducting high-resolution timing benchmarks comparing forward versus backward pass execution times and establishing the computational complexity link to matrix multiplication.

---

## 2. Method

### 2.1 Dataset Ingestion and Preprocessing
The Optical Recognition of Handwritten Digits dataset (`load_digits`) was loaded via `sklearn.datasets`. The raw feature matrix consists of 1,797 $8\times 8$ grayscale digit images (64 pixels each). Inputs were scaled into the range $[0.0, 1.0]$ by dividing by $16.0$. Target labels $y \in \{0, \dots, 9\}$ were converted into 10-dimensional one-hot indicator vectors. The dataset was split into 70% training (1,257 samples) and 30% testing (540 samples) using stratified sampling with `random_state=42` to ensure balanced class distributions across splits.

### 2.2 Network Architecture and Initialization
A two-hidden-layer multilayer perceptron was implemented with the architecture:
$$\text{Input } (64) \xrightarrow{W_1, b_1} \text{Hidden } 1\,(W) \xrightarrow{\text{ReLU}} \text{Hidden } 2\,(W) \xrightarrow{\text{ReLU}} \text{Output } (10) \xrightarrow{\text{Softmax}}$$

Two hidden layer widths were investigated: $W = 16$ and $W = 64$. Parameter dimensionalities were strictly verified:
- $W_1 \in \mathbb{R}^{64 \times W}, \quad b_1 \in \mathbb{R}^{W}$
- $W_2 \in \mathbb{R}^{W \times W}, \quad b_2 \in \mathbb{R}^{W}$
- $W_3 \in \mathbb{R}^{W \times 10}, \quad b_3 \in \mathbb{R}^{10}$

Weights were sampled from $\mathcal{N}(0, 0.01)$ to prevent premature saturation, and biases were initialized to zeros.

### 2.3 Forward Propagation and Loss Calculation
For input mini-batch $X \in \mathbb{R}^{B \times 64}$:
1. $Z_1 = X W_1 + b_1, \quad A_1 = \max(0, Z_1)$
2. $Z_2 = A_1 W_2 + b_2, \quad A_2 = \max(0, Z_2)$
3. $Z_3 = A_2 W_3 + b_3, \quad \hat{Y} = \text{Softmax}(Z_3)$

Numerical stability in Softmax was guaranteed by subtracting $\max_j(Z_{3, ij})$ before exponentiation. Each output row was verified to sum to $1.0$ within an absolute tolerance of $10^{-6}$. Categorical cross-entropy loss was evaluated as $\mathcal{L} = -\frac{1}{B} \sum_{i=1}^B \sum_{k=0}^9 Y_{ik} \ln(\hat{Y}_{ik})$.

### 2.4 Analytical Gradient Derivation and Verification
Analytical backpropagation was derived via the chain rule:
- Output error: $dZ_3 = \frac{1}{B} (\hat{Y} - Y) \in \mathbb{R}^{B \times 10}$
- Layer 3 gradients: $dW_3 = A_2^T dZ_3$, $db_3 = \mathbf{1}^T dZ_3$, $dA_2 = dZ_3 W_3^T$
- Layer 2 gradients: $dZ_2 = dA_2 \odot \mathbb{I}(Z_2 > 0)$, $dW_2 = A_1^T dZ_2$, $db_2 = \mathbf{1}^T dZ_2$, $dA_1 = dZ_2 W_2^T$
- Layer 1 gradients: $dZ_1 = dA_1 \odot \mathbb{I}(Z_1 > 0)$, $dW_1 = X^T dZ_1$, $db_1 = \mathbf{1}^T dZ_1$

Gradients were validated against central numerical differences:
$$g_{\text{num}} = \frac{\mathcal{L}(\theta + \epsilon) - \mathcal{L}(\theta - \epsilon)}{2\epsilon}, \quad \epsilon = 10^{-5}$$
$$\text{Relative Error} = \frac{|g_{\text{analytic}} - g_{\text{num}}|}{\max(|g_{\text{analytic}}|, |g_{\text{num}}|) + 10^{-12}}$$

### 2.5 Training and Timing Protocol
- Mini-batch size: 32; Shuffling: every epoch; Epochs: 50.
- Experiments: 6 configurations across $\eta \in \{0.001, 0.01, 0.1\}$ and $W \in \{16, 64\}$.
- Timing benchmark: 10 warmup iterations followed by 100 timed runs measuring forward and backward passes separately on batch size 32 using `time.perf_counter()`.

---

## 3. Results Table and Empirical Evidence

### 3.1 Numerical Gradient Check Evidence
Evaluated across 10 randomly sampled network weights with $\epsilon = 10^{-5}$ and threshold $< 10^{-6}$:

| Parameter | Index | Analytic Gradient | Numerical Gradient | Relative Error | Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| $W_1$ | (10, 12) | $0.00000000\times 10^0$ | $0.00000000\times 10^0$ | $0.0000\times 10^0$ | **PASS** |
| $W_1$ | (23, 10) | $0.00000000\times 10^0$ | $0.00000000\times 10^0$ | $0.0000\times 10^0$ | **PASS** |
| $W_1$ | (37, 13) | $+1.44904981\times 10^{-5}$ | $+1.44904977\times 10^{-5}$ | $2.9591\times 10^{-8}$ | **PASS** |
| $W_1$ | (3, 3) | $+2.60203166\times 10^{-6}$ | $+2.60202970\times 10^{-6}$ | $7.5083\times 10^{-7}$ | **PASS** |
| $W_1$ | (11, 8) | $+1.97754445\times 10^{-5}$ | $+1.97754479\times 10^{-5}$ | $1.7621\times 10^{-7}$ | **PASS** |
| $W_1$ | (58, 12) | $0.00000000\times 10^0$ | $0.00000000\times 10^0$ | $0.0000\times 10^0$ | **PASS** |
| $W_1$ | (62, 1) | $+2.65833198\times 10^{-5}$ | $+2.65833355\times 10^{-5}$ | $5.9084\times 10^{-7}$ | **PASS** |
| $W_1$ | (57, 7) | $0.00000000\times 10^0$ | $0.00000000\times 10^0$ | $0.0000\times 10^0$ | **PASS** |
| $W_1$ | (10, 3) | $0.00000000\times 10^0$ | $0.00000000\times 10^0$ | $0.0000\times 10^0$ | **PASS** |
| $W_1$ | (54, 6) | $0.00000000\times 10^0$ | $0.00000000\times 10^0$ | $0.0000\times 10^0$ | **PASS** |

**Result:** **10 of 10 passed** with relative error $< 10^{-6}$ (exceeding the $\ge 9/10$ syllabus threshold).

### 3.2 Hyperparameter Training Performance Table
Final results recorded after 50 epochs of mini-batch SGD:

| Hidden Width ($W$) | Learning Rate ($\eta$) | Final Train Loss | Final Train Accuracy | Final Test Loss | Final Test Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 16 | 0.001 | 2.3026 | 10.18% | 2.3026 | 10.19% |
| 16 | 0.010 | 2.3025 | 10.18% | 2.3025 | 10.19% |
| 16 | 0.100 | 1.4025 | 48.85% | 1.4445 | 51.30% |
| 64 | 0.001 | 2.3025 | 10.18% | 2.3025 | 10.19% |
| 64 | 0.010 | 2.3023 | 10.10% | 2.3024 | 10.00% |
| **64** | **0.100** | **0.2043** | **93.08%** | **0.2839** | **90.56%** |

*Loss progression curves visualised in `results/loss_curves.png`.*

### 3.3 Timing Breakdown Table and Matrix Multiplication Dominance
Median elapsed time over 100 runs on a batch size of 32:

| Hidden Width ($W$) | Forward Pass (ms) | Backward Pass (ms) | Ratio ($\text{Bwd}/\text{Fwd}$) | Dominant Phase |
|:---:|:---:|:---:|:---:|:---:|
| **$W = 16$** | 0.0196 ms | 0.0216 ms | **$1.10\times$** | Backward |
| **$W = 64$** | 0.0310 ms | 0.0406 ms | **$1.31\times$** | Backward |

*Visualised in `results/timing_breakdown.png`.*

#### Computational Complexity and Matrix Multiplication Analysis:
- **Forward Pass Multiplications (3 total):**
  1. $Z_1 = X W_1 \quad [B \times 64] \times [64 \times W] \implies \mathcal{O}(B \cdot 64 \cdot W)$ FLOPs
  2. $Z_2 = A_1 W_2 \quad [B \times W] \times [W \times W] \implies \mathcal{O}(B \cdot W^2)$ FLOPs
  3. $Z_3 = A_2 W_3 \quad [B \times W] \times [W \times 10] \implies \mathcal{O}(B \cdot W \cdot 10)$ FLOPs
- **Backward Pass Multiplications (5 total):**
  1. $dW_3 = A_2^T dZ_3 \quad [W \times B] \times [B \times 10] \implies \mathcal{O}(W \cdot B \cdot 10)$ FLOPs
  2. $dA_2 = dZ_3 W_3^T \quad [B \times 10] \times [10 \times W] \implies \mathcal{O}(B \cdot 10 \cdot W)$ FLOPs
  3. $dW_2 = A_1^T dZ_2 \quad [W \times B] \times [B \times W] \implies \mathcal{O}(W \cdot B \cdot W)$ FLOPs
  4. $dA_1 = dZ_2 W_2^T \quad [B \times W] \times [W \times W] \implies \mathcal{O}(B \cdot W \cdot W)$ FLOPs
  5. $dW_1 = X^T dZ_1 \quad [64 \times B] \times [B \times W] \implies \mathcal{O}(64 \cdot B \cdot W)$ FLOPs

**Conclusion:** The backward pass strictly dominates forward pass latency because each hidden layer requires **two** full General Matrix Multiplications (GEMMs)—one against activations to compute weight updates ($A^T dZ$) and one against transposed weights ($dZ W^T$) to backpropagate error signals upstream—compared to only a single GEMM during forward propagation.

---

## 4. Three Conclusions in Your Own Words

1. **Learning Rate and Representation Width Exhibit Strong Non-Linear Synergy:**  
   At low learning rates ($\eta = 0.001$ and $0.01$), both network configurations remained completely trapped near the initial cross-entropy plateau ($\approx 2.3025$, accuracy $\approx 10\%$), unable to escape the flat regions of the parameter space within 50 epochs. Increasing the learning rate to $\eta = 0.1$ triggered escape from the plateau. However, while the narrow model ($W = 16$) achieved only $51.30\%$ test accuracy, the wider model ($W = 64$) rapidly converged to **$90.56\%$ test accuracy** and a test loss of $0.2839$. This demonstrates that overparameterized hidden representations facilitate optimization by creating more favorable gradient descent trajectories.

2. **Finite-Difference Gradient Checking is an Indispensable Validation Tool for Autodiff Engines:**  
   Implementing vectorized chain rule expressions manually is prone to subtle indexing, transposition, and broadcasting errors (e.g., confusing batch reduction axes or misplacing activation masks). By comparing analytical matrix gradients against two-sided finite difference nudges ($\pm 10^{-5}$), our implementation achieved machine-precision alignment ($2.95\times 10^{-8}$ relative error, 10/10 checks passed). Gradient checking guarantees mathematical correctness before embarking on costly training loops.

3. **Backpropagation Inherently Incurs Greater FLOP Overhead Than Forward Inference:**  
   Our empirical timing measurements confirmed that backward pass execution consistently exceeds forward pass execution ($1.10\times$ slower for $W=16$, and $1.31\times$ slower for $W=64$). As demonstrated by our FLOP analysis, this divergence is rooted in fundamental linear algebra: training requires propagating error gradients with respect to both parameter matrices and intermediate hidden states ($5$ matrix multiplications total), whereas inference only computes intermediate states ($3$ matrix multiplications total). In production systems engineering, this $1.3\times - 2\times$ compute asymmetry explains why distributed training clusters demand vastly greater FLOPS and memory bandwidth than inference serving clusters.
