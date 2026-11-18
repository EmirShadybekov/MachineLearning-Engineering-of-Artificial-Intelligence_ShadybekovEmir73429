# Lab 5: Frameworks and Automatic Differentiation

**Course:** Machine Learning [CII5SE039CI], Group 03  
**Institution:** Faculty of Computer Science, Graphics and Architecture, Akademia Finansów i Biznesu Vistula w Warszawie  
**Instructor:** mgr. inż. Kumar Nalinaksh  
**Student:** Emir Shadybekov (Album: 73429)  
**Prescribed Reading:** Vijay Janapa Reddi, *Introduction to Machine Learning Systems: Principles and Practices of Engineering Artificially Intelligent Systems*, Chapter 7 AI Frameworks (§§7.3 Fundamental Concepts, 7.4 Framework Architecture, 7.6 System Integration, 7.7–7.8 Framework Implementation/Deployment).

---

## 1. Goal

The objective of Laboratory 5 is to understand and implement the foundational mechanisms powering modern deep learning frameworks:
1. **Scalar Automatic Differentiation Engine:** Design and construct a lightweight reverse-mode autograd engine using a directed acyclic computation graph (DAG) centered on a `Value` node abstraction.
2. **Topological Gradient Propagation:** Implement topological sorting to execute reverse-mode automatic differentiation with rigorous gradient accumulation (`+=`) across fan-out branches.
3. **Mathematical & Numerical Verification:** Empirically validate the custom engine against PyTorch's native C++ autograd engine and central finite-difference numerical approximations on the composite non-linear expression $((a \cdot b + c)^2).\text{relu}().\text{sum}()$, ensuring maximum absolute error remains below $10^{-6}$.
4. **End-to-End Model Training on `make_moons`:** Train a complete neural network ($2 \to 8 \to 1$, sigmoid activation, Binary Cross-Entropy loss) using pure scalar autograd and compare training throughput and wall-clock convergence speed against equivalent PyTorch tensor execution (median of 5 runs).
5. **Framework Internals, Graph Execution Modes & Memory Contiguity:** Inspect PyTorch's internal computation graph representations (TorchScript IR), quantify eager vs. compiled/traced execution modes, and measure the performance degradation caused by non-contiguous strided memory layouts.

---

## 2. Method

### 2.1 The `Value` Node Architecture and Graph Construction
The scalar autograd engine is constructed around the `Value` class, which encapsulates:
- `data`: Floating-point scalar scalar value computed during forward propagation.
- `grad`: Accumulated gradient $\frac{\partial \mathcal{L}}{\partial \text{self}}$, initialized to $0.0$.
- `_prev`: Set of immediate antecedent `Value` nodes in the DAG.
- `_op`: String descriptor representing the generating primitive operation (e.g., `+`, `*`, `**`, `ReLU`).
- `_backward`: A closure capturing local partial derivatives according to the multivariate chain rule:
  $$\frac{\partial \mathcal{L}}{\partial x_i} = \sum_{j \in \text{children}} \frac{\partial \mathcal{L}}{\partial y_j} \frac{\partial y_j}{\partial x_i}$$

#### Implemented Primitive Operations:
- **Addition ($y = u + v$):**  
  $\frac{\partial y}{\partial u} = 1.0, \quad \frac{\partial y}{\partial v} = 1.0 \implies u.\text{grad} \mathrel{+}= 1.0 \times y.\text{grad}$
- **Multiplication ($y = u \cdot v$):**  
  $\frac{\partial y}{\partial u} = v, \quad \frac{\partial y}{\partial v} = u \implies u.\text{grad} \mathrel{+}= v \times y.\text{grad}, \quad v.\text{grad} \mathrel{+}= u \times y.\text{grad}$
- **Power ($y = u^n$):**  
  $\frac{\partial y}{\partial u} = n \cdot u^{n - 1} \implies u.\text{grad} \mathrel{+}= (n \cdot u^{n - 1}) \times y.\text{grad}$
- **Rectified Linear Unit ($y = \max(0, u)$):**  
  $\frac{\partial y}{\partial u} = \mathbb{I}(u > 0) \implies u.\text{grad} \mathrel{+}= (1.0 \text{ if } u > 0 \text{ else } 0.0) \times y.\text{grad}$
- **Sigmoid ($y = \sigma(u) = \frac{1}{1 + e^{-u}}$):**  
  $\frac{\partial y}{\partial u} = y(1 - y) \implies u.\text{grad} \mathrel{+}= y(1 - y) \times y.\text{grad}$
- **Logarithm ($y = \ln(u)$):**  
  $\frac{\partial y}{\partial u} = \frac{1}{u} \implies u.\text{grad} \mathrel{+}= \frac{1}{u} \times y.\text{grad}$

### 2.2 Topological Backward Propagation Pass
Unlike eager evaluation which traverses from inputs to outputs, backpropagation requires visiting nodes in reverse dependency order. The `backward()` method:
1. Performs a post-order Depth-First Search (DFS) traversal to construct a topological ordering of the DAG.
2. Initializes the scalar output node's seed gradient to $1.0$ ($\frac{\partial \text{out}}{\partial \text{out}} = 1.0$).
3. Iterates backwards across the reversed topological sequence, invoking `_backward()` on each node.
4. **Gradient Accumulation:** Uses `+=` instead of assignment (`=`) so that variables used multiple times across different computational paths correctly accumulate partial derivatives without overwrite corruption.

### 2.3 Verification Protocol
To prove mathematical precision, the expression $\mathcal{F} = \sum_{i=1}^5 \max(0, (a_i \cdot b_i + c_i)^2)$ was evaluated across heterogeneous values:
- **PyTorch Autograd:** Evaluated with `torch.float64` double precision to eliminate 32-bit floating-point rounding bias.
- **Central Finite Difference:** Computed with step size $\epsilon = 10^{-6}$:
  $$g_{\text{num}}(\theta) = \frac{\mathcal{F}(\theta + \epsilon) - \mathcal{F}(\theta - \epsilon)}{2\epsilon}$$
- **Pass Threshold:** $\max |g_{\text{custom}} - g_{\text{torch}}| < 10^{-6}$ and $\max |g_{\text{custom}} - g_{\text{num}}| < 10^{-6}$.

### 2.4 End-to-End Model Training on `make_moons`
- **Dataset:** `sklearn.datasets.make_moons(n_samples=1000, noise=0.1, random_state=42)`.
- **Architecture:** $2 \text{ inputs} \to 8 \text{ hidden (ReLU)} \to 1 \text{ output (Sigmoid)}$.
- **Loss:** Binary Cross-Entropy (BCE): $\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N [y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i)]$.
- **Training Protocol:** 25 epochs, mini-batch size 64, learning rate $\eta = 0.1$, 1 warmup run followed by 5 timed repetitions to report median execution time.

### 2.5 PyTorch Graph Inspection, Execution Modes & Memory Contiguity
- **Graph Inspection:** The model computation graph was captured and exported to TorchScript IR via `torch.jit.trace`.
- **Execution Modes:** Eager mode was benchmarked against traced (`torch.jit.trace`) and scripted (`torch.jit.script`) execution over 100 runs after 10 warmup runs.
- **Memory Contiguity Penalty:** A $2048 \times 2048$ tensor (float32, 16 MB) was compared against its non-contiguous transposed view (`stride = (1, 2048)`) under a row-wise reduction operation (`sum(dim=1)`), measuring the hardware cache penalty of non-contiguous memory layouts over 100 runs.

---

## 3. Results Table and Empirical Evidence

### 3.1 Task 4: Autograd Engine Verification Table
Evaluated on $\mathcal{F} = \sum ((a \cdot b + c)^2).\text{relu}()$ across 15 scalar variables:

| Variable | Input Value | Custom Engine Grad | PyTorch Autograd Grad | Numerical Grad ($\epsilon=10^{-6}$) | \|Custom - PyTorch\| | \|Custom - Numerical\| | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $a[0]$ | -2.00 | -30.00000000 | -30.00000000 | -30.00000000 | $0.00\times 10^0$ | $2.42\times 10^{-9}$ | **PASS** |
| $a[1]$ | 1.50 | 10.00000000 | 10.00000000 | 10.00000000 | $0.00\times 10^0$ | $3.93\times 10^{-9}$ | **PASS** |
| $a[2]$ | 0.50 | -8.00000000 | -8.00000000 | -8.00000000 | $0.00\times 10^0$ | $1.12\times 10^{-9}$ | **PASS** |
| $a[3]$ | -1.00 | 2.00000000 | 2.00000000 | 2.00000000 | $0.00\times 10^0$ | $2.06\times 10^{-9}$ | **PASS** |
| $a[4]$ | 2.50 | -0.25000000 | -0.25000000 | -0.25000000 | $0.00\times 10^0$ | $6.31\times 10^{-10}$ | **PASS** |
| $b[0]$ | 3.00 | 20.00000000 | 20.00000000 | 20.00000000 | $0.00\times 10^0$ | $7.57\times 10^{-10}$ | **PASS** |
| $b[1]$ | -2.00 | -7.50000000 | -7.50000000 | -7.50000000 | $0.00\times 10^0$ | $2.38\times 10^{-9}$ | **PASS** |
| $b[2]$ | 4.00 | -1.00000000 | -1.00000000 | -1.00000000 | $0.00\times 10^0$ | $1.03\times 10^{-9}$ | **PASS** |
| $b[3]$ | 1.00 | -2.00000000 | -2.00000000 | -2.00000000 | $0.00\times 10^0$ | $2.06\times 10^{-9}$ | **PASS** |
| $b[4]$ | -0.50 | 1.25000000 | 1.25000000 | 1.25000000 | $0.00\times 10^0$ | $3.97\times 10^{-10}$ | **PASS** |
| $c[0]$ | 1.00 | -10.00000000 | -10.00000000 | -10.00000000 | $0.00\times 10^0$ | $3.79\times 10^{-10}$ | **PASS** |
| $c[1]$ | 0.50 | -5.00000000 | -5.00000000 | -5.00000000 | $0.00\times 10^0$ | $1.97\times 10^{-9}$ | **PASS** |
| $c[2]$ | -3.00 | -2.00000000 | -2.00000000 | -2.00000000 | $0.00\times 10^0$ | $2.06\times 10^{-9}$ | **PASS** |
| $c[3]$ | 2.00 | 2.00000000 | 2.00000000 | 2.00000000 | $0.00\times 10^0$ | $2.06\times 10^{-9}$ | **PASS** |
| $c[4]$ | 1.50 | 0.50000000 | 0.50000000 | 0.50000000 | $0.00\times 10^0$ | $1.26\times 10^{-9}$ | **PASS** |

- **Forward Output:** $\text{Custom} = 33.31250000, \quad \text{PyTorch} = 33.31250000$ (Exact match)
- **Max Absolute Difference vs PyTorch:** $\mathbf{0.00 \times 10^0} < 10^{-6}$
- **Max Absolute Difference vs Numerical:** $\mathbf{3.93 \times 10^{-9}} < 10^{-6}$
- **Outcome:** **15 / 15 gradients passed with zero discrepancy.**

*(Saved in `results/autograd_verification.csv`)*

---

### 3.2 Task 5: Tiny Model Training Benchmark on `make_moons`
Trained for 25 epochs (mini-batch 64, learning rate 0.1) on 1,000 moon samples:

| Framework / Engine | Model Architecture | Final BCE Loss | Final Train Accuracy | Median Training Time (5 runs) | Speedup vs Scalar Engine |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Custom Scalar Autograd (`Value`)** | $2 \to 8 \to 1$ (Sigmoid) | **0.2554** | **87.80%** | 12.6157 s | $1.00\times$ (Baseline) |
| **PyTorch (Tensor C++ Core)** | $2 \to 8 \to 1$ (Sigmoid) | **0.2557** | **87.80%** | **0.1048 s** | **$120.43\times$ faster** |

*(Saved in `results/tiny_model_benchmark.csv` and visualised in `results/decision_boundary.png`)*

Both models converged to the exact same classification boundary and identical $87.80\%$ training accuracy. However, PyTorch executed over **$120\times$ faster**, illustrating the vast efficiency gap between interpreted scalar DAG node traversal and vectorized C++/SIMD tensor kernels.

---

### 3.3 Task 6: Execution Modes & Contiguity Benchmarks

#### A. Execution Mode Latencies (Batch Size 128, Median of 100 Runs):
| Execution Mode | Backend / Implementation | Median Latency (ms) | Speedup vs Eager |
|:---|:---|:---:|:---:|
| **PyTorch Eager** | Dynamic dispatch per operator | 0.0223 ms | $1.00\times$ (Baseline) |
| **TorchScript (JIT Trace)** | Static intermediate graph IR | **0.0172 ms** | **$1.30\times$ faster** |
| **TorchScript (JIT Script)** | Bytecode compiler | **0.0163 ms** | **$1.37\times$ faster** |

*(Saved in `results/eager_vs_compiled.csv`)*

#### B. Tensor Representation & Contiguity Penalty (2048 x 2048 Matrix, Median of 100 Runs):
| Tensor Layout | Memory Strides | `is_contiguous()` | Operation | Median Latency (ms) | Performance Impact |
|:---|:---:|:---:|:---:|:---:|:---|
| **Contiguous (Row-major)** | $(2048, 1)$ | `True` | `sum(dim=1)` | **0.2961 ms** | **Baseline (Optimal cache locality)** |
| **Non-Contiguous (Transposed)** | $(1, 2048)$ | `False` | `sum(dim=1)` | **1.0133 ms** | **$3.42\times$ SLOWER (Cache miss penalty)** |

*(Saved in `results/contiguity_benchmark.csv` and visualised in `results/contiguity_benchmark.png`)*

---

### 3.4 Short Note on When Compilation Helps (Reddi §7.7–7.8)

In modern machine learning systems engineering, graph compilation (via `torch.compile`, TorchScript, or XLA) provides substantial performance gains under specific architectural conditions:

1. **Kernel Fusion (Eliminating Memory Bandwidth Bottlenecks):**  
   In eager execution, every mathematical operator launches a discrete kernel that reads inputs from DRAM and writes results back to DRAM. Graph compilers perform **vertical and horizontal kernel fusion**, chaining element-wise operations (e.g., $\text{Linear} + \text{Bias} + \text{ReLU}$) into a single fused GPU/CPU kernel. This keeps intermediate activations inside high-speed SRAM registers or L1 cache, drastically reducing memory bus traffic.
2. **Elimination of Host-Side Runtime Overhead:**  
   Python interpreter overhead and dynamic framework dispatch checks (shape validation, type checking, auto-differentiation tape recording) occur on every eager call. Graph compilation eliminates host dispatch overhead by compiling the static DAG into optimized native machine code.
3. **When Compilation Does NOT Help:**  
   - Highly dynamic control flow where tensor shapes, rank, or execution branches change on every iteration (triggering constant recompilation graph-breaks).
   - Compute-bound monolithic operations (e.g., huge matrix multiplications $4096 \times 4096$) where execution time is already $99\%$ dominated by highly optimized BLAS/cuBLAS routines, leaving negligible framework overhead to optimize.
   - Tiny models on CPU where compilation warm-up latency far exceeds any runtime execution savings.

---

## 4. Three Conclusions in Your Own Words

1. **Vectorized Tensor Abstractions are Essential for Computational Efficiency:**  
   While our scalar `Value` autograd engine successfully achieved mathematical parity with PyTorch (converging to the exact same $87.80\%$ accuracy and $0.2554$ BCE loss on `make_moons`), it was **$120.4\times$ slower** than PyTorch. In our scalar engine, each connection requires individual Python heap allocations, reference counting, and dynamic pointer dereferencing during backward traversal. Modern AI frameworks achieve high performance by grouping numbers into contiguous multi-dimensional tensors, enabling SIMD vectorization (AVX-512) and parallel cache-tiled matrix multiplication kernels.

2. **Physical Memory Layout (Contiguity) Dominates Operation Latency:**  
   Our tensor memory experiments revealed that performing a row-wise reduction (`sum(dim=1)`) on a non-contiguous transposed matrix was **$3.42\times$ slower** ($1.013\text{ ms}$ vs $0.296\text{ ms}$) than on an identical contiguous matrix. Even though both tensors contain the exact same $4,194,304$ floats and perform the exact same number of arithmetic additions, strided column-major memory access breaks CPU spatial locality, causing L1/L2 cache misses on consecutive element reads. High-performance ML systems must enforce tensor contiguity before executing memory-bound operators.

3. **Reverse-Mode Autograd Derivation is General and Dimension-Agnostic:**  
   Implementing automatic differentiation from first principles demonstrated that reverse-mode backpropagation is fundamentally independent of network complexity. By maintaining an explicit computational DAG and enforcing topological sorting with additive gradient accumulation (`+=`), complex non-linear networks naturally compute exact analytical gradients matching PyTorch to machine precision ($0.00\times 10^0$ error vs. PyTorch and $3.93\times 10^{-9}$ vs. finite differences). This proves that backpropagation is simply reverse-order chain-rule accumulation across a directed graph.
