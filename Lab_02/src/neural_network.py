"""
Lab 2: A Neural Network from Scratch
Course: Machine Learning [CII5SE039CI], Group 03
Institution: Akademia Finansów i Biznesu Vistula w Warszawie
Author: Emir Shadybekov (Album: 73429)

Implementation of a two-hidden-layer neural network from scratch using NumPy.
Architecture: Input (64) -> Hidden 1 (W) -> ReLU -> Hidden 2 (W) -> ReLU -> Output (10) -> Softmax
"""

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    """Element-wise Rectified Linear Unit activation."""
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Element-wise derivative of ReLU activation: 1 if z > 0 else 0."""
    return (z > 0.0).astype(float)


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable Softmax activation.
    Subtracts row-wise maximum before exponentiation to prevent overflow.
    """
    shift_z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shift_z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-15) -> float:
    """
    Multi-class Categorical Cross-Entropy Loss:
    L = - (1 / N) * sum( sum( y_true * ln(y_pred) ) )
    
    Parameters:
    - y_pred: Predicted class probabilities (batch_size, num_classes)
    - y_true: One-hot encoded ground truth labels (batch_size, num_classes)
    - eps: Small epsilon to prevent log(0)
    """
    clipped_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -float(np.mean(np.sum(y_true * np.log(clipped_pred), axis=1)))


class TwoHiddenLayerNN:
    """
    Two-Hidden-Layer Fully Connected Neural Network:
    Input (64) -> Hidden 1 (W) -> ReLU -> Hidden 2 (W) -> ReLU -> Output (10) -> Softmax
    
    Parameters:
    - input_dim: int, dimension of input features (default: 64)
    - hidden_dim: int, width W of each hidden layer (e.g. 16 or 64)
    - output_dim: int, number of output classes (default: 10)
    - weight_scale: float, standard deviation for weight initialization N(0, 0.01)
    - random_state: optional int for reproducible weight initialization
    """
    def __init__(self, input_dim: int = 64, hidden_dim: int = 16, output_dim: int = 10,
                 weight_scale: float = 0.01, random_state: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.weight_scale = weight_scale
        
        rng = np.random.default_rng(random_state)
        
        # Step 2: Weight shapes: (64, W), (W, W), (W, 10). Initialised from N(0, 0.01)
        self.W1 = rng.normal(loc=0.0, scale=weight_scale, size=(input_dim, hidden_dim))
        self.W2 = rng.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, hidden_dim))
        self.W3 = rng.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, output_dim))
        
        # Step 2: Bias shapes: (W,), (W,), (10,). Initialised to zeros
        self.b1 = np.zeros(hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.b3 = np.zeros(output_dim)
        
        # Cache for intermediate values during forward pass
        self.cache = {}
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Step 3: Forward pass layer by layer.
        
        Layer 1:
            Z1 = X @ W1 + b1
            A1 = ReLU(Z1)
        Layer 2:
            Z2 = A1 @ W2 + b2
            A2 = ReLU(Z2)
        Layer 3:
            Z3 = A2 @ W3 + b3
            A3 = Softmax(Z3)
        """
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        
        Z2 = A1 @ self.W2 + self.b2
        A2 = relu(Z2)
        
        Z3 = A2 @ self.W3 + self.b3
        A3 = softmax(Z3)
        
        # Save cache for backpropagation
        self.cache = {
            'X': X,
            'Z1': Z1, 'A1': A1,
            'Z2': Z2, 'A2': A2,
            'Z3': Z3, 'A3': A3
        }
        return A3

    def backward(self, Y: np.ndarray) -> dict:
        """
        Step 5: Analytical gradients via the chain rule.
        
        Derivations:
            L = - (1/N) * sum_i sum_k Y_{ik} ln(A3_{ik})
            dZ3 = (A3 - Y) / N                           shape: (N, 10)
            dW3 = A2.T @ dZ3                             shape: (W, 10)
            db3 = sum_i dZ3_{i, :}                       shape: (10,)
            
            dA2 = dZ3 @ W3.T                             shape: (N, W)
            dZ2 = dA2 * (Z2 > 0)                         shape: (N, W)
            dW2 = A1.T @ dZ2                             shape: (W, W)
            db2 = sum_i dZ2_{i, :}                       shape: (W,)
            
            dA1 = dZ2 @ W2.T                             shape: (N, W)
            dZ1 = dA1 * (Z1 > 0)                         shape: (N, W)
            dW1 = X.T @ dZ1                              shape: (64, W)
            db1 = sum_i dZ1_{i, :}                       shape: (W,)
        """
        X = self.cache['X']
        A1 = self.cache['A1']
        Z1 = self.cache['Z1']
        A2 = self.cache['A2']
        Z2 = self.cache['Z2']
        A3 = self.cache['A3']
        
        N = X.shape[0]
        
        # Output layer gradients (Softmax + Cross-Entropy)
        dZ3 = (A3 - Y) / N
        dW3 = A2.T @ dZ3
        db3 = np.sum(dZ3, axis=0)
        
        # Hidden layer 2 gradients
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * relu_derivative(Z2)
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0)
        
        # Hidden layer 1 gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0)
        
        grads = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2,
            'dW3': dW3, 'db3': db3
        }
        return grads

    def update_params(self, grads: dict, lr: float):
        """Update network weights and biases using Mini-batch Gradient Descent."""
        self.W1 -= lr * grads['dW1']
        self.b1 -= lr * grads['db1']
        self.W2 -= lr * grads['dW2']
        self.b2 -= lr * grads['db2']
        self.W3 -= lr * grads['dW3']
        self.b3 -= lr * grads['db3']

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for given samples."""
        probs = self.forward(X)
        return np.argmax(probs, axis=-1)

    def evaluate(self, X: np.ndarray, Y: np.ndarray) -> tuple[float, float]:
        """
        Evaluate network on dataset (X, Y).
        Returns:
            (loss, accuracy)
        """
        probs = self.forward(X)
        loss = cross_entropy_loss(probs, Y)
        y_true = np.argmax(Y, axis=-1)
        y_pred = np.argmax(probs, axis=-1)
        acc = float(np.mean(y_pred == y_true))
        return loss, acc

    def check_numerical_gradients(self, X: np.ndarray, Y: np.ndarray, num_checks: int = 10,
                                   eps: float = 1e-5, tol: float = 1e-6, sample_seed: int = 0) -> dict:
        """
        Step 5: Verify with numerical gradient checking:
        Nudge 10 random weights by ±1e-5, compare analytic and numerical slopes.
        Relative error must be < 1e-6 for at least 9 of 10 weights.
        
        Relative Error Formula:
            rel_err = |analytic - numeric| / (max(|analytic|, |numeric|) + 1e-12)
        """
        # Forward and backward pass to get analytic gradients
        self.forward(X)
        analytic_grads = self.backward(Y)
        
        param_list = [
            ('W1', self.W1, analytic_grads['dW1']),
            ('W2', self.W2, analytic_grads['dW2']),
            ('W3', self.W3, analytic_grads['dW3']),
        ]
        
        # Flattened pointers to select random weights across all weight matrices
        all_weights = []
        for name, param, grad in param_list:
            for idx in np.ndindex(param.shape):
                all_weights.append((name, param, grad, idx))
                
        # Sample weights using sample_seed
        rng = np.random.default_rng(sample_seed)
        sample_indices = rng.choice(len(all_weights), size=num_checks, replace=False)
        results = []
        passed_count = 0
        
        for s_idx in sample_indices:
            name, param, grad_arr, idx = all_weights[s_idx]
            analytic_slope = grad_arr[idx]
            
            # Save original weight value
            orig_val = param[idx]
            
            # Compute f(theta + eps)
            param[idx] = orig_val + eps
            probs_plus = self.forward(X)
            loss_plus = cross_entropy_loss(probs_plus, Y)
            
            # Compute f(theta - eps)
            param[idx] = orig_val - eps
            probs_minus = self.forward(X)
            loss_minus = cross_entropy_loss(probs_minus, Y)
            
            # Restore original weight
            param[idx] = orig_val
            
            # Two-sided numerical slope
            numeric_slope = (loss_plus - loss_minus) / (2.0 * eps)
            
            # Relative error
            denom = max(abs(analytic_slope), abs(numeric_slope)) + 1e-12
            rel_error = abs(analytic_slope - numeric_slope) / denom
            passed = bool(rel_error < tol)
            if passed:
                passed_count += 1
                
            results.append({
                'param': name,
                'index': idx,
                'analytic': float(analytic_slope),
                'numeric': float(numeric_slope),
                'rel_error': float(rel_error),
                'passed': passed
            })
            
        # Re-run forward pass to restore pristine cached state
        self.forward(X)
        
        return {
            'results': results,
            'passed_count': passed_count,
            'num_checks': num_checks,
            'all_passed_threshold': (passed_count >= 9)
        }
