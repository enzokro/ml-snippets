import numpy as np
from typing import Callable

def relu(x: np.ndarray, derivative=False) -> np.ndarray:
    """Rectified Linear Unit (ReLU) activation function."""
    if derivative:
        return np.where(x > 0, 1, 0)
    return np.maximum(0, x)

def sigmoid(x: np.ndarray, derivative=False) -> np.ndarray:
    """Sigmoid activation function."""
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

class DenseLayer:
    """A fully connected dense layer in a neural network."""
    
    def __init__(self, input_size: int, output_size: int, activation: Callable = None):
        """Initialize the dense layer with weights and bias."""
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)
        self.activation = activation
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Perform forward propagation through the dense layer."""
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias
        
        if self.activation:
            self.output = self.activation(self.output)
        
        return self.output
    
    def backward(self, grad_output: np.ndarray, learning_rate: float) -> np.ndarray:
        """Perform backward propagation and update weights and bias."""
        if self.activation:
            grad_output = grad_output * self.activation(self.output, derivative=True)
        
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.inputs.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)
        
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input

class NeuralNetwork:
    """A simple feedforward neural network."""
    
    def __init__(self, layer_sizes: list, activation: Callable = None, output_activation: Callable = None):
        """Initialize the neural network with the specified layer sizes."""
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            if i == len(layer_sizes) - 2:
                self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1], output_activation))
            else:
                self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1], activation))
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Perform forward propagation through the neural network."""
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, grad_output: np.ndarray, learning_rate: float):
        """Perform backward propagation and update weights and biases."""
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)
    
    def train(self, inputs: np.ndarray, targets: np.ndarray, epochs: int, learning_rate: float):
        """Train the neural network using the provided inputs and targets."""
        for epoch in range(epochs):
            outputs = self.forward(inputs)
            loss = np.mean((outputs - targets) ** 2)
            grad_output = 2 * (outputs - targets) / len(inputs)
            self.backward(grad_output, learning_rate)
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
