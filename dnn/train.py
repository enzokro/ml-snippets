import numpy as np
from model import NeuralNetwork
from model import relu, sigmoid
from utils import generate_data, plot_decision_boundary

def main():
    """Main function to train and evaluate the neural network."""
    # Generate toy dataset
    inputs, targets = generate_data(num_samples=100, num_classes=2)
    
    # Create a neural network
    layer_sizes = [2, 4, 4, 1]
    activation = relu
    output_activation = sigmoid
    learning_rate = 0.1
    epochs = 1000
    
    nn = NeuralNetwork(layer_sizes, activation, output_activation)
    
    # Train the neural network
    nn.train(inputs, targets, epochs, learning_rate)
    
    # Evaluate the trained network
    outputs = nn.forward(inputs)
    predicted_classes = np.round(outputs)
    accuracy = np.mean(predicted_classes == targets)
    print(f"Accuracy: {accuracy:.2f}")
    
    # Plot the decision boundary
    plot_decision_boundary(nn, inputs, targets)

if __name__ == "__main__":
    main()
