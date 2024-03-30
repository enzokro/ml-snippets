import numpy as np
import matplotlib.pyplot as plt

def generate_data(num_samples: int, num_classes: int) -> tuple:
    """Generate a toy dataset for binary classification."""
    class_centers = np.array([[-1, -1], [1, 1]])
    class_labels = np.arange(num_classes)
    
    inputs = []
    targets = []
    for _ in range(num_samples):
        class_idx = np.random.choice(class_labels)
        center = class_centers[class_idx]
        x = center + np.random.randn(2) * 0.5
        inputs.append(x)
        targets.append(class_idx)
    
    inputs = np.array(inputs)
    targets = np.array(targets).reshape(-1, 1)
    
    return inputs, targets

def plot_decision_boundary(model, inputs: np.ndarray, targets: np.ndarray):
    """Plot the decision boundary of a trained model."""
    x_min, x_max = inputs[:, 0].min() - 0.5, inputs[:, 0].max() + 0.5
    y_min, y_max = inputs[:, 1].min() - 0.5, inputs[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(inputs[:, 0], inputs[:, 1], c=targets.ravel(), cmap=plt.cm.Spectral)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Decision Boundary")
    plt.show()
