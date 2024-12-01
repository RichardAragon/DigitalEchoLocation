# app.py
"""
Digital Echolocation - Generalized Framework
Released under the MIT License
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------
# Simulation Parameters
# -----------------------
CUBE_RADIUS = 1.0
MAX_REFLECTIONS = 5
NUM_SAMPLES = 1000
EPOCHS = 50
BATCH_SIZE = 32

# -----------------------
# Generate Hyperbolic Points
# -----------------------
def generate_point_inside():
    """Generate a random point inside the Poincaré disk (hyperbolic cube)."""
    r = np.sqrt(np.random.uniform(0, CUBE_RADIUS**2))
    theta = np.random.uniform(0, 2 * np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y])

# -----------------------
# Echolocation Simulation
# -----------------------
def simulate_echolocation(points):
    """
    Simulate echolocation signals for given points.
    :param points: List of points inside the hyperbolic cube
    :return: Echolocation signal reflections
    """
    reflections = np.zeros(MAX_REFLECTIONS)
    for point in points:
        norm_point = np.linalg.norm(point)
        if norm_point < CUBE_RADIUS:
            for i in range(1, MAX_REFLECTIONS + 1):
                distance = np.log((1 + norm_point) / (1 - norm_point))
                signal_strength = np.exp(-distance * i)
                reflections[i - 1] += signal_strength
    return reflections

# -----------------------
# Data Generation
# -----------------------
def generate_data(num_samples):
    """
    Generate dataset for "one" vs "two" object classification.
    :param num_samples: Number of samples to generate
    :return: Data and labels
    """
    data = []
    labels = []
    for _ in range(num_samples):
        if np.random.rand() > 0.5:
            # One object
            points = [generate_point_inside()]
            label = 1  # "One"
        else:
            # Two objects
            points = [generate_point_inside(), generate_point_inside()]
            label = 2  # "Two"
        
        signals = simulate_echolocation(points)
        data.append(signals)
        labels.append(label)
    
    return np.array(data), np.array(labels)

# -----------------------
# Neural Network Model
# -----------------------
class EcholocationNet(nn.Module):
    def __init__(self):
        super(EcholocationNet, self).__init__()
        self.fc1 = nn.Linear(MAX_REFLECTIONS, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)  # Output layer for 2 classes (one, two)
        
    def forward(self, x):
        x = torch.Tensor(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -----------------------
# Training Loop
# -----------------------
def train_model(model, X_train, y_train, epochs, batch_size):
    """
    Train the neural network model.
    :param model: Neural network model
    :param X_train: Training data
    :param y_train: Training labels
    :param epochs: Number of training epochs
    :param batch_size: Batch size
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        permutation = np.random.permutation(X_train.shape[0])
        epoch_loss = 0
        for i in range(0, X_train.shape[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, torch.LongTensor(batch_y - 1))  # Labels must start at 0
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

# -----------------------
# Evaluation
# -----------------------
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    :param model: Neural network model
    :param X_test: Test data
    :param y_test: Test labels
    :return: Test accuracy
    """
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs.data, 1)
        accuracy = accuracy_score(y_test, predicted.numpy() + 1)  # Convert back to labels 1, 2
        return accuracy

# -----------------------
# Main Function
# -----------------------
def main(args):
    # Generate data
    print("Generating data...")
    data, labels = generate_data(args.num_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    
    # Initialize model
    print("Initializing model...")
    model = EcholocationNet()

    # Train model
    print("Training model...")
    train_model(model, X_train, y_train, args.epochs, args.batch_size)
    
    # Evaluate model
    print("Evaluating model...")
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    # Visualize results (optional)
    if args.visualize:
        one_signals = data[labels == 1]
        two_signals = data[labels == 2]
        
        plt.figure(figsize=(12, 6))
        for i in range(MAX_REFLECTIONS):
            plt.subplot(1, MAX_REFLECTIONS, i+1)
            plt.hist(one_signals[:, i], bins=20, alpha=0.5, label='One')
            plt.hist(two_signals[:, i], bins=20, alpha=0.5, label='Two')
            plt.title(f'Reflection {i+1}')
            plt.legend()
        plt.tight_layout()
        plt.show()

# -----------------------
# Command Line Arguments
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digital Echolocation Framework")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES, help="Number of samples to generate")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--visualize", action="store_true", help="Visualize the results")
    args = parser.parse_args()
    main(args)
