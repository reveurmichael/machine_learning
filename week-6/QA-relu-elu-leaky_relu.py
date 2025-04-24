"""
This is very idiomatic PyTorch code, well-suited for beginners, educational demos, or baseline experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic neural network with ReLU + Dropout
class BasicReLUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(BasicReLUNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # ReLU activation
        x = self.dropout(x)  # Apply dropout after activation
        x = self.fc2(x)
        return x

# Neural network with Leaky ReLU + Dropout
class LeakyReLUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5, negative_slope=0.01):
        super(LeakyReLUNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.negative_slope = negative_slope
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Neural network with ELU + Dropout
class ELUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5, alpha=1.0):
        super(ELUNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.alpha = alpha
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.elu(x, alpha=self.alpha)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

## Comparing Performance: A Simple Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define a function to train and evaluate models
def train_and_evaluate(model, train_loader, test_loader, epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)  # Flatten the inputs
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Testing
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.view(inputs.size(0), -1)  # Flatten the inputs
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return train_losses, test_accuracies

# Example usage
if __name__ == "__main__":
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST('./data_gitignore', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data_gitignore", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define models
    input_dim = 28 * 28  # MNIST images are 28x28
    hidden_dim = 128
    output_dim = 10  # 10 digits

    relu_model = BasicReLUNet(input_dim, hidden_dim, output_dim)
    leaky_relu_model = LeakyReLUNet(input_dim, hidden_dim, output_dim)
    elu_model = ELUNet(input_dim, hidden_dim, output_dim)

    # Train and evaluate models
    print("Training with ReLU...")
    relu_losses, relu_accs = train_and_evaluate(relu_model, train_loader, test_loader)

    print("\nTraining with Leaky ReLU...")
    leaky_losses, leaky_accs = train_and_evaluate(leaky_relu_model, train_loader, test_loader)

    print("\nTraining with ELU...")
    elu_losses, elu_accs = train_and_evaluate(elu_model, train_loader, test_loader)

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(relu_losses, label='ReLU')
    plt.plot(leaky_losses, label='Leaky ReLU')
    plt.plot(elu_losses, label='ELU')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(relu_accs, label='ReLU')
    plt.plot(leaky_accs, label='Leaky ReLU')
    plt.plot(elu_accs, label='ELU')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()
