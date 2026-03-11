import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

from features import get_processed_data
from models import get_model

######## Download and prepare the MNIST dataset ########

# Covert PIL images to PyTorch tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download the MNIST dataset
# Training Data
train_dataset = torchvision.datasets.MNIST(
    root='./data',      # Where to store the data
    train=True,         # Training set not test set
    download=True,      # Download if not present
    transform=transform # Apply the transformations we defined above
)

# Test Data
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False,        # Test set not training set
    download=True, 
    transform=transform
)

######## User inputs for model configuration and training methods ########

print("--- MNIST Machine Learning Project ---")

# 1. Choose Feature Selection
print("\nAvailable Feature Selection:")
print("1. Raw Pixels (784 features)")
print("2. PCA (Reduced to 50 features)")
feature_choice = input("Select (1 or 2): ")

# Initialize the number of features
if feature_choice == "2":
    selection_mode = "pca"
    # Ask for the specific PCA amount
    pca_input = input("How many PCA components? (Recommended 10-100): ")
    n_features = int(pca_input)
else:
    selection_mode = "raw"
    n_features = 784

# 2. Choose Model Complexity
print("\nAvailable Models:")
print("1. Simple Linear (Fast)")
print("2. Multi-Layer Perceptron (Standard)")
print("3. Multi-Layer Perceptron (Dropout & Batch Normalization)") # forces the model to find multiple ways to recognize a digit, rather than relying on one specific pixel pattern
print("4. Convolutional Neural Network (Best for Images, but more complex)")
model_choice = input("Select (1, 2, 3, or 4): ")

if model_choice == "3":
    dropout_input = input("Dropout Rate? (e.g., 0.2 for 20%): ")
    dropout_rate = float(dropout_input)
elif model_choice == "4":
    is_cnn = True
    dropout_rate = 0
else:   
    dropout_rate = 0
    is_cnn = False

# 3. Choose Training Intensity
epochs_input = input("\nHow many training epochs? (e.g., 2 or 5): ")
num_epochs = int(epochs_input)

# 4. Choose learning rate
lr_input = input("\nLearning Rate? (e.g., 0.001): ")
lr = float(lr_input)

######## Process the data based on user selection ########

print(f"\n--- Model Shape ---")

# flattening and PCA handled in features.py
x_train, y_train, x_test, y_test = get_processed_data(
    train_dataset, test_dataset, selection_mode, n_features, is_cnn
)

# Create the DataLoaders using the PROCESSED data
# TensorDataset used because the data is now custom tensors, not raw MNIST images
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64, shuffle=False)

# Verify the new batch shape after processing
images, labels = next(iter(train_loader))
print(f"New Batch shape: {images.shape}") # If PCA was 50: torch.Size([64, 50]), if raw: torch.Size([64, 784])

######## Initialize the model, loss function, and optimizer ########

# Initialize the model using the imported function from models.py
model = get_model(model_choice, n_features, dropout_rate)

# Define the Loss Function
# CrossEntropyLoss is used for classification (0-9)
criterion = nn.CrossEntropyLoss()

# Define the Optimizer (The "Adjuster")
# Adam - a smart algorithm that updates the weights of the model
optimizer = optim.Adam(model.parameters(), lr=lr)

print(f"\n[Status] Model initialized with {n_features} inputs.")

print(f"\n--- Starting Training for {num_epochs} Epochs ---")

####### Training Loop ########

for epoch in range(num_epochs):
    model.train()  # 1. Set the model to training mode
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Clear previous gradients (The old guesses)
        optimizer.zero_grad()
        
        # Forward Pass: Let the model try to guess the digit
        outputs = model(images)
        
        # Calculate Loss: How far was the guess from the actual label?
        loss = criterion(outputs, labels)
        
        # Backward Pass: Calculate the "gradient" (direction to change weights)
        # Uses calculus (Autograd) to trace back from the final error all the way 
        # through the layers to figure out which specific neurons were responsible for the mistake.
        loss.backward()
        
        # Update weights by LR inside the model
        optimizer.step()
        
        running_loss += loss.item()
    
    # Calculate the average loss for this epoch to see if it's going down
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

print("\n--- Training Complete! ---")

######## Evaluation on Test Data ########

print("\n--- Evaluating on Test Data ---")
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

# Torch.no_grad() used because we aren't training, so we don't need to track math
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        # The output is 10 numbers (probabilities). We take the highest one.
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Final Test Accuracy: {accuracy:.2f}%")