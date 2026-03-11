import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Layer 1: Looks for 16 different patterns using 3x3 filters
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) # Shrinks 28x28 to 14x14
        
        # Layer 2: Shrinks 14x14 to 7x7
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Final Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Flatten(), # Finally flatten to feed into the 10-digit output
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.fc(x)
        return x

def get_model(model_choice, n_features, dropout_rate=0.2):
    if model_choice == "1":
        # Simple Linear Model
        return nn.Sequential(
            nn.Linear(n_features, 10) # No hidden layers. It looks at the features and votes directly for a digit.
        )
    elif model_choice == "2":
        # Complex Multi-Layer Perceptron
        return nn.Sequential(
            nn.Linear(n_features, 128), # Takes 'n_features' and expands them into 128 "concepts"
            nn.ReLU(), # Removes negative values, helping the model learn complex shapes.
            nn.Linear(128, 10) # Compresses those 128 concepts down to 10 (the digits 0-9)
        )
    elif model_choice == "3":
        # MLP with Dropout (The "Robust" Model) forces the model to find multiple ways to recognize a digit, rather than relying on one specific pixel pattern
        return nn.Sequential(
                nn.Linear(n_features, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate), # % of neurons sleep each turn
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
    elif model_choice == "4":
        # Convolutional Neural Network (Best for Images, but more complex)
        return CNNModel()