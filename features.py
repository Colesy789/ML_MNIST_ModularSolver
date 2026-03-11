import torch
from sklearn.decomposition import PCA

def get_processed_data(train_dataset, test_dataset, mode, n_components=None, is_cnn=False):
    
    y_train = train_dataset.targets
    y_test = test_dataset.targets

    if is_cnn:
        # Keep the 2D shape (Channels=1, H=28, W=28)
        # We add the .unsqueeze(1) to add the "Channel" dimension
        x_train = train_dataset.data.float().unsqueeze(1) 
        x_test = test_dataset.data.float().unsqueeze(1)
        return x_train, y_train, x_test, y_test

    # Flatten the 28x28 images into 784 vectors
    x_train = train_dataset.data.float().view(-1, 784).numpy()
    x_test = test_dataset.data.float().view(-1, 784).numpy()

    # Apply user selection
    if mode == "pca":
        # Reduces the 784 dimensions down to most important n_components.
        pca = PCA(n_components=n_components)
        x_train = pca.fit_transform(x_train) # Learn patterns AND transform
        x_test = pca.transform(x_test) # Use learned patterns to transform test data
        print(f"Total Variance Explained: {sum(pca.explained_variance_ratio_) * 100:.2f}%")
    
    # Convert back to Tensors ans Scikit-learn (PCA) returns NumPy arrays. PyTorch needs Tensors.
    return (torch.tensor(x_train).float(), y_train, 
            torch.tensor(x_test).float(), y_test)