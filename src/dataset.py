import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_data(csv_path="heart.csv"):
    """
    Loads the UCI Heart Disease dataset and performs preprocessing.
    
    Args:
        csv_path (str): Path to the heart.csv file.
        
    Returns:
        X_train, X_test, y_train, y_test (numpy arrays)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Error: Could not find '{csv_path}'. Please download it to the project root.")

    # 1. Load Data
    df = pd.read_csv(csv_path)

    # 2. Separate Target (y) and Features (X)
    # Target: 1 = Disease, 0 = No Disease
    y = df['target'].values
    X = df.drop(['target'], axis=1).values

    # 3. Preprocessing: Standardization
    # Neural Networks require scaled data (mean=0, var=1) for stable convergence.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 4. Split into Train/Test (80% Train, 20% Test)
    # This Test set is reserved for the Global Server to evaluate the final model.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def get_client_data(client_id, total_clients, X_train, y_train, batch_size=32):
    """
    Simulates a 'Non-IID' (Non-Independent and Identically Distributed) data split.
    We sort the data by a specific feature (Age) before slicing.
    
    This means:
    - Client 0 gets mostly younger patients.
    - Client N gets mostly older patients.
    
    This simulates real-world hospital differences (e.g., a pediatric clinic vs. a geriatric center).
    """
    
    # 1. Sort indices by the first column (Age is usually index 0 in UCI Heart)
    # Sorting ensures the data distribution is NOT random (Non-IID).
    sort_indices = np.argsort(X_train[:, 0])
    
    # 2. Calculate slice range for this specific client
    samples_per_client = len(X_train) // total_clients
    start_idx = client_id * samples_per_client
    end_idx = start_idx + samples_per_client
    
    # Handle the last client getting any remaining samples
    if client_id == total_clients - 1:
        end_idx = len(X_train)
        
    # 3. Slice the sorted data
    client_indices = sort_indices[start_idx:end_idx]
    
    # 4. Convert to PyTorch Tensors
    X_client = torch.tensor(X_train[client_indices], dtype=torch.float32)
    # Ensure y is shape (N, 1) for BCELoss
    y_client = torch.tensor(y_train[client_indices], dtype=torch.float32).unsqueeze(1) 
    
    # 5. Create DataLoader
    dataset = TensorDataset(X_client, y_client)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"   [Client {client_id}] Loaded {len(dataset)} samples (Non-IID: Age-biased).")
    return dataloader

def get_test_loader(X_test, y_test, batch_size=32):
    """
    Creates a DataLoader for the global validation set.
    """
    X_t = torch.tensor(X_test, dtype=torch.float32)
    y_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size)

# --- Local Testing Block ---
if __name__ == "__main__":
    print("Testing dataset.py...")
    try:
        # Load raw data
        X_train, X_test, y_train, y_test = load_data()
        print(f"✅ Data Loaded. Global Train Size: {X_train.shape[0]}, Test Size: {X_test.shape[0]}")
        
        # Simulate getting data for Client 0 (Younger patients)
        loader = get_client_data(0, 3, X_train, y_train)
        
        # Check batch shape
        data, target = next(iter(loader))
        print(f"✅ Batch Shape: {data.shape} | Target Shape: {target.shape}")
        print("Success! Dataset logic is ready.")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")