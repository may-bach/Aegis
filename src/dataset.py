import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(path="heart.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist.")
    
    df = pd.read_csv(path)
    y = df["target"].values
    X = df.drop(["target"], axis=1).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=16)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    return Xtr, Xte, ytr, yte

def get_client_data(client_id, total_clients, Xtr, ytr, batch=32):
    sorted_idx = np.argsort(Xtr[:, 0])
    spc = len(Xtr) // total_clients  #samples per client
    start_idx = client_id * spc
    end_idx = start_idx + spc

    if client_id == total_clients - 1:
        end_idx = len(Xtr)

    client_idx = sorted_idx[start_idx:end_idx]

    X_client = torch.tensor(Xtr[client_idx], dtype=torch.float32)
    y_client = torch.tensor(ytr[client_idx], dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_client, y_client)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)

    return dataloader

def get_test_loader(X_test, y_test, batch_size=32):
    X_t = torch.tensor(X_test, dtype=torch.float32)
    y_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size)


if __name__ == "__main__":
    print("Testing dataset.py...")
    try:
        
        X_train, X_test, y_train, y_test = load_data()
        print(f"Data Loaded. Global Train Size: {X_train.shape[0]}, Test Size: {X_test.shape[0]}")
        
        loader = get_client_data(0, 3, X_train, y_train)
        
        data, target = next(iter(loader))
        print(f"Batch Shape: {data.shape} | Target Shape: {target.shape}")   #checking dimensions of loader
        print("Success! Dataset logic is ready.")
        
    except Exception as e:
        print(f"Error during test: {e}")