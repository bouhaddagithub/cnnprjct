# common_utils.py
"""
Common utilities for MNIST experiments:
- Dataset loader
- Train/test loops
- Parameter export to CSV/NPY with shape meta
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_tensor_csv(tensor: torch.Tensor, out_path: str):
    """Save a tensor to CSV, NPY, and meta.json."""
    ensure_dir(os.path.dirname(out_path) or '.')
    arr = tensor.detach().cpu().numpy()
    meta = {"shape": list(arr.shape)}

    # Save meta
    with open(out_path + ".meta.json", "w") as f:
        json.dump(meta, f)

    # Save numpy
    np.save(out_path + ".npy", arr)

    # Save flattened CSV
    with open(out_path + ".csv", "w") as f:
        f.write(json.dumps(meta) + "\n")
        flat = arr.ravel()
        f.write(",".join(map(str, flat.tolist())))


def get_mnist_loaders(batch_size=64, train_subset=None):
    """Load MNIST dataset and return DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)

    if train_subset is not None:
        train_dataset.data = train_dataset.data[:train_subset]
        train_dataset.targets = train_dataset.targets[:train_subset]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(model, train_loader, epochs=5, lr=1e-3, device=None):
    """Train model on MNIST."""
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
        print(f"Epoch {epoch+1}/{epochs}  Loss={total_loss/total:.4f}  Acc={100.*correct/total:.2f}%")

    return model


def test_model(model, test_loader, device=None):
    """Evaluate model on MNIST test set."""
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    acc = 100.0 * correct / total
    return acc, np.concatenate(all_preds), np.concatenate(all_targets)
