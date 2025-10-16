# fc_only.py
from common_utils import *
import torch.nn as nn
import json
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train FC-only model and export parameters")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--out-dir", type=str, default="./exports/fc_only")
    parser.add_argument("--train-subset", type=int, default=None)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    train_loader, test_loader = get_mnist_loaders(args.batch_size, args.train_subset)

    class FCOnlyNet(nn.Module):
        def __init__(self, hidden=128):
            super().__init__()
            self.fc1 = nn.Linear(28*28, hidden)
            self.fc2 = nn.Linear(hidden, 10)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    model = FCOnlyNet(args.hidden)
    print("Training FC-only model...")
    model = train_model(model, train_loader, epochs=args.epochs)
    acc, preds, targets = test_model(model, test_loader)
    print(f"Test Accuracy: {acc:.2f}%")

    save_tensor_csv(model.fc1.weight, f"{args.out_dir}/fc1_weight")
    save_tensor_csv(model.fc1.bias, f"{args.out_dir}/fc1_bias")
    save_tensor_csv(model.fc2.weight, f"{args.out_dir}/fc2_weight")
    save_tensor_csv(model.fc2.bias, f"{args.out_dir}/fc2_bias")

    np.savetxt(f"{args.out_dir}/test_preds.csv", preds, fmt="%d")
    np.savetxt(f"{args.out_dir}/test_targets.csv", targets, fmt="%d")
    with open(f"{args.out_dir}/test_summary.json", "w") as f:
        json.dump({"accuracy": acc}, f)
