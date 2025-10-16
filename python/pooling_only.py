# pooling_only.py
from common_utils import *
import torch.nn as nn
import json
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Pooling-only model and export parameters")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pool-k", type=int, default=2)
    parser.add_argument("--out-dir", type=str, default="./exports/pooling_only")
    parser.add_argument("--train-subset", type=int, default=None)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    train_loader, test_loader = get_mnist_loaders(args.batch_size, args.train_subset)

    class PoolOnlyNet(nn.Module):
        def __init__(self, pool_k=2):
            super().__init__()
            self.pool = nn.MaxPool2d(pool_k)
            self.fc = nn.Linear((28 // pool_k) * (28 // pool_k), 10)

        def forward(self, x):
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = PoolOnlyNet(args.pool_k)
    print("Training Pooling-only model...")
    model = train_model(model, train_loader, epochs=args.epochs)
    acc, preds, targets = test_model(model, test_loader)
    print(f"Test Accuracy: {acc:.2f}%")

    with open(f"{args.out_dir}/pooling.meta.json", "w") as f:
        json.dump({"kernel_size": args.pool_k}, f)

    save_tensor_csv(model.fc.weight, f"{args.out_dir}/fc_weight")
    save_tensor_csv(model.fc.bias, f"{args.out_dir}/fc_bias")

    np.savetxt(f"{args.out_dir}/test_preds.csv", preds, fmt="%d")
    np.savetxt(f"{args.out_dir}/test_targets.csv", targets, fmt="%d")
    with open(f"{args.out_dir}/test_summary.json", "w") as f:
        json.dump({"accuracy": acc}, f)
