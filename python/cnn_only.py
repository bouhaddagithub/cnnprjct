# cnn_only.py
from common_utils import *
import torch.nn as nn
import json
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CNN-only model and export parameters")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-dir", type=str, default="./exports/cnn_only")
    parser.add_argument("--train-subset", type=int, default=None)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    train_loader, test_loader = get_mnist_loaders(args.batch_size, args.train_subset)

    class ConvOnlyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 8, kernel_size=5)
            self.fc = nn.Linear(8 * 24 * 24, 10)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = ConvOnlyNet()
    print("Training CNN-only model...")
    model = train_model(model, train_loader, epochs=args.epochs)
    acc, preds, targets = test_model(model, test_loader)
    print(f"Test Accuracy: {acc:.2f}%")

    # Export parameters
    save_tensor_csv(model.conv.weight, f"{args.out_dir}/conv_weight")
    save_tensor_csv(model.conv.bias, f"{args.out_dir}/conv_bias")
    save_tensor_csv(model.fc.weight, f"{args.out_dir}/fc_weight")
    save_tensor_csv(model.fc.bias, f"{args.out_dir}/fc_bias")

    np.savetxt(f"{args.out_dir}/test_preds.csv", preds, fmt="%d")
    np.savetxt(f"{args.out_dir}/test_targets.csv", targets, fmt="%d")
    with open(f"{args.out_dir}/test_summary.json", "w") as f:
        json.dump({"accuracy": acc}, f)
