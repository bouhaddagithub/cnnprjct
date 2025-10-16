# pipeline.py
from common_utils import *
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Full MNIST pipeline")
    parser.add_argument("--mode", choices=["train", "assemble"], default="train")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cnn-dir", type=str, default="./exports/cnn_only")
    parser.add_argument("--pool-dir", type=str, default="./exports/pooling_only")
    parser.add_argument("--fc-dir", type=str, default="./exports/fc_only")
    parser.add_argument("--out-dir", type=str, default="./exports/pipeline")
    parser.add_argument("--train-subset", type=int, default=None)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    train_loader, test_loader = get_mnist_loaders(args.batch_size, args.train_subset)

    def load_param(prefix):
        npy = prefix + ".npy"
        return np.load(npy) if Path(npy).exists() else None

    class PipelineNet(nn.Module):
        def __init__(self, conv_w=None, conv_b=None, fc_w=None, fc_b=None):
            super().__init__()
            if conv_w is not None:
                out_c, in_c, k, _ = conv_w.shape
                self.conv = nn.Conv2d(in_c, out_c, k)
                self.conv.weight = nn.Parameter(torch.tensor(conv_w))
                self.conv.bias = nn.Parameter(torch.tensor(conv_b))
            else:
                self.conv = nn.Conv2d(1, 8, 5)

            self.pool = nn.MaxPool2d(2)
            flat = 8 * 12 * 12
            self.fc = nn.Linear(flat, 10)
            if fc_w is not None and fc_b is not None:
                self.fc.weight = nn.Parameter(torch.tensor(fc_w))
                self.fc.bias = nn.Parameter(torch.tensor(fc_b))

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    if args.mode == "train":
        model = PipelineNet()
        print("Training full pipeline model...")
        model = train_model(model, train_loader, epochs=args.epochs)
        acc, preds, targets = test_model(model, test_loader)
        print(f"Test Accuracy: {acc:.2f}%")

        save_tensor_csv(model.conv.weight, f"{args.out_dir}/conv_weight")
        save_tensor_csv(model.conv.bias, f"{args.out_dir}/conv_bias")
        save_tensor_csv(model.fc.weight, f"{args.out_dir}/fc_weight")
        save_tensor_csv(model.fc.bias, f"{args.out_dir}/fc_bias")

        with open(f"{args.out_dir}/test_summary.json", "w") as f:
            json.dump({"accuracy": acc}, f)

    else:
        print("Assembling pipeline from exported parameters...")
        conv_w = load_param(f"{args.cnn-dir}/conv_weight")
        conv_b = load_param(f"{args.cnn-dir}/conv_bias")
        fc_w = load_param(f"{args.fc-dir}/fc2_weight") or load_param(f"{args.fc-dir}/fc_weight")
        fc_b = load_param(f"{args.fc-dir}/fc2_bias") or load_param(f"{args.fc-dir}/fc_bias")

        model = PipelineNet(conv_w, conv_b, fc_w, fc_b)
        acc, preds, targets = test_model(model, test_loader)
        print(f"Assembled pipeline accuracy: {acc:.2f}%")

        with open(f"{args.out_dir}/test_summary.json", "w") as f:
            json.dump({"accuracy": acc}, f)
