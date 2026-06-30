#!/usr/bin/env python3
"""Train CIFAR-10 ResNet-18 and a no-residual PlainNet-18 ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if residual and (stride != 1 or in_planes != planes):
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)
        return self.relu(out)


class CIFARResNet(nn.Module):
    """ResNet-18 style network adapted for 32x32 CIFAR images.

    The stem uses a 3x3 stride-1 convolution and no max-pooling, which is the
    common CIFAR adaptation of ImageNet ResNet.
    """

    def __init__(
        self,
        layers: Iterable[int],
        num_classes: int = 10,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.in_planes = 64
        self.residual = residual
        layer_counts = list(layers)

        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, layer_counts[0], stride=1)
        self.layer2 = self._make_layer(128, layer_counts[1], stride=2)
        self.layer3 = self._make_layer(256, layer_counts[2], stride=2)
        self.layer4 = self._make_layer(512, layer_counts[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(
                BasicBlock(
                    self.in_planes,
                    planes,
                    stride=current_stride,
                    residual=self.residual,
                )
            )
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def build_model(name: str) -> nn.Module:
    if name == "resnet18":
        return CIFARResNet([2, 2, 2, 2], residual=True)
    if name == "plain18":
        return CIFARResNet([2, 2, 2, 2], residual=False)
    if name == "resnet34":
        return CIFARResNet([3, 4, 6, 3], residual=True)
    if name == "plain34":
        return CIFARResNet([3, 4, 6, 3], residual=False)
    raise ValueError(f"Unsupported model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def split_indices(total: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(total))
    generator = random.Random(seed)
    generator.shuffle(indices)
    val_size = int(total * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def take_subset(indices: list[int], limit: int) -> list[int]:
    if limit and limit > 0:
        return indices[: min(limit, len(indices))]
    return indices


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    data_dir = Path(args.data_dir)
    train_full_aug = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=args.download,
        transform=train_transform,
    )
    train_full_eval = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=eval_transform,
    )
    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=args.download,
        transform=eval_transform,
    )

    train_indices, val_indices = split_indices(len(train_full_aug), args.val_ratio, args.seed)
    train_indices = take_subset(train_indices, args.subset_train)
    val_indices = take_subset(val_indices, args.subset_val)
    test_indices = take_subset(list(range(len(test_set))), args.subset_test)

    train_set = Subset(train_full_aug, train_indices)
    val_set = Subset(train_full_eval, val_indices)
    test_set = Subset(test_set, test_indices)

    common_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **common_kwargs)
    return train_loader, val_loader, test_loader


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float
    seconds: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            if is_train:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total_seen += batch_size

    return total_loss / total_seen, total_correct / total_seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, float]:
    return run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=use_amp)


def write_metrics_csv(metrics: list[EpochMetrics], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        for row in metrics:
            writer.writerow(asdict(row))


def plot_curves(metrics: list[EpochMetrics], output_path: Path, title: str) -> None:
    epochs = [row.epoch for row in metrics]
    train_loss = [row.train_loss for row in metrics]
    val_loss = [row.val_loss for row in metrics]
    train_acc = [row.train_acc for row in metrics]
    val_acc = [row.val_acc for row in metrics]

    best_val_index = max(range(len(val_acc)), key=val_acc.__getitem__)
    best_epoch = epochs[best_val_index]
    tick_step = 10 if max(epochs) >= 40 else max(1, max(epochs) // 5)
    epoch_ticks = [1] + list(range(tick_step, max(epochs) + 1, tick_step))
    if 1 not in epoch_ticks:
        epoch_ticks = [1] + epoch_ticks

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=180)
    colors = {"train": "#2563eb", "validation": "#dc2626", "best": "#111827"}

    axes[0].plot(epochs, train_loss, linewidth=2.0, color=colors["train"], label="train")
    axes[0].plot(epochs, val_loss, linewidth=2.0, color=colors["validation"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross entropy")
    axes[0].set_xticks(epoch_ticks)
    axes[0].grid(True, alpha=0.18)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, train_acc, linewidth=2.0, color=colors["train"], label="train")
    axes[1].plot(epochs, val_acc, linewidth=2.0, color=colors["validation"], label="validation")
    axes[1].scatter(
        [best_epoch],
        [val_acc[best_val_index]],
        s=42,
        color=colors["best"],
        zorder=3,
        label=f"best val {val_acc[best_val_index]:.3f}",
    )
    axes[1].annotate(
        f"epoch {best_epoch}",
        xy=(best_epoch, val_acc[best_val_index]),
        xytext=(8, -18),
        textcoords="offset points",
        fontsize=8,
        color=colors["best"],
        arrowprops={"arrowstyle": "-", "color": colors["best"], "lw": 0.8},
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(max(0.0, min(train_acc + val_acc) - 0.08), 1.0)
    axes[1].set_xticks(epoch_ticks)
    axes[1].grid(True, alpha=0.18)
    axes[1].legend(frameon=False, loc="lower right")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["resnet18", "plain18", "resnet34", "plain34"],
        default="resnet18",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--subset-train", type=int, default=0, help="Use first N train examples after split; 0 means full split.")
    parser.add_argument("--subset-val", type=int, default=0, help="Use first N validation examples; 0 means full validation split.")
    parser.add_argument("--subset-test", type=int, default=0, help="Use first N test examples; 0 means full test split.")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    if args.output_dir is None:
        args.output_dir = f"results/{args.model}_e{args.epochs}"
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders(args)
    model = build_model(args.model).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    use_amp = args.amp and args.device.type == "cuda"
    scaler = torch.amp.GradScaler(device=args.device.type, enabled=use_amp)

    config = vars(args).copy()
    config["device"] = str(args.device)
    config["parameter_count"] = count_parameters(model)
    config["train_size"] = len(train_loader.dataset)
    config["val_size"] = len(val_loader.dataset)
    config["test_size"] = len(test_loader.dataset)
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_val_acc = -math.inf
    best_epoch = 0
    metrics: list[EpochMetrics] = []

    print(
        f"Training {args.model} on {args.device} | "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}"
    )
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            args.device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device, use_amp=use_amp)
        scheduler.step()
        elapsed = time.time() - start
        current_lr = scheduler.get_last_lr()[0]
        row = EpochMetrics(epoch, train_loss, train_acc, val_loss, val_acc, current_lr, elapsed)
        metrics.append(row)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={current_lr:.5f} time={elapsed:.1f}s",
            flush=True,
        )

    torch.save(model.state_dict(), output_dir / "last_model.pt")
    write_metrics_csv(metrics, output_dir / "metrics.csv")
    (output_dir / "metrics.json").write_text(
        json.dumps([asdict(row) for row in metrics], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_curves(metrics, output_dir / "curves.png", title=f"{args.model} CIFAR-10")

    best_state = torch.load(output_dir / "best_model.pt", map_location=args.device)
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, args.device, use_amp=use_amp)

    summary = {
        "model": args.model,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "parameter_count": count_parameters(model),
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "test_size": len(test_loader.dataset),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Finished {args.model}: best_val_acc={best_val_acc:.4f} "
        f"test_acc={test_acc:.4f} best_epoch={best_epoch}"
    )


if __name__ == "__main__":
    main()
