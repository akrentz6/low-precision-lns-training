"""Shared training loop for the CIFAR LNS experiments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
import sys
import time
from typing import Callable

import torch
import torch.nn as nn
import torchdt
from torchdt.lns import LNS16, LNS32
from torchdt.optim import Madam, SGD, lr_scheduler
from torchdt.transforms import DTypeNormalize, ToDType
from torchvision import transforms


CIFAR10_MEAN = (
    0.49139961600303649902,
    0.48215851187705993652,
    0.44653093814849853516,
)
CIFAR10_STD = (
    0.24703231453895568848,
    0.24348483979701995850,
    0.26158782839775085449,
)
CIFAR100_MEAN = (
    0.50707548856735229492,
    0.48654884099960327148,
    0.44091776013374328613,
)
CIFAR100_STD = (
    0.26733365654945373535,
    0.25643849372863769531,
    0.27615079283714294434,
)

ModelFactory = Callable[[int, object, str], nn.Module]


@dataclass(frozen=True)
class ExperimentConfig:
    """The small set of choices that differs between experiment scripts."""

    file_stem: str
    model_name: str
    dataset_type: type
    num_classes: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    model_factory: ModelFactory

    @property
    def dataset_name(self) -> str:
        return self.dataset_type.__name__


@dataclass(frozen=True)
class OutputPaths:
    history: str
    best_model: str
    final_model: str


def parse_args(config: ExperimentConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Train {config.model_name} on {config.dataset_name} in LNS16"
    )
    parser.add_argument(
        "--accumulator",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use LNS32 for reduction accumulators (default: False)",
    )
    parser.add_argument(
        "--accumulator_prec",
        type=int,
        default=16,
        help="Precision for the LNS32 accumulator (default: 16)",
    )
    parser.add_argument(
        "--accumulator_table",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use lookup table for LNS32 accumulator (default: False)",
    )
    parser.add_argument(
        "--prec", type=int, default=16, help="Precision for LNS16 (default: 16)"
    )
    parser.add_argument(
        "--table",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use lookup table for LNS16 operations (default: True)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for training (default: cuda:0)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Batches between logging training status (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training (default: 128)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "madam"],
        help="Optimizer to use (default: sgd)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="SGD weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.999,
        help="Madam beta parameter (default: 0.999)",
    )
    parser.add_argument(
        "--p_scale",
        type=float,
        default=3.0,
        help="Madam weight bound (default: 3.0)",
    )
    parser.add_argument(
        "--g_bound",
        type=float,
        default=10.0,
        help="Madam gradient bound (default: 10.0)",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_outputs(file_stem: str) -> tuple[logging.Logger, OutputPaths]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    log_filename = os.path.join(output_dir, f"{file_stem}_train_{timestamp}.log")
    paths = OutputPaths(
        history=os.path.join(output_dir, f"{file_stem}_history_{timestamp}.json"),
        best_model=os.path.join(output_dir, f"{file_stem}_best_{timestamp}.pt"),
        final_model=os.path.join(output_dir, f"{file_stem}_final_{timestamp}.pt"),
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(file_stem), paths


def configure_lns(args: argparse.Namespace) -> None:
    if args.accumulator:
        LNS32.set_prec(
            args.accumulator_prec,
            table=args.accumulator_table,
            table_device=args.device,
            filestem="./outputs/tab",
        )
    LNS16.set_prec(
        args.prec,
        table=args.table,
        table_device=args.device,
        filestem="./outputs/tab",
    )
    LNS16.enable_triton(accumulator=args.accumulator)


def create_data_loaders(
    config: ExperimentConfig, args: argparse.Namespace, seed: int
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            ToDType(LNS16, device=args.device),
            DTypeNormalize(
                LNS16, mean=config.mean, std=config.std, device=args.device
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            ToDType(LNS16, device=args.device),
            DTypeNormalize(
                LNS16, mean=config.mean, std=config.std, device=args.device
            ),
        ]
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        config.dataset_type(
            root="./data", train=True, download=True, transform=train_transform
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
    )
    test_loader = torch.utils.data.DataLoader(
        config.dataset_type(
            root="./data", train=False, download=True, transform=test_transform
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def create_optimizer(model: nn.Module, args: argparse.Namespace):
    if args.optimizer == "sgd":
        return SGD(
            LNS16,
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    return Madam(
        LNS16,
        model.parameters(),
        lr=args.lr,
        beta=args.beta,
        p_scale=args.p_scale,
        g_bound=args.g_bound,
        use_pow=True,
    )


def create_scheduler(optimizer, epochs: int, initial_lr: float):
    warmup_epochs = min(5, epochs)
    linear_scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=initial_lr * 1e-3,
    )
    return lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[linear_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def log_configuration(
    logger: logging.Logger, config: ExperimentConfig, args: argparse.Namespace
) -> None:
    logger.info(
        "Training %s on %s with LNS "
        "(value_precision=%s, lookup_table=%s, accumulator=%s, "
        "accumulator_precision=%s, accumulator_lookup_table=%s) on device %s",
        config.model_name,
        config.dataset_name,
        args.prec,
        args.table,
        args.accumulator,
        args.accumulator_prec if args.accumulator else None,
        args.accumulator_table if args.accumulator else None,
        args.device,
    )
    if args.optimizer == "sgd":
        logger.info(
            "Hyperparameters: epochs=%s, batch_size=%s, lr=%s, momentum=%s, "
            "weight_decay=%s",
            args.epochs,
            args.batch_size,
            args.lr,
            args.momentum,
            args.weight_decay,
        )
    else:
        logger.info(
            "Hyperparameters: epochs=%s, batch_size=%s, lr=%s, beta=%s, p_scale=%s, g_bound=%s",
            args.epochs,
            args.batch_size,
            args.lr,
            args.beta,
            args.p_scale,
            args.g_bound,
        )


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    device: str,
    log_interval: int,
    logger: logging.Logger,
) -> dict[str, float | int]:
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    epoch_start = time.time()
    batch_start = epoch_start

    model.train()
    for batch_index, (data, target) in enumerate(loader, start=1):
        optimizer.zero_grad()
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            total_loss += loss.item() * target.size(0)
            predictions = output.to_float().argmax(dim=1)
            correct = (predictions == target).sum().item()
            total_correct += correct
            total_samples += target.size(0)

            if batch_index % log_interval == 0:
                batch_end = time.time()
                logger.info(
                    "Batch %s | Loss: %.4f | Correct: %s/%s | Elapsed Time: %.2fs",
                    batch_index,
                    loss.item(),
                    correct,
                    target.size(0),
                    batch_end - batch_start,
                )
                batch_start = batch_end

    scheduler.step()
    elapsed = time.time() - epoch_start
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples * 100,
        "correct": total_correct,
        "samples": total_samples,
        "time": elapsed,
    }


def test(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict[str, float | int]:
    test_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    samples = 0

    model.eval()
    start = time.time()
    with torch.no_grad():
        for data, target in loader:
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * target.size(0)
            logits = output.to_float()

            top1_correct += (logits.argmax(dim=1) == target).sum().item()
            _, top5 = logits.topk(5, dim=1, largest=True, sorted=True)
            top5_correct += (
                (top5 == target.unsqueeze(1)).any(dim=1).sum().item()
            )
            samples += target.size(0)

    return {
        "loss": test_loss / samples,
        "top1_acc": top1_correct / samples * 100,
        "top5_acc": top5_correct / samples * 100,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "samples": samples,
        "time": time.time() - start,
    }


def save_checkpoint(filename: str, model: nn.Module, optimizer, epoch=None) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if epoch is not None:
        checkpoint["epoch"] = epoch
    torch.save(checkpoint, filename)


def run_experiment(config: ExperimentConfig, seed: int = 42) -> None:
    args = parse_args(config)
    seed_everything(seed)
    logger, paths = configure_outputs(config.file_stem)
    configure_lns(args)
    train_loader, test_loader = create_data_loaders(config, args, seed)

    model = config.model_factory(config.num_classes, LNS16, args.device)
    criterion = nn.NLLLoss()
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args.epochs, args.lr)
    log_configuration(logger, config, args)

    history = []
    best_test_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            args.device,
            args.log_interval,
            logger,
        )
        logger.info(
            "Epoch [%s/%s] - Loss: %.4f | Correct: %s/%s (%.2f%%, %.2fs)",
            epoch,
            args.epochs,
            train_stats["loss"],
            train_stats["correct"],
            train_stats["samples"],
            train_stats["accuracy"],
            train_stats["time"],
        )

        test_stats = test(model, test_loader, criterion, args.device)
        logger.info(
            "Test - Loss: %.4f | Top1: %s/%s (%.2f%%) | "
            "Top5: %s/%s (%.2f%%) (%.2fs)",
            test_stats["loss"],
            test_stats["top1_correct"],
            test_stats["samples"],
            test_stats["top1_acc"],
            test_stats["top5_correct"],
            test_stats["samples"],
            test_stats["top5_acc"],
            test_stats["time"],
        )

        history.append({"epoch": epoch, "train": train_stats, "test": test_stats})
        with open(paths.history, "w") as history_file:
            json.dump(history, history_file, indent=2)

        if test_stats["top1_acc"] > best_test_acc:
            best_test_acc = test_stats["top1_acc"]
            save_checkpoint(paths.best_model, model, optimizer, epoch=epoch)
            logger.info(
                "New best model saved at epoch %s with Top1 Acc: %.2f%% "
                "(Top5: %.2f%%)",
                epoch,
                test_stats["top1_acc"],
                test_stats["top5_acc"],
            )

    logger.info("Training complete.")
    save_checkpoint(paths.final_model, model, optimizer)
    logger.info("Final model saved to %s", paths.final_model)
