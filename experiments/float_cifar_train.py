"""Shared training loop for the CIFAR floating-point experiments."""

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

FLOAT_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

ModelFactory = Callable[[int, torch.dtype, str], nn.Module]


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
        description=f"Train {config.model_name} on {config.dataset_name} "
        "in floating point"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=FLOAT_DTYPES,
        help="Floating-point dtype to use (default: float32)",
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


def create_data_loaders(
    config: ExperimentConfig,
    args: argparse.Namespace,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    to_dtype = transforms.Lambda(
        lambda tensor: tensor.to(device=args.device, dtype=dtype)
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            to_dtype,
            transforms.Normalize(mean=config.mean, std=config.std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            to_dtype,
            transforms.Normalize(mean=config.mean, std=config.std),
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


class Madam(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        lr=0.01,
        beta=0.999,
        eps=1e-8,
        g_bound=10.0,
        use_pow=False,
        *,
        maximize=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if g_bound <= 0.0:
            raise ValueError(f"Invalid gradient bound: {g_bound}")

        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            g_bound=g_bound,
            use_pow=use_pow,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta = group["beta"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Madam does not support sparse gradients")

                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(param)

                state["step"] += 1
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sq.mul_(beta).addcmul_(grad, grad, value=1.0 - beta)

                bias_correction = 1.0 - beta ** state["step"]
                normalized = grad / torch.sqrt(
                    exp_avg_sq / bias_correction + group["eps"]
                )
                clipped = normalized.clamp(
                    min=-group["g_bound"], max=group["g_bound"]
                )
                delta = group["lr"] * clipped * torch.sign(param)
                if not group["maximize"]:
                    delta.neg_()

                if group["use_pow"]:
                    updated = param * torch.exp(delta)
                else:
                    updated = param * (1.0 + delta)
                param.copy_(updated)

        return loss


def create_optimizer(model: nn.Module, args: argparse.Namespace):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    return Madam(
        model.parameters(), lr=args.lr, beta=args.beta, use_pow=True
    )


def create_scheduler(optimizer, epochs: int):
    warmup_epochs = min(5, epochs)
    linear_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs - warmup_epochs),
        eta_min=1e-4,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[linear_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def log_configuration(
    logger: logging.Logger, config: ExperimentConfig, args: argparse.Namespace
) -> None:
    logger.info(
        "Training %s on %s with floating point (dtype=%s) on device %s",
        config.model_name,
        config.dataset_name,
        args.dtype,
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
            "Hyperparameters: epochs=%s, batch_size=%s, lr=%s, beta=%s",
            args.epochs,
            args.batch_size,
            args.lr,
            args.beta,
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
            predictions = output.argmax(dim=1)
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

            top1_correct += (output.argmax(dim=1) == target).sum().item()
            _, top5 = output.topk(5, dim=1, largest=True, sorted=True)
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
    dtype = FLOAT_DTYPES[args.dtype]
    train_loader, test_loader = create_data_loaders(config, args, dtype, seed)

    model = config.model_factory(config.num_classes, dtype, args.device)
    criterion = nn.NLLLoss()
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args.epochs)
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
