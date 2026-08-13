from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
from torchdt.lns import LNS16, LNS32
from torchdt.optim import Madam, SGD, lr_scheduler
from torchdt.transforms import DTypeNormalize, ToDType
from torchvision import datasets, transforms

from float_cifar_train import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
)
from models.mobilenetv1 import MobileNetV1
from models.resnet18 import ResNet18
from models.shufflenetv2 import ShuffleNetV2


BatchNormState = list[
    tuple[nn.BatchNorm2d, torch.Tensor, torch.Tensor, torch.Tensor | None]
]


@dataclass(frozen=True)
class ArchitectureConfig:
    model_name: str
    dataset_type: type
    num_classes: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @property
    def dataset_name(self) -> str:
        return self.dataset_type.__name__


@dataclass(frozen=True)
class OutputPaths:
    batch: str
    epoch_summary: str
    final_model: str
    log: str


ARCHITECTURES = {
    "mobilenetv1": ArchitectureConfig(
        model_name="MobileNetV1",
        dataset_type=datasets.CIFAR10,
        num_classes=10,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    ),
    "resnet18": ArchitectureConfig(
        model_name="ResNet18",
        dataset_type=datasets.CIFAR100,
        num_classes=100,
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
    ),
    "shufflenetv2": ArchitectureConfig(
        model_name="ShuffleNetV2",
        dataset_type=datasets.CIFAR10,
        num_classes=10,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    ),
}


class EpochMetricStore:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

    def add(self, assumption: str, layer: str, name: str, value: float) -> None:
        if math.isfinite(value):
            self.values[assumption][layer][name].append(float(value))

    def as_summary(self) -> dict[str, dict[str, dict[str, dict[str, float | None]]]]:
        return {
            assumption: {
                layer: {
                    name: summarize(values)
                    for name, values in sorted(metrics.items())
                }
                for layer, metrics in sorted(layers.items())
            }
            for assumption, layers in sorted(self.values.items())
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CIFAR model in LNS16 and compute high-precision DRT "
            "assumption 1-4 diagnostics on the first batches of each epoch"
        )
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=ARCHITECTURES,
        default="resnet18",
        help="CNN architecture to train and probe (default: resnet18)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device used for LNS training and high-precision probes",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of LNS training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=128,
        help="Training batch size (default: 128)",
    )
    parser.add_argument(
        "--probe-batches",
        "--probe_batches",
        dest="probe_batches",
        type=int,
        default=10,
        help="Number of initial batches to probe per epoch (default: 10)",
    )
    parser.add_argument(
        "--optimizer",
        choices=("sgd", "madam"),
        default="sgd",
        help="LNS optimizer used for the real trajectory (default: sgd)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Initial learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--dampening",
        type=float,
        default=0.0,
        help="SGD momentum dampening (default: 0)",
    )
    parser.add_argument(
        "--weight-decay",
        "--weight_decay",
        dest="weight_decay",
        type=float,
        default=1e-4,
        help="SGD weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--beta", type=float, default=0.999, help="Madam beta (default: 0.999)"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8, help="Madam epsilon (default: 1e-8)"
    )
    parser.add_argument(
        "--p-scale",
        "--p_scale",
        dest="p_scale",
        type=float,
        default=3.0,
        help="Madam weight bound (default: 3)",
    )
    parser.add_argument(
        "--g-bound",
        "--g_bound",
        dest="g_bound",
        type=float,
        default=10.0,
        help="Madam normalized-gradient bound (default: 10)",
    )
    parser.add_argument(
        "--lns-prec",
        "--lns_prec",
        dest="lns_prec",
        type=int,
        default=7,
        help="Precision used by LNS16 (default: 7)",
    )
    parser.add_argument(
        "--table",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use LNS lookup tables (default: True)",
    )
    parser.add_argument(
        "--accumulator",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use LNS32 for reduction accumulators (default: False)",
    )
    parser.add_argument(
        "--accumulator-prec",
        "--accumulator_prec",
        dest="accumulator_prec",
        type=int,
        default=16,
        help="Precision for the optional LNS32 accumulator (default: 16)",
    )
    parser.add_argument(
        "--accumulator-table",
        "--accumulator_table",
        dest="accumulator_table",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use lookup tables for LNS32 accumulators (default: False)",
    )
    parser.add_argument(
        "--diagnostic-dtype",
        "--diagnostic_dtype",
        dest="diagnostic_dtype",
        choices=("float64", "float32"),
        default="float64",
        help="Floating dtype for DRT diagnostics (default: float64)",
    )
    parser.add_argument(
        "--probe-module-mode",
        "--probe_module_mode",
        dest="probe_module_mode",
        choices=("train", "eval"),
        default="train",
        help="BatchNorm/dropout mode for high-precision probes (default: train)",
    )
    parser.add_argument(
        "--jvp-probes",
        "--jvp_probes",
        dest="jvp_probes",
        type=int,
        default=4,
        help="Random input directions per probed block (default: 4)",
    )
    parser.add_argument(
        "--jvp-batch-size",
        "--jvp_batch_size",
        dest="jvp_batch_size",
        type=int,
        default=1,
        help="Examples from the current batch used for block JVPs (default: 2)",
    )
    parser.add_argument(
        "--max-jvp-blocks",
        "--max_jvp_blocks",
        dest="max_jvp_blocks",
        type=int,
        default=8,
        help=(
            "Maximum blocks to probe with JVPs; blocks are spread across depth "
            "(default: 8, use 0 for all)"
        ),
    )
    parser.add_argument(
        "--svd-max-elements",
        "--svd_max_elements",
        dest="svd_max_elements",
        type=int,
        default=4_000_000,
        help="Maximum elements in a weight matrix included in SVD probes",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        type=str,
        default="./data",
        help="Dataset directory (default: ./data)",
    )
    parser.add_argument(
        "--download",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Download CIFAR data if needed (default: True)",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default="./outputs/drt_assumptions",
        help="Output directory",
    )
    parser.add_argument(
        "--log-interval",
        "--log_interval",
        dest="log_interval",
        type=int,
        default=10,
        help="Batches between status messages (default: 10)",
    )
    parser.add_argument(
        "--save-final",
        "--save_final",
        dest="save_final",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Save the final LNS model and optimizer state (default: False)",
    )
    args = parser.parse_args()

    if not args.device.startswith("cuda"):
        parser.error("LNS training requires a CUDA device")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.probe_batches < 1:
        parser.error("--probe-batches must be at least 1")
    if args.log_interval < 1:
        parser.error("--log-interval must be at least 1")
    if args.jvp_probes < 1:
        parser.error("--jvp-probes must be at least 1")
    if args.jvp_batch_size < 1:
        parser.error("--jvp-batch-size must be at least 1")
    if args.max_jvp_blocks < 0:
        parser.error("--max-jvp-blocks must be nonnegative")
    if args.svd_max_elements < 1:
        parser.error("--svd-max-elements must be at least 1")
    if args.lr < 0.0:
        parser.error("--lr must be nonnegative")
    if args.momentum < 0.0 or args.dampening < 0.0:
        parser.error("--momentum and --dampening must be nonnegative")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay must be nonnegative")
    if not 0.0 < args.beta < 1.0:
        parser.error("--beta must be between 0 and 1")
    if args.eps <= 0.0 or args.p_scale <= 0.0 or args.g_bound <= 0.0:
        parser.error("--eps, --p-scale, and --g-bound must be positive")
    return args


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_outputs(args: argparse.Namespace) -> OutputPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, f"{args.arch}_{timestamp}")
    return OutputPaths(
        batch=f"{prefix}_batch.json",
        epoch_summary=f"{prefix}_epoch_summary.csv",
        final_model=f"{prefix}_final.pt",
        log=f"{prefix}.log",
    )


def write_epoch_summaries(
    path: str,
    epoch: int,
    config: ArchitectureConfig,
    args: argparse.Namespace,
    batches: int,
    learning_rate: float,
    train_stats: dict[str, float | int],
    summary: dict[str, dict[str, dict[str, dict[str, float | None]]]],
) -> None:
    fieldnames = [
        "epoch",
        "arch",
        "dataset",
        "optimizer",
        "lns_prec",
        "batches",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "assumption",
        "layer",
        "metric",
        "mean",
    ]
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for assumption, layers in summary.items():
            for layer, metrics in layers.items():
                for metric, statistics in metrics.items():
                    writer.writerow(
                        {
                            "epoch": epoch,
                            "arch": args.arch,
                            "dataset": config.dataset_name,
                            "optimizer": args.optimizer,
                            "lns_prec": args.lns_prec,
                            "batches": batches,
                            "learning_rate": learning_rate,
                            "train_loss": train_stats["loss"],
                            "train_accuracy": train_stats["accuracy"],
                            "assumption": assumption,
                            "layer": layer,
                            "metric": metric,
                            **statistics,
                        }
                    )


def configure_logger(path: str) -> logging.Logger:
    logger = logging.getLogger("DRT assumptions experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def configure_lns(args: argparse.Namespace) -> None:
    table_stem = os.path.join(args.output_dir, "tab")
    if args.accumulator:
        LNS32.set_prec(
            args.accumulator_prec,
            table=args.accumulator_table,
            table_device=args.device,
            filestem=table_stem,
        )
    LNS16.set_prec(
        args.lns_prec,
        table=args.table,
        table_device=args.device,
        filestem=table_stem,
    )
    LNS16.enable_triton(accumulator=args.accumulator)


def create_model(arch: str, num_classes: int, dtype: Any, device: str) -> nn.Module:
    kwargs = dict(
        num_classes=num_classes,
        cifar=True,
        madam=True,
        dtype=dtype,
        device=device,
    )
    if arch == "mobilenetv1":
        return MobileNetV1(**kwargs)
    if arch == "shufflenetv2":
        return ShuffleNetV2(in_channels=3, **kwargs)
    return ResNet18(**kwargs)


def create_data_loader(
    config: ArchitectureConfig,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> torch.utils.data.DataLoader:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            ToDType(LNS16, device=args.device),
            DTypeNormalize(LNS16, mean=config.mean, std=config.std, device=args.device),
        ]
    )
    dataset = config.dataset_type(
        root=args.data_dir,
        train=True,
        download=args.download,
        transform=train_transform,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )


def create_trajectory_optimizer(model: nn.Module, args: argparse.Namespace):
    if args.optimizer == "sgd":
        return SGD(
            LNS16,
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            dampening=args.dampening,
            weight_decay=args.weight_decay,
        )
    return Madam(
        LNS16,
        model.parameters(),
        lr=args.lr,
        beta=args.beta,
        eps=args.eps,
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


def diagnostic_torch_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.diagnostic_dtype == "float64":
        return torch.float64
    return torch.float32


def dequantize_to_tensor(value: Any) -> torch.Tensor:
    if hasattr(value, "to_float"):
        value = value.to_float()
    if isinstance(value, torch.Tensor):
        return value.detach()
    return torch.as_tensor(value)


def convert_state_value(
    source_value: Any,
    target_value: torch.Tensor,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    tensor = dequantize_to_tensor(source_value)
    if target_value.dtype.is_floating_point:
        return tensor.to(device=device, dtype=dtype)
    return tensor.to(device=device, dtype=target_value.dtype)


def sync_diagnostic_model(
    lns_model: nn.Module,
    diagnostic_model: nn.Module,
    dtype: torch.dtype,
    device: str,
) -> None:
    lns_state = lns_model.state_dict()
    diagnostic_template = diagnostic_model.state_dict()
    converted = {
        name: convert_state_value(lns_state[name], target, dtype, device)
        for name, target in diagnostic_template.items()
    }
    diagnostic_model.load_state_dict(converted, strict=True)


def to_diagnostic_batch(
    data: Any,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    return dequantize_to_tensor(data).to(device=device, dtype=dtype)


def summarize(values: list[float]) -> dict[str, float | None]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"mean": None}
    tensor = torch.tensor(finite, dtype=torch.float64)
    return {"mean": float(tensor.mean().item())}


def save_bn_state(module: nn.Module) -> BatchNormState:
    state = []
    for child in module.modules():
        if isinstance(child, nn.BatchNorm2d):
            state.append(
                (
                    child,
                    child.running_mean.detach().clone(),
                    child.running_var.detach().clone(),
                    child.num_batches_tracked.detach().clone()
                    if child.num_batches_tracked is not None
                    else None,
                )
            )
    return state


def restore_bn_state(state: BatchNormState) -> None:
    for bn, mean, var, tracked in state:
        bn.running_mean.copy_(mean)
        bn.running_var.copy_(var)
        if tracked is not None and bn.num_batches_tracked is not None:
            bn.num_batches_tracked.copy_(tracked)


def scalar_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value.detach().to(dtype=torch.float64))


def scalar_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: float = 1e-30,
) -> float:
    return float(
        (
            numerator.to(dtype=torch.float64)
            / denominator.to(dtype=torch.float64).clamp_min(eps)
        ).item()
    )


def register_activation_hooks(
    model: nn.Module,
    norm_squares: dict[str, list[float]],
) -> list[Any]:
    handles = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.ReLU):
            continue

        def hook(_module, inputs, output, name=module_name):
            if not inputs:
                return
            activation_input = inputs[0].detach()
            activation_output = output.detach()
            totals = norm_squares.setdefault(name, [0.0, 0.0])
            totals[0] += float(
                activation_input.to(dtype=torch.float64).square().sum().item()
            )
            totals[1] += float(
                activation_output.to(dtype=torch.float64).square().sum().item()
            )

        handles.append(module.register_forward_hook(hook))
    return handles


def collect_activation_transmission(
    model: nn.Module,
    data: torch.Tensor,
) -> dict[str, float]:
    norm_squares: dict[str, list[float]] = {}
    handles = register_activation_hooks(model, norm_squares)
    bn_state = save_bn_state(model)
    try:
        with torch.no_grad():
            _ = model(data)
    finally:
        restore_bn_state(bn_state)
        for handle in handles:
            handle.remove()
    return {
        name: math.sqrt(output_square_sum)
        / max(math.sqrt(input_square_sum), 1e-30)
        for name, (input_square_sum, output_square_sum) in norm_squares.items()
    }


def collect_weight_conditioning(
    model: nn.Module,
    max_elements: int,
) -> dict[str, float]:
    results: dict[str, float] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.detach().reshape(module.out_channels, -1)
        elif isinstance(module, nn.Linear):
            weight = module.weight.detach()
        else:
            continue

        if weight.numel() > max_elements:
            continue
        singular_values = torch.linalg.svdvals(
            weight.to(dtype=torch.float64, device="cpu")
        )
        positive = singular_values[singular_values > 0]
        if positive.numel() == 0:
            continue

        sigma_max = positive.max()
        sigma_min = positive.min()
        results[name] = float(
            (sigma_max / sigma_min.clamp_min(1e-30)).item()
        )
    return results


def record_add_merge(
    results: dict[str, dict[str, float]],
    name: str,
    branch_a: torch.Tensor,
    branch_b: torch.Tensor,
    merged: torch.Tensor,
) -> None:
    norm_a = scalar_norm(branch_a)
    norm_b = scalar_norm(branch_b)
    norm_z = torch.sqrt(norm_a.square() + norm_b.square())
    results[name] = {
        "merge_ratio": scalar_ratio(scalar_norm(merged), norm_z),
        "branch_b_over_branch_a": scalar_ratio(norm_b, norm_a),
    }


def record_concat_merge(
    results: dict[str, dict[str, float]],
    name: str,
    branch_a: torch.Tensor,
    branch_b: torch.Tensor,
    merged: torch.Tensor,
) -> None:
    norm_a = scalar_norm(branch_a)
    norm_b = scalar_norm(branch_b)
    norm_z = torch.sqrt(norm_a.square() + norm_b.square())
    results[name] = {
        "merge_ratio": scalar_ratio(scalar_norm(merged), norm_z),
    }


def register_merge_hooks(
    model: nn.Module,
    arch: str,
    results: dict[str, dict[str, float]],
) -> list[Any]:
    handles = []
    if arch == "resnet18":
        from models.resnet18 import BasicBlock

        for module_name, block in model.named_modules():
            if not isinstance(block, BasicBlock):
                continue

            def hook(module, inputs, _output, name=module_name):
                if not inputs:
                    return
                x = inputs[0].detach()
                bn_state = save_bn_state(module)
                with torch.no_grad():
                    residual = module.conv1(x)
                    residual = module.bn1(residual)
                    residual = module.relu(residual)
                    residual = module.conv2(residual)
                    residual = module.bn2(residual)
                    shortcut = module.shortcut(x)
                    merged = shortcut + residual
                restore_bn_state(bn_state)
                record_add_merge(results, name, shortcut, residual, merged)

            handles.append(block.register_forward_hook(hook))
    elif arch == "shufflenetv2":
        from models.shufflenetv2 import ShuffleV2Block, channel_shuffle

        for module_name, block in model.named_modules():
            if not isinstance(block, ShuffleV2Block):
                continue

            def hook(module, inputs, _output, name=module_name):
                if not inputs:
                    return
                x = inputs[0].detach()
                bn_state = save_bn_state(module)
                with torch.no_grad():
                    if module.stride == 1:
                        branch_a, branch_input = x.chunk(2, dim=1)
                        branch_b = module.branch2(branch_input)
                    else:
                        branch_a = module.branch1(x)
                        branch_b = module.branch2(x)
                    merged = channel_shuffle(torch.cat((branch_a, branch_b), dim=1))
                restore_bn_state(bn_state)
                record_concat_merge(results, name, branch_a, branch_b, merged)

            handles.append(block.register_forward_hook(hook))
    return handles


def collect_merge_transmission(
    model: nn.Module,
    arch: str,
    data: torch.Tensor,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    handles = register_merge_hooks(model, arch, results)
    bn_state = save_bn_state(model)
    try:
        with torch.no_grad():
            _ = model(data)
    finally:
        restore_bn_state(bn_state)
        for handle in handles:
            handle.remove()

    return results


def candidate_blocks(model: nn.Module, arch: str) -> list[tuple[str, nn.Module]]:
    if arch == "resnet18":
        from models.resnet18 import BasicBlock

        return [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, BasicBlock)
        ]
    if arch == "mobilenetv1":
        from models.mobilenetv1 import DepthwiseSeparableConv

        named = dict(model.named_modules())
        blocks = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, DepthwiseSeparableConv)
        ]
        return ([("stem", named["stem"])] if "stem" in named else []) + blocks
    if arch == "shufflenetv2":
        from models.shufflenetv2 import ShuffleV2Block

        named = dict(model.named_modules())
        prefix = [("conv1", named["conv1"])] if "conv1" in named else []
        blocks = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, ShuffleV2Block)
        ]
        suffix = [("conv5", named["conv5"])] if "conv5" in named else []
        return prefix + blocks + suffix
    return []


def select_blocks(
    blocks: list[tuple[str, nn.Module]], max_blocks: int
) -> list[tuple[str, nn.Module]]:
    if max_blocks == 0 or len(blocks) <= max_blocks:
        return blocks
    indices = (
        torch.linspace(0, len(blocks) - 1, max_blocks)
        .round()
        .long()
        .tolist()
    )
    return [blocks[index] for index in sorted(set(indices))]


def capture_block_inputs(
    model: nn.Module,
    blocks: list[tuple[str, nn.Module]],
    data: torch.Tensor,
) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for name, module in blocks:

        def pre_hook(_module, inputs, block_name=name):
            if block_name not in captured and inputs:
                captured[block_name] = inputs[0].detach().clone()

        handles.append(module.register_forward_pre_hook(pre_hook))

    bn_state = save_bn_state(model)
    with torch.no_grad():
        _ = model(data)
    restore_bn_state(bn_state)
    for handle in handles:
        handle.remove()
    return captured


def make_orthonormal_directions(x: torch.Tensor, count: int) -> list[torch.Tensor]:
    directions: list[torch.Tensor] = []
    flat_basis: list[torch.Tensor] = []
    flat_size = x.numel()
    for _ in range(count):
        flat = torch.randn(flat_size, device=x.device, dtype=x.dtype)
        for basis_vector in flat_basis:
            flat = flat - torch.dot(flat, basis_vector) * basis_vector
        norm = torch.linalg.vector_norm(flat)
        if norm <= 0:
            continue
        flat = flat / norm
        flat_basis.append(flat)
        directions.append(flat.reshape_as(x))
    return directions


def response_condition_number(response_matrix: torch.Tensor) -> float | None:
    if response_matrix.numel() == 0 or not torch.isfinite(response_matrix).all():
        return None

    response_matrix = response_matrix.to(dtype=torch.float64, device="cpu")
    singular_values = torch.linalg.svdvals(response_matrix)
    positive = singular_values[singular_values > 0]
    if positive.numel() == 0:
        return None

    sigma_max = positive.max()
    sigma_min = positive.min()
    return float((sigma_max / sigma_min.clamp_min(1e-30)).item())


def jvp_response_matrix(
    module: nn.Module,
    x: torch.Tensor,
    directions: list[torch.Tensor],
) -> torch.Tensor:
    responses = []
    bn_state = save_bn_state(module)
    training_state = [(child, child.training) for child in module.modules()]

    def fn(inp):
        return module(inp)

    module.eval()
    try:
        for direction in directions:
            restore_bn_state(bn_state)
            _, jvp = torch.autograd.functional.jvp(
                fn,
                (x,),
                (direction,),
                create_graph=False,
                strict=False,
            )
            responses.append(jvp.detach().flatten().to(dtype=torch.float64))
    finally:
        restore_bn_state(bn_state)
        for child, training in training_state:
            child.training = training
    return torch.stack(responses, dim=1)


def collect_block_jacobian_conditioning(
    model: nn.Module,
    arch: str,
    data: torch.Tensor,
    n_probes: int,
    jvp_batch_size: int,
    max_blocks: int,
) -> dict[str, float]:
    blocks = select_blocks(candidate_blocks(model, arch), max_blocks)
    captured = capture_block_inputs(model, blocks, data)
    results: dict[str, float] = {}

    for name, module in blocks:
        x = captured.get(name)
        if x is None:
            continue
        x = x[: min(jvp_batch_size, x.shape[0])].detach().requires_grad_(True)
        directions = make_orthonormal_directions(x, n_probes)
        if not directions:
            continue

        response_matrix = jvp_response_matrix(module, x, directions)
        condition_number = response_condition_number(response_matrix)
        if condition_number is not None:
            results[name] = condition_number
    return results


def update_epoch_store(
    store: EpochMetricStore,
    activation_transmission: dict[str, float],
    weight_conditioning: dict[str, float],
    merge_transmission: dict[str, dict[str, float]],
    block_conditioning: dict[str, float],
) -> None:
    for layer, value in activation_transmission.items():
        store.add(
            "activation_transmission",
            layer,
            "relu_norm_transmission",
            value,
        )

    for layer, value in weight_conditioning.items():
        store.add("weight_conditioning", layer, "condition_number", value)

    for layer, metrics in merge_transmission.items():
        for metric_name, value in metrics.items():
            store.add("merge_transmission", layer, metric_name, value)

    for layer, value in block_conditioning.items():
        store.add(
            "block_jacobian_conditioning",
            layer,
            "condition_number",
            value,
        )


def probe_batch(
    lns_model: nn.Module,
    diagnostic_model: nn.Module,
    arch: str,
    data: Any,
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> dict[str, Any]:
    sync_diagnostic_model(lns_model, diagnostic_model, dtype, args.device)
    diagnostic_model.train(args.probe_module_mode == "train")
    diagnostic_data = to_diagnostic_batch(data, dtype, args.device)

    activation_transmission = collect_activation_transmission(
        diagnostic_model,
        diagnostic_data,
    )
    weight_conditioning = collect_weight_conditioning(
        diagnostic_model,
        args.svd_max_elements,
    )
    merge_transmission = collect_merge_transmission(
        diagnostic_model,
        arch,
        diagnostic_data,
    )
    block_conditioning = collect_block_jacobian_conditioning(
        diagnostic_model,
        arch,
        diagnostic_data,
        args.jvp_probes,
        args.jvp_batch_size,
        args.max_jvp_blocks,
    )
    return {
        "activation_transmission": activation_transmission,
        "weight_conditioning_kernel_flat_proxy": weight_conditioning,
        "merge_transmission": merge_transmission,
        "block_jacobian_conditioning_random_subspace": block_conditioning,
    }


@torch.no_grad()
def prediction_count(output: Any, target: torch.Tensor) -> int:
    logits = output.to_float() if hasattr(output, "to_float") else output
    return int((logits.argmax(dim=1) == target).sum().item())


def save_final_model(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer,
    scheduler,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "arguments": vars(args),
    }
    torch.save(payload, path)


def run_experiment() -> None:
    args = parse_args()
    config = ARCHITECTURES[args.arch]
    seed_everything(args.seed)
    paths = configure_outputs(args)
    logger = configure_logger(paths.log)
    os.makedirs(args.output_dir, exist_ok=True)

    configure_lns(args)
    diagnostic_dtype = diagnostic_torch_dtype(args)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = create_data_loader(config, args, loader_generator)

    lns_model = create_model(args.arch, config.num_classes, LNS16, args.device)
    diagnostic_model = create_model(
        args.arch, config.num_classes, diagnostic_dtype, args.device
    )
    criterion = nn.NLLLoss()
    optimizer = create_trajectory_optimizer(lns_model, args)
    scheduler = create_scheduler(optimizer, args.epochs, args.lr)
    logger.info(
        "Training %s on %s in LNS16 | optimizer=%s | lns_prec=%s | "
        "diagnostics=%s | epochs=%s",
        config.model_name,
        config.dataset_name,
        args.optimizer,
        args.lns_prec,
        args.diagnostic_dtype,
        args.epochs,
    )
    logger.info(
        "DRT probes | probe_batches=%s | svd_max_elements=%s | jvp_probes=%s | "
        "jvp_batch_size=%s | max_jvp_blocks=%s | "
        "probe_module_mode=%s",
        args.probe_batches,
        args.svd_max_elements,
        args.jvp_probes,
        args.jvp_batch_size,
        args.max_jvp_blocks,
        args.probe_module_mode,
    )
    if args.optimizer == "sgd":
        logger.info(
            "SGD hyperparameters | initial_lr=%s | momentum=%s | "
            "dampening=%s | weight_decay=%s",
            args.lr,
            args.momentum,
            args.dampening,
            args.weight_decay,
        )
    else:
        logger.info(
            "Madam hyperparameters | initial_lr=%s | beta=%s | eps=%s | "
            "p_scale=%s | g_bound=%s",
            args.lr,
            args.beta,
            args.eps,
            args.p_scale,
            args.g_bound,
        )

    with open(paths.batch, "w", buffering=1) as batch_file:
        batch_file.write("{\n")
        for epoch in range(1, args.epochs + 1):
            if epoch > 1:
                batch_file.write(",\n")
            batch_file.write(f'  "epoch_{epoch:03d}": [')
            first_json_row = True
            epoch_start = time.time()
            batch_start = epoch_start
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            probed_batches = 0
            epoch_store = EpochMetricStore()
            learning_rate = float(optimizer.param_groups[0]["lr"])

            lns_model.train()
            for batch_idx, (data, target) in enumerate(train_loader, start=1):

                probe = None
                if batch_idx <= args.probe_batches:
                    probe = probe_batch(
                        lns_model,
                        diagnostic_model,
                        args.arch,
                        data,
                        args,
                        diagnostic_dtype,
                    )
                    update_epoch_store(
                        epoch_store,
                        probe["activation_transmission"],
                        probe["weight_conditioning_kernel_flat_proxy"],
                        probe["merge_transmission"],
                        probe["block_jacobian_conditioning_random_subspace"],
                    )
                    probed_batches += 1

                optimizer.zero_grad()
                target = target.to(args.device)
                output = lns_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                batch_samples = int(target.size(0))
                batch_correct = prediction_count(output, target)
                total_loss += float(loss.item()) * batch_samples
                total_correct += batch_correct
                total_samples += batch_samples

                if probe is not None:
                    batch_row = {
                        "batch_idx": batch_idx,
                        "optimizer": args.optimizer,
                        **probe,
                    }
                    if not first_json_row:
                        batch_file.write(",")
                    batch_file.write(
                        "\n    " + json.dumps(batch_row, allow_nan=False)
                    )
                    first_json_row = False

                if batch_idx % args.log_interval == 0:
                    batch_end = time.time()
                    logger.info(
                        "Epoch [%s/%s] | Batch %s | Loss: %.4f | "
                        "Correct: %s/%s | Elapsed Time: %.2fs",
                        epoch,
                        args.epochs,
                        batch_idx,
                        float(loss.item()),
                        batch_correct,
                        batch_samples,
                        batch_end - batch_start,
                    )
                    batch_start = batch_end

            batch_file.write("\n  ]")
            if total_samples == 0:
                raise RuntimeError("the training loader produced no complete batches")

            scheduler.step()
            train_stats = {
                "loss": total_loss / total_samples,
                "accuracy": total_correct / total_samples * 100.0,
                "correct": total_correct,
                "samples": total_samples,
                "time": time.time() - epoch_start,
            }
            write_epoch_summaries(
                paths.epoch_summary,
                epoch,
                config,
                args,
                probed_batches,
                learning_rate,
                train_stats,
                epoch_store.as_summary(),
            )
            logger.info(
                "Epoch [%s/%s] complete | Loss: %.4f | Acc: %.2f%% | Time: %.2fs",
                epoch,
                args.epochs,
                train_stats["loss"],
                train_stats["accuracy"],
                train_stats["time"],
            )
        batch_file.write("\n}\n")

    if args.save_final:
        save_final_model(
            paths.final_model,
            args.epochs,
            lns_model,
            optimizer,
            scheduler,
            args,
        )
        logger.info("Final LNS model saved to %s", paths.final_model)

    logger.info("Batch diagnostics: %s", paths.batch)
    logger.info("Epoch summaries: %s", paths.epoch_summary)


if __name__ == "__main__":
    run_experiment()
