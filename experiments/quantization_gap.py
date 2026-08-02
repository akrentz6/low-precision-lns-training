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
from typing import Any

import torch
import torch.nn as nn
from torchdt.lns import LNS16, LNS32
from torchdt.optim import Madam as LNSMadam
from torchdt.optim import SGD as LNSSGD
from torchvision import datasets, transforms

from float_cifar_train import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    CIFAR100_MEAN,
    CIFAR100_STD,
    Madam,
    create_scheduler,
)
from models.mobilenetv1 import MobileNetV1
from models.resnet18 import ResNet18
from models.shufflenetv2 import ShuffleNetV2


FLOAT_DTYPES = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
LNS_DTYPES = {
    "lns16": LNS16,
    "lns32": LNS32,
}
SUPPORTED_FORMATS = ("float32", "float16", "bfloat16", "lns16", "lns32")
SUPPORTED_OPTIMIZERS = ("sgd", "madam")
SUMMARY_METRICS = (
    "relative_l2_error",
    "cosine",
    "angle_degrees",
    "magnitude_ratio",
    "log_magnitude_ratio",
    "sign_disagreement",
    "zero_update_fraction",
)


@dataclass(frozen=True)
class ArchitectureConfig:
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
    log: str


ARCHITECTURES = {
    "mobilenetv1": ArchitectureConfig(
        dataset_type=datasets.CIFAR10,
        num_classes=10,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    ),
    "shufflenetv2": ArchitectureConfig(
        dataset_type=datasets.CIFAR10,
        num_classes=10,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    ),
    "resnet18": ArchitectureConfig(
        dataset_type=datasets.CIFAR100,
        num_classes=100,
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
    ),
}


def comma_separated_choices(value: str, choices: tuple[str, ...], argument_name: str) -> list[str]:
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(values) - set(choices))
    if not values:
        raise argparse.ArgumentTypeError(f"{argument_name} cannot be empty")
    if invalid:
        raise argparse.ArgumentTypeError(
            f"invalid {argument_name}: {', '.join(invalid)}; "
            f"choose from {', '.join(choices)}"
        )
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe low precision quantization gaps along FP64 SGD and Madam training"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="mobilenetv1",
        choices=ARCHITECTURES,
        help="CNN architecture to train (default: mobilenetv1)",
    )
    parser.add_argument(
        "--formats",
        type=lambda value: comma_separated_choices(
            value, SUPPORTED_FORMATS, "formats"
        ),
        default=comma_separated_choices(
            "float16,bfloat16,lns16", SUPPORTED_FORMATS, "formats"
        ),
        help="Comma-separated shadow formats (default: float16,bfloat16,lns16)",
    )
    parser.add_argument(
        "--optimizers",
        type=lambda value: comma_separated_choices(
            value, SUPPORTED_OPTIMIZERS, "optimizers"
        ),
        default=comma_separated_choices(
            "sgd,madam", SUPPORTED_OPTIMIZERS, "optimizers"
        ),
        help="Comma-separated FP64 trajectory optimizers (default: sgd,madam)",
    )
    parser.add_argument(
        "--lns-prec",
        type=int,
        default=7,
        help="Precision used by requested LNS formats (default: 7)",
    )
    parser.add_argument(
        "--table",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use LNS lookup tables (default: True)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device used for training and probes (default: cuda:0)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of FP64 trajectory epochs (default: 100)",
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
        "--max-batches",
        "--max_batches",
        dest="max_batches",
        type=int,
        default=None,
        help="Maximum training batches per epoch, for debugging",
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
        "--lr", type=float, default=0.1, help="Initial learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)"
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
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Dataset directory (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/update_gap_full",
        help="Output directory (default: ./outputs/update_gap_full)",
    )
    parser.add_argument(
        "--evaluate",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Evaluate the FP64 model after each epoch (default: True)",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be at least 1")
    if args.log_interval < 1:
        parser.error("--log-interval must be at least 1")
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
    if any(name.startswith("lns") for name in args.formats):
        if not args.device.startswith("cuda"):
            parser.error("LNS probes require a CUDA device")
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
        log=f"{prefix}.log",
    )


def open_epoch_json_section(path: str, epoch: int):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        output_file = open(path, "w+", buffering=1)
        output_file.write("{\n")
    else:
        output_file = open(path, "r+", buffering=1)
        output_file.seek(0, os.SEEK_END)
        position = output_file.tell()
        while position > 0:
            position -= 1
            output_file.seek(position)
            if not output_file.read(1).isspace():
                break
        output_file.seek(position)
        if output_file.read(1) != "}":
            output_file.close()
            raise RuntimeError(f"Cannot append to incomplete JSON file: {path}")
        output_file.seek(position)
        output_file.truncate()
        output_file.write(",\n")
    section_name = f"epoch_{epoch:03d}"
    output_file.write(f"  {json.dumps(section_name)}: [")
    return output_file


def close_epoch_json_section(output_file) -> None:
    output_file.write("\n  ]\n}\n")
    output_file.close()


def configure_logger(path: str) -> logging.Logger:
    logger = logging.getLogger("Quantization gap experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def configure_lns(args: argparse.Namespace) -> None:
    table_stem = os.path.join(args.output_dir, "tab")
    if "lns32" in args.formats:
        LNS32.set_prec(
            args.lns_prec,
            table=args.table,
            table_device=args.device,
            filestem=table_stem,
        )
        LNS32.enable_triton()
    if "lns16" in args.formats:
        LNS16.set_prec(
            args.lns_prec,
            table=args.table,
            table_device=args.device,
            filestem=table_stem,
        )
        LNS16.enable_triton()


def create_model(
    arch: str, num_classes: int, device: str
) -> nn.Module:
    kwargs = dict(
        num_classes=num_classes,
        cifar=True,
        madam=True,
        dtype=torch.float64,
        device=device,
    )
    if arch == "mobilenetv1":
        return MobileNetV1(**kwargs)
    if arch == "shufflenetv2":
        return ShuffleNetV2(in_channels=3, **kwargs)
    return ResNet18(**kwargs)


def create_data_loaders(
    config: ArchitectureConfig,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    to_fp64 = transforms.Lambda(
        lambda tensor: tensor.to(device=args.device, dtype=torch.float64)
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            to_fp64,
            transforms.Normalize(mean=config.mean, std=config.std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            to_fp64,
            transforms.Normalize(mean=config.mean, std=config.std),
        ]
    )
    train_loader = torch.utils.data.DataLoader(
        config.dataset_type(
            root=args.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    test_loader = torch.utils.data.DataLoader(
        config.dataset_type(
            root=args.data_dir,
            train=False,
            download=True,
            transform=test_transform,
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def create_trajectory_optimizer(
    optimizer_name: str, model: nn.Module, args: argparse.Namespace
):
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            dampening=args.dampening,
            weight_decay=args.weight_decay,
        )
    return Madam(
        model.parameters(),
        lr=args.lr,
        beta=args.beta,
        eps=args.eps,
        p_scale=args.p_scale,
        g_bound=args.g_bound,
        use_pow=True,
    )


def quantize(value: torch.Tensor | float, format_name: str, device: str):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device)
    else:
        tensor = torch.tensor(value, dtype=torch.float64, device=device)
    if format_name in FLOAT_DTYPES:
        return tensor.to(dtype=FLOAT_DTYPES[format_name])
    return LNS_DTYPES[format_name](tensor)


def dequantize(value) -> torch.Tensor:
    if hasattr(value, "to_float"):
        value = value.to_float()
    return value.detach().to(dtype=torch.float64)


def clone_shadow_value(value, device: str):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device).clone()
    return value


def create_shadow_optimizer(
    optimizer_name: str,
    format_name: str,
    parameters: list[torch.Tensor],
    args: argparse.Namespace,
    lr: float,
):
    if format_name in LNS_DTYPES:
        dtype = LNS_DTYPES[format_name]
        if optimizer_name == "sgd":
            return LNSSGD(
                dtype,
                parameters,
                lr=lr,
                momentum=args.momentum,
                dampening=args.dampening,
                weight_decay=args.weight_decay,
            )
        return LNSMadam(
            dtype,
            parameters,
            lr=lr,
            beta=args.beta,
            eps=args.eps,
            p_scale=args.p_scale,
            g_bound=args.g_bound,
            use_pow=True,
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=args.momentum,
            dampening=args.dampening,
            weight_decay=args.weight_decay,
        )
    return Madam(
        parameters,
        lr=lr,
        beta=args.beta,
        eps=args.eps,
        p_scale=args.p_scale,
        g_bound=args.g_bound,
        use_pow=True,
    )


@torch.no_grad()
def shadow_step(
    optimizer_name: str,
    format_name: str,
    weights_before: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
    parameters_after: dict[str, nn.Parameter],
    shadow_state: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    lr: float,
) -> dict[str, float | None]:
    names = list(gradients)
    shadow_parameters = []
    shadow_weights_before = {}
    for name in names:
        value = quantize(weights_before[name], format_name, args.device)
        if format_name in FLOAT_DTYPES:
            value = nn.Parameter(value, requires_grad=False)
        value.grad = quantize(gradients[name], format_name, args.device)
        shadow_parameters.append(value)
        shadow_weights_before[name] = dequantize(value).clone()

    optimizer = create_shadow_optimizer(
        optimizer_name,
        format_name,
        shadow_parameters,
        args,
        lr,
    )
    for name, parameter in zip(names, shadow_parameters):
        saved_state = shadow_state.get(name, {})
        optimizer.state[parameter].update(
            {
                key: clone_shadow_value(value, args.device)
                for key, value in saved_state.items()
            }
        )

    optimizer.step()
    totals = torch.zeros(7, dtype=torch.float64, device=args.device)
    new_state = dict(shadow_state)
    for name, parameter in zip(names, shadow_parameters):
        quantized_update = dequantize(parameter) - shadow_weights_before[name]
        reference_update = parameters_after[name].detach() - weights_before[name]
        totals += metric_totals_tensor(quantized_update, reference_update)
        new_state[name] = {
            key: clone_shadow_value(value, args.device)
            for key, value in optimizer.state[parameter].items()
        }
    shadow_state.clear()
    shadow_state.update(new_state)
    return metrics_from_totals(totals_from_tensor(totals))


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def metrics_from_totals(totals: dict[str, float]) -> dict[str, float | None]:
    ref_norm = math.sqrt(max(0.0, totals["ref_sq"]))
    quantized_norm = math.sqrt(max(0.0, totals["quantized_sq"]))
    error_norm = math.sqrt(max(0.0, totals["error_sq"]))

    if ref_norm == 0.0:
        relative_l2_error = 0.0 if error_norm == 0.0 else math.inf
        magnitude_ratio = 1.0 if quantized_norm == 0.0 else math.inf
    else:
        relative_l2_error = error_norm / ref_norm
        magnitude_ratio = quantized_norm / ref_norm

    if ref_norm == 0.0 and quantized_norm == 0.0:
        cosine = 1.0
    elif ref_norm == 0.0 or quantized_norm == 0.0:
        cosine = 0.0
    else:
        cosine = totals["dot"] / (ref_norm * quantized_norm)
        cosine = max(-1.0, min(1.0, cosine))
    angle_degrees = math.degrees(math.acos(cosine))

    if magnitude_ratio == 0.0:
        log_magnitude_ratio = -math.inf
    elif math.isfinite(magnitude_ratio):
        log_magnitude_ratio = math.log(magnitude_ratio)
    else:
        log_magnitude_ratio = math.inf

    numel = totals["numel"]
    return {
        "relative_l2_error": finite_or_none(relative_l2_error),
        "cosine": finite_or_none(cosine),
        "angle_degrees": finite_or_none(angle_degrees),
        "magnitude_ratio": finite_or_none(magnitude_ratio),
        "log_magnitude_ratio": finite_or_none(log_magnitude_ratio),
        "sign_disagreement": totals["sign_disagree"] / numel,
        "zero_update_fraction": totals["zero"] / numel,
    }


def metric_totals_tensor(
    quantized_update: torch.Tensor,
    reference_update: torch.Tensor,
) -> torch.Tensor:
    quantized = quantized_update.reshape(-1)
    reference = reference_update.reshape(-1)
    difference = quantized - reference
    return torch.stack(
        [
            torch.sum(quantized * reference),
            torch.sum(quantized * quantized),
            torch.sum(reference * reference),
            torch.sum(difference * difference),
            torch.sum(
                (quantized != 0)
                & (torch.sign(quantized) != torch.sign(reference))
            ).to(torch.float64),
            torch.sum(quantized == 0).to(torch.float64),
            torch.tensor(
                reference.numel(), dtype=torch.float64, device=reference.device
            ),
        ]
    )


def totals_from_tensor(values: torch.Tensor) -> dict[str, float]:
    values_list = values.cpu().tolist()
    return {
        "dot": values_list[0],
        "quantized_sq": values_list[1],
        "ref_sq": values_list[2],
        "error_sq": values_list[3],
        "sign_disagree": values_list[4],
        "zero": values_list[5],
        "numel": values_list[6],
    }



def summarize(values: list[float | None]) -> dict[str, float | None]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    tensor = torch.tensor(finite, dtype=torch.float64)
    return {
        "mean": tensor.mean().item(),
        "median": torch.quantile(tensor, 0.5).item(),
        "p10": torch.quantile(tensor, 0.1).item(),
        "p90": torch.quantile(tensor, 0.9).item(),
    }


def epoch_summary_fieldnames() -> list[str]:
    fields = [
        "epoch",
        "optimizer",
        "format",
        "batches",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_top1_accuracy",
        "test_top5_accuracy",
    ]
    for metric in SUMMARY_METRICS:
        fields.extend(
            f"{metric}_{statistic}"
            for statistic in ("mean", "median", "p10", "p90")
        )
    return fields


def write_epoch_summaries(
    path: str,
    epoch: int,
    args: argparse.Namespace,
    learning_rates: dict[str, float],
    epoch_metrics: dict[str, dict[str, list[float | None]]],
    train_stats: dict[str, dict[str, float]],
    test_stats: dict[str, dict[str, float | None]],
) -> None:
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    fieldnames = epoch_summary_fieldnames()
    with open(path, "a", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for key, metric_values in epoch_metrics.items():
            optimizer_name, format_name = key.split(":", maxsplit=1)
            row: dict[str, Any] = {
                "epoch": epoch,
                "optimizer": optimizer_name,
                "format": format_name,
                "batches": len(metric_values[SUMMARY_METRICS[0]]),
                "learning_rate": learning_rates[optimizer_name],
                "train_loss": train_stats[optimizer_name]["loss"],
                "train_accuracy": train_stats[optimizer_name]["accuracy"],
                "test_loss": test_stats[optimizer_name]["loss"],
                "test_top1_accuracy": test_stats[optimizer_name]["top1_accuracy"],
                "test_top5_accuracy": test_stats[optimizer_name]["top5_accuracy"],
            }
            for metric in SUMMARY_METRICS:
                for statistic, value in summarize(metric_values[metric]).items():
                    row[f"{metric}_{statistic}"] = value
            writer.writerow(row)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict[str, float | None]:
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    samples = 0
    model.eval()
    for data, target in loader:
        target = target.to(device)
        output = model(data)
        total_loss += criterion(output, target).item() * target.size(0)
        top1_correct += (output.argmax(dim=1) == target).sum().item()
        _, top5 = output.topk(5, dim=1, largest=True, sorted=True)
        top5_correct += (top5 == target.unsqueeze(1)).any(dim=1).sum().item()
        samples += target.size(0)
    return {
        "loss": total_loss / samples,
        "top1_accuracy": top1_correct / samples * 100.0,
        "top5_accuracy": top5_correct / samples * 100.0,
    }


def run_experiment() -> None:
    args = parse_args()
    config = ARCHITECTURES[args.arch]
    seed_everything(args.seed)
    paths = configure_outputs(args)

    logger = configure_logger(paths.log)
    os.makedirs(args.output_dir, exist_ok=True)
    configure_lns(args)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader, test_loader = create_data_loaders(config, args, loader_generator)
    first_model = create_model(args.arch, config.num_classes, args.device)
    models = {args.optimizers[0]: first_model}
    for optimizer_name in args.optimizers[1:]:
        model = create_model(args.arch, config.num_classes, args.device)
        model.load_state_dict(first_model.state_dict())
        models[optimizer_name] = model
    optimizers = {
        name: create_trajectory_optimizer(name, model, args)
        for name, model in models.items()
    }
    schedulers = {
        name: create_scheduler(optimizer, args.epochs, args.lr)
        for name, optimizer in optimizers.items()
    }
    criterion = nn.NLLLoss()
    shadow_states = {
        optimizer_name: {format_name: {} for format_name in args.formats}
        for optimizer_name in args.optimizers
    }
    logger.info(
        "Run | arch=%s | dataset=%s | device=%s | epochs=%s | batch_size=%s",
        args.arch,
        config.dataset_name,
        args.device,
        args.epochs,
        args.batch_size,
    )
    probe_details = [
        f"optimizers={','.join(args.optimizers)}",
        f"formats={','.join(args.formats)}",
        "gradient_source=shared_fp64",
        "shadow_state=persistent",
    ]
    if any(name.startswith("lns") for name in args.formats):
        probe_details.extend(
            [
                f"lns_prec={args.lns_prec}",
                f"table={args.table}",
            ]
        )
    logger.info("Probe | %s", " | ".join(probe_details))
    if "sgd" in args.optimizers:
        logger.info(
            "SGD hyperparameters | initial_lr=%s | momentum=%s | "
            "dampening=%s | weight_decay=%s",
            args.lr,
            args.momentum,
            args.dampening,
            args.weight_decay,
        )
    if "madam" in args.optimizers:
        logger.info(
            "Madam hyperparameters | initial_lr=%s | beta=%s | eps=%s | "
            "g_bound=%s",
            args.lr,
            args.beta,
            args.eps,
            args.g_bound,
        )

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        batch_file = open_epoch_json_section(paths.batch, epoch)
        first_json_row = True
        totals = {
            name: {"loss": 0.0, "correct": 0, "samples": 0}
            for name in args.optimizers
        }
        epoch_metrics = {
            f"{optimizer_name}:{format_name}": {
                metric: [] for metric in SUMMARY_METRICS
            }
            for optimizer_name in args.optimizers
            for format_name in args.formats
        }
        learning_rates = {
            name: float(optimizer.param_groups[0]["lr"])
            for name, optimizer in optimizers.items()
        }

        for model in models.values():
            model.train()
        for batch_idx, (data, target) in enumerate(train_loader, start=1):
            if args.max_batches is not None and batch_idx > args.max_batches:
                break
            target = target.to(args.device)
            batch_rows = {}

            for optimizer_name in args.optimizers:
                model = models[optimizer_name]
                optimizer = optimizers[optimizer_name]
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                weights_before = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
                gradients = {
                    name: parameter.grad.detach()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
                optimizer.step()
                parameters_after = dict(model.named_parameters())

                for format_name in args.formats:
                    metrics = shadow_step(
                        optimizer_name,
                        format_name,
                        weights_before,
                        gradients,
                        parameters_after,
                        shadow_states[optimizer_name][format_name],
                        args,
                        learning_rates[optimizer_name],
                    )
                    key = f"{optimizer_name}:{format_name}"
                    batch_rows[key] = metrics
                    row = {
                        "batch_idx": batch_idx,
                        "optimizer": optimizer_name,
                        "format": format_name,
                        **metrics,
                    }
                    if not first_json_row:
                        batch_file.write(",")
                    batch_file.write("\n    " + json.dumps(row, allow_nan=False))
                    first_json_row = False
                    for metric in SUMMARY_METRICS:
                        epoch_metrics[key][metric].append(metrics[metric])

                totals[optimizer_name]["loss"] += loss.item() * target.size(0)
                totals[optimizer_name]["correct"] += (
                    (output.argmax(dim=1) == target).sum().item()
                )
                totals[optimizer_name]["samples"] += target.size(0)

            if batch_idx % args.log_interval == 0:
                loss_summary = " | ".join(
                    f"{name.upper()} loss: "
                    f"{totals[name]['loss'] / totals[name]['samples']:.4f}"
                    for name in args.optimizers
                )
                logger.info(
                    "Epoch [%s/%s] | Batch %s | %s | Probes written: %s",
                    epoch,
                    args.epochs,
                    batch_idx,
                    loss_summary,
                    len(batch_rows),
                )

        close_epoch_json_section(batch_file)
        if any(value["samples"] == 0 for value in totals.values()):
            raise RuntimeError("the training loader produced no complete batches")
        train_stats = {
            name: {
                "loss": value["loss"] / value["samples"],
                "accuracy": value["correct"] / value["samples"] * 100.0,
            }
            for name, value in totals.items()
        }
        if args.evaluate:
            test_stats = {
                name: evaluate(model, test_loader, criterion, args.device)
                for name, model in models.items()
            }
        else:
            test_stats = {
                name: {
                    "loss": None,
                    "top1_accuracy": None,
                    "top5_accuracy": None,
                }
                for name in args.optimizers
            }
        write_epoch_summaries(
            paths.epoch_summary,
            epoch,
            args,
            learning_rates,
            epoch_metrics,
            train_stats,
            test_stats,
        )
        for scheduler in schedulers.values():
            scheduler.step()

        train_summary = " | ".join(
            f"{name.upper()} loss/acc: {train_stats[name]['loss']:.4f}/"
            f"{train_stats[name]['accuracy']:.2f}%"
            for name in args.optimizers
        )
        logger.info(
            "Epoch [%s/%s] complete | %s | Time: %.2fs",
            epoch,
            args.epochs,
            train_summary,
            time.time() - epoch_start,
        )

    logger.info("Experiment complete")


if __name__ == "__main__":
    run_experiment()
