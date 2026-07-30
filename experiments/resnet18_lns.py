from torchvision import datasets

from lns_cifar_train import (
    CIFAR100_MEAN,
    CIFAR100_STD,
    ExperimentConfig,
    run_experiment,
)
from models.resnet18 import ResNet18


EXPERIMENT = ExperimentConfig(
    file_stem="resnet18_lns",
    model_name="ResNet18",
    dataset_type=datasets.CIFAR100,
    num_classes=100,
    mean=CIFAR100_MEAN,
    std=CIFAR100_STD,
    model_factory=lambda num_classes, dtype, device: ResNet18(
        num_classes, cifar=True, madam=True, dtype=dtype, device=device
    ),
)


if __name__ == "__main__":
    run_experiment(EXPERIMENT)
