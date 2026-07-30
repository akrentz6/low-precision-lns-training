from torchvision import datasets

from lns_cifar_train import CIFAR10_MEAN, CIFAR10_STD, ExperimentConfig, run_experiment
from models.shufflenetv2 import ShuffleNetV2


EXPERIMENT = ExperimentConfig(
    file_stem="shufflenetv2_lns",
    model_name="ShuffleNetV2",
    dataset_type=datasets.CIFAR10,
    num_classes=10,
    mean=CIFAR10_MEAN,
    std=CIFAR10_STD,
    model_factory=lambda num_classes, dtype, device: ShuffleNetV2(
        num_classes, 3, cifar=True, madam=True, dtype=dtype, device=device
    ),
)


if __name__ == "__main__":
    run_experiment(EXPERIMENT)
