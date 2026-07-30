from torchvision import datasets

from float_cifar_train import CIFAR10_MEAN, CIFAR10_STD, ExperimentConfig, run_experiment
from models.mobilenetv1 import MobileNetV1


EXPERIMENT = ExperimentConfig(
    file_stem="mobilenetv1_float",
    model_name="MobileNetV1",
    dataset_type=datasets.CIFAR10,
    num_classes=10,
    mean=CIFAR10_MEAN,
    std=CIFAR10_STD,
    model_factory=lambda num_classes, dtype, device: MobileNetV1(
        num_classes, cifar=True, madam=True, dtype=dtype, device=device
    ),
)


if __name__ == "__main__":
    run_experiment(EXPERIMENT)
