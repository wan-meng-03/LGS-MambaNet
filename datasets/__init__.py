from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from utils import (
    build_generator,
    seed_worker,
)

from .isic import (
    ISICCSVClassificationDataset,
)


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    validation: DataLoader


def _build_dataset(
    config: dict,
    transform,
    split: str,
):
    return ISICCSVClassificationDataset(
        csv_path=(
            Path(config["split_dir"])
            / f"{split}.csv"
        ),
        dataset_root=config[
            "dataset_root"
        ],
        class_names=config[
            "class_names"
        ],
        transform=transform,
        image_dirs=config["image_dirs"],
        validate_paths=True,
    )


def _build_loader(
    dataset,
    config: dict,
    training: bool,
) -> DataLoader:
    workers = int(
        config["num_workers"]
    )

    loader_kwargs = {
        "batch_size": int(
            config["batch_size"]
            if training
            else config[
                "evaluation_batch_size"
            ]
        ),
        "shuffle": training,
        "drop_last": bool(
            config["drop_last"]
            and training
        ),
        "num_workers": workers,
        "pin_memory": bool(
            config.get(
                "pin_memory",
                False,
            )
        ),
        "persistent_workers": (
            workers > 0
        ),
        "worker_init_fn": seed_worker,
    }

    if workers > 0:
        loader_kwargs[
            "prefetch_factor"
        ] = 2

    if training:
        loader_kwargs["generator"] = (
            build_generator(
                int(config["seed"])
            )
        )

    return DataLoader(
        dataset,
        **loader_kwargs,
    )


def build_loaders(
    config: dict,
    spec,
) -> LoaderBundle:
    transforms = spec.build_transforms(
        config
    )

    train_dataset = _build_dataset(
        config,
        transforms.train,
        "train",
    )

    validation_dataset = _build_dataset(
        config,
        transforms.evaluation,
        "val",
    )

    return LoaderBundle(
        train=_build_loader(
            train_dataset,
            config,
            training=True,
        ),
        validation=_build_loader(
            validation_dataset,
            config,
            training=False,
        ),
    )


def build_loader(
    config: dict,
    spec,
    split: str,
) -> DataLoader:
    if split not in {
        "train",
        "val",
        "test",
    }:
        raise ValueError(
            f"不支持的数据划分：{split}"
        )

    transforms = spec.build_transforms(
        config
    )

    transform = (
        transforms.train
        if split == "train"
        else transforms.evaluation
    )

    dataset = _build_dataset(
        config,
        transform,
        split,
    )

    return _build_loader(
        dataset,
        config,
        training=split == "train",
    )


__all__ = [
    "ISICCSVClassificationDataset",
    "LoaderBundle",
    "build_loader",
    "build_loaders",
]
