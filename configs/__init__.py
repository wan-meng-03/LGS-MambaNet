from copy import deepcopy
from pathlib import Path

from .datasets import get_dataset_config, resolve_split_dir
from .defaults import DEFAULTS


def _merge(base: dict, update: dict) -> dict:

    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)

    return base


def normalize_tag(tag: str | None) -> str:

    tag = str(tag or "main").strip()

    for character in (" ", "/", "\\", ":"):
        tag = tag.replace(character, "_")

    while "__" in tag:
        tag = tag.replace("__", "_")

    return tag.strip("_") or "main"


def build_config(
    model: str,
    dataset: str,
    protocol: str,
    fold: int | None = None,
    seed: int = 1,
    overrides: dict | None = None,
) -> dict:

    from models import get_model_spec

    spec = get_model_spec(model)
    dataset_config = get_dataset_config(dataset)

    config = _merge(
        deepcopy(DEFAULTS),
        deepcopy(spec.defaults),
    )


    config["method_epochs"] = int(config["epochs"])

    config.update(
        {
            "model": spec.name,
            "dataset": dataset_config["name"],
            "dataset_root": dataset_config["image_root"],
            "image_dirs": dataset_config["image_dirs"],
            "class_names": dataset_config["class_names"],
            "num_classes": dataset_config["num_classes"],
            "protocol": protocol,
            "fold": fold,
            "split_dir": str(
                resolve_split_dir(
                    dataset=dataset,
                    protocol=protocol,
                    fold=fold,
                )
            ),
            "seed": int(seed),
        }
    )

    if overrides:
        _merge(config, overrides)

    for key in (
        "epochs",
        "batch_size",
        "evaluation_batch_size",
        "num_workers",
    ):
        config[key] = int(config[key])

    if config["epochs"] <= 0:
        raise ValueError("epochs必须大于0。")

    if config["batch_size"] <= 0:
        raise ValueError("batch_size必须大于0。")

    if config["evaluation_batch_size"] <= 0:
        raise ValueError("evaluation_batch_size必须大于0。")

    if config["num_workers"] < 0:
        raise ValueError("num_workers不能小于0。")

    return config


def build_run_directory(
    config: dict,
    run_root: str | Path = "runs",
    tag: str | None = None,
) -> Path:

    parts = []

    if config.get("fold") is not None:
        parts.append(f"fold{int(config['fold'])}")

    parts.extend(
        (
            f"seed{int(config['seed'])}",
            normalize_tag(tag or config.get("tag")),
        )
    )

    return (
        Path(run_root)
        / str(config["dataset"])
        / str(config["model"])
        / str(config["protocol"])
        / "_".join(parts)
    )


def build_test_directory(
    config: dict,
    test_root: str | Path = "test_results",
    tag: str | None = None,
) -> Path:


    experiment_tag = normalize_tag(
        tag or config.get("tag")
    )

    return (
        Path(test_root)
        / str(config["dataset"])
        / str(config["model"])
        / experiment_tag
        / str(config["protocol"])
        / f"seed{int(config['seed'])}"
    )


__all__ = [
    "build_config",
    "build_run_directory",
    "build_test_directory",
    "get_dataset_config",
    "normalize_tag",
    "resolve_split_dir",
]
