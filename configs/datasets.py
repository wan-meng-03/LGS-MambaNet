
import os
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATASETS = {
    "derm7pt": {
        "class_names": (
            "BCC",
            "NEV",
            "MEL",
            "MISC",
            "SK",
        ),
        "image_root": os.getenv(
            "DERM7PT_ROOT",
            str(PROJECT_ROOT / "data/Derm7pt/release_v0/images"),
        ),
        "list_root": str(
            PROJECT_ROOT / "lists/lists_Derm7pt"
        ),
        "image_dirs": ("",),
    },

    "ham10000": {
        "class_names": (
            "AKIEC",
            "BCC",
            "BKL",
            "DF",
            "MEL",
            "NV",
            "VASC",
        ),
        "image_root": os.getenv(
            "HAM10000_ROOT",
            str(PROJECT_ROOT / "data/HAM10000"),
        ),
        "list_root": str(
            PROJECT_ROOT / "lists/lists_HAM10000"
        ),
        "image_dirs": (
            "",
            "HAM10000_images",
            "HAM10000_images_part_1",
            "HAM10000_images_part_2",
            "ISIC2018_Task3_Training_Input",
            "images",
        ),
    },

    "isic2019": {
        "class_names": (
            "AK",
            "BCC",
            "BKL",
            "DF",
            "MEL",
            "NV",
            "SCC",
            "VASC",
        ),
        "image_root": os.getenv(
            "ISIC2019_ROOT",
            str(PROJECT_ROOT / "data/ISIC2019"),
        ),
        "list_root": str(
            PROJECT_ROOT / "lists/lists_ISIC2019"
        ),
        "image_dirs": (
            "",
            "ISIC2019_Dataset",
            "ISIC_2019_Training_Input",
            "images",
        ),
    },

    "padufes20": {
        "class_names": (
            "ACK",
            "BCC",
            "MEL",
            "NEV",
            "SCC",
            "SEK",
        ),
        "image_root": os.getenv(
            "PAD_UFES20_ROOT",
            str(PROJECT_ROOT / "data/PAD-UFES-20"),
        ),
        "list_root": str(
            PROJECT_ROOT / "lists/lists_PADUFES20"
        ),
        "image_dirs": ("",),
    },
}


def get_dataset_config(name: str) -> dict:
    key = (
        name.strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
    )

    if key not in DATASETS:
        raise ValueError(f"Unsupported dataset: {name}")

    config = deepcopy(DATASETS[key])
    config["name"] = key
    config["num_classes"] = len(config["class_names"])

    return config


def resolve_split_dir(
    dataset: str,
    protocol: str = "cross_5fold",
    fold: int | None = None,
) -> Path:
    if protocol != "cross_5fold":
        raise ValueError(
            f"Unsupported protocol: {protocol}"
        )

    if fold not in range(5):
        raise ValueError(
            "cross_5fold requires fold 0-4"
        )

    root = Path(
        get_dataset_config(dataset)["list_root"]
    )

    return root / "cross_5fold" / f"fold{fold}"
