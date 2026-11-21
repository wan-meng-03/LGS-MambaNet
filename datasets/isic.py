from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = (
    ".jpg",
    ".jpeg",
    ".png",
)


def normalize_category(value: object) -> str:
    return str(value).strip().upper()


def _image_names(value: object) -> tuple[Path, ...]:
    path = Path(str(value).strip())

    if not str(path):
        raise ValueError(
            "CSV中存在空图像字段。"
        )

    if path.suffix:
        return (path,)

    return tuple(
        Path(f"{path}{suffix}")
        for suffix in IMAGE_SUFFIXES
    )


def resolve_image_path(
    value: object,
    category: str,
    dataset_root: str | Path,
    image_dirs: Sequence[str] = ("",),
) -> Path:
    root = Path(dataset_root)
    candidates = []

    for name in _image_names(value):
        if name.is_absolute():
            candidates.append(name)
            continue

        for subdir in image_dirs:
            base = root / subdir
            candidates.extend(
                (
                    base / name,
                    base / category / name,
                )
            )

    for path in candidates:
        if path.is_file():
            return path.resolve()

    tried = "\n".join(
        f"  - {path}"
        for path in candidates[:20]
    )

    raise FileNotFoundError(
        f"无法定位图像：{value}\n"
        f"已尝试：\n{tried}"
    )


class ISICCSVClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        dataset_root: str | Path,
        class_names: Sequence[str],
        transform=None,
        image_dirs: Sequence[str] = ("",),
        validate_paths: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.dataset_root = Path(dataset_root)
        self.class_names = tuple(class_names)
        self.transform = transform
        self.image_dirs = tuple(image_dirs)

        if not self.csv_path.is_file():
            raise FileNotFoundError(
                self.csv_path
            )

        if (
            validate_paths
            and not self.dataset_root.is_dir()
        ):
            raise FileNotFoundError(
                self.dataset_root
            )

        data = pd.read_csv(self.csv_path)

        missing = {
            "image",
            "category",
        } - set(data.columns)

        if missing:
            raise ValueError(
                f"{self.csv_path}缺少列："
                f"{sorted(missing)}"
            )

        data = data.copy()
        data["image"] = (
            data["image"]
            .astype(str)
            .str.strip()
        )
        data["category"] = (
            data["category"]
            .map(normalize_category)
        )

        unknown = sorted(
            set(data["category"])
            - set(self.class_names)
        )

        if unknown:
            raise ValueError(
                f"{self.csv_path}包含未知类别："
                f"{unknown}"
            )

        if data["image"].duplicated().any():
            raise ValueError(
                f"{self.csv_path}存在重复图像。"
            )

        class_to_index = {
            name: index
            for index, name
            in enumerate(self.class_names)
        }

        expected = (
            data["category"]
            .map(class_to_index)
            .astype(int)
        )

        if "label" in data.columns:
            provided = data["label"].astype(int)

            if not provided.equals(expected):
                raise ValueError(
                    f"{self.csv_path}中的label"
                    "与类别顺序不一致。"
                )

        data["label"] = expected
        self.dataframe = data.reset_index(
            drop=True
        )
        self.labels = (
            self.dataframe["label"].tolist()
        )

        self.image_paths = [
            resolve_image_path(
                row["image"],
                row["category"],
                self.dataset_root,
                self.image_dirs,
            )
            if validate_paths
            else Path(str(row["image"]))
            for _, row
            in self.dataframe.iterrows()
        ]

    @property
    def class_counts(self) -> tuple[int, ...]:
        counts = (
            self.dataframe["category"]
            .value_counts()
        )

        return tuple(
            int(counts.get(name, 0))
            for name in self.class_names
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        with Image.open(
            self.image_paths[index]
        ) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long,
        )

        return image, label
