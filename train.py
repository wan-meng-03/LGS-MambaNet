import argparse
import json

import torch

from configs import (
    build_config,
    build_run_directory,
    normalize_tag,
)
from datasets import build_loaders
from engine import Trainer
from utils import set_global_seed


def _parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _set_nested(target: dict, key: str, value) -> None:
    current = target
    parts = key.split(".")

    for part in parts[:-1]:
        current = current.setdefault(part, {})

    current[parts[-1]] = value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Skin lesion classification training."
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        default="isic2019",
        choices=("isic2019", "derm7pt", "padufes20"),
    )

    parser.add_argument(
        "--protocol",
        default="cross_5fold",
        choices=("cross_5fold",),
    )

    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(5),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--workers", type=int)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--run-root", default="runs")

    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
    )

    parser.add_argument(
        "--reset-best",
        action="store_true",
    )

    parser.add_argument(
        "--resume-non-strict",
        action="store_true",
    )

    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)

    if device.type != "cuda":
        return device

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    index = (
        device.index
        if device.index is not None
        else 0
    )

    if index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device {name} does not exist. "
            f"Available GPU count: {torch.cuda.device_count()}."
        )

    return device


def main():
    args = parse_args()

    overrides = {}

    direct_overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "evaluation_batch_size": args.eval_batch_size,
        "num_workers": args.workers,
        "pretrained": args.pretrained,
        "normalize": args.normalize,
    }

    for key, value in direct_overrides.items():
        if value is not None:
            overrides[key] = value

    for item in args.set:
        if "=" not in item:
            raise ValueError(
                f"Invalid --set format: {item}. "
                "Expected KEY=VALUE."
            )

        key, value = item.split("=", 1)

        _set_nested(
            overrides,
            key,
            _parse_value(value),
        )

    from models import get_model_spec

    spec = get_model_spec(args.model)

    config = build_config(
        model=args.model,
        dataset=args.dataset,
        protocol=args.protocol,
        fold=args.fold,
        seed=args.seed,
        overrides=overrides,
    )

    config["tag"] = normalize_tag(args.tag)

    device = resolve_device(args.device)
    config["pin_memory"] = device.type == "cuda"

    set_global_seed(
        seed=config["seed"],
        deterministic=config["deterministic"],
    )

    run_directory = build_run_directory(
        config=config,
        run_root=args.run_root,
        tag=config["tag"],
    )

    config["run_root"] = str(args.run_root)
    config["run_directory"] = str(run_directory)

    config["resume_checkpoint"] = (
        str(args.resume) if args.resume else None
    )

    config["resume_reset_best"] = bool(
        args.reset_best
    )

    config["resume_strict"] = not (
        args.resume_non_strict
    )

    checkpoint_directory = run_directory / "checkpoints"

    if (
        checkpoint_directory.exists()
        and any(checkpoint_directory.glob("*.pth"))
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Experiment already exists: {run_directory}. "
            "Use --overwrite to replace existing checkpoints."
        )

    loaders = build_loaders(
        config=config,
        spec=spec,
    )

    class_counts = loaders.train.dataset.class_counts

    if any(count <= 0 for count in class_counts):
        raise ValueError(
            f"Empty class detected in training set: {class_counts}"
        )

    model = spec.build_model(
        config=config,
        class_counts=class_counts,
    )

    summary = {
        "model": config["model"],
        "dataset": config["dataset"],
        "protocol": config["protocol"],
        "fold": config["fold"],
        "seed": config["seed"],
        "tag": config["tag"],
        "train_samples": len(loaders.train.dataset),
        "validation_samples": len(
            loaders.validation.dataset
        ),
        "class_counts": class_counts,
        "run_directory": str(run_directory),
    }

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    trainer = Trainer(
        model=model,
        spec=spec,
        train_loader=loaders.train,
        valid_loader=loaders.validation,
        device=device,
        run_directory=run_directory,
        config=config,
    )

    trainer.fit()


if __name__ == "__main__":
    main()