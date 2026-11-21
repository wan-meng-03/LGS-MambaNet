import argparse
import csv
import json
from numbers import Number
from pathlib import Path
from statistics import fmean, pstdev

import torch

from configs import (
    build_run_directory,
    build_test_directory,
    get_dataset_config,
    normalize_tag,
)
from datasets import build_loader
from engine import evaluate_classifier
from utils import set_global_seed


METRICS = {
    "loss": "Loss",
    "accuracy": "ACC",

    "macro_precision": "Macro Precision",
    "weighted_precision": "Weighted Precision",

    "macro_recall": "Macro Recall",

    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",

    "macro_specificity": "Macro Specificity",
    "mcc": "MCC",

    "macro_roc_auc": "Macro ROC-AUC",
    "weighted_roc_auc": "Weighted ROC-AUC",
    "micro_roc_auc": "Micro ROC-AUC",

    "macro_ap": "Macro AP",
    "weighted_ap": "Weighted AP",
    "micro_ap": "Micro AP",
}


PERCENT_METRICS = {
    "accuracy",

    "macro_precision",
    "weighted_precision",

    "macro_recall",

    "macro_f1",
    "weighted_f1",

    "macro_specificity",

    "macro_roc_auc",
    "weighted_roc_auc",
    "micro_roc_auc",

    "macro_ap",
    "weighted_ap",
    "micro_ap",


    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "ap",
}


RAW_METRICS = {
    "loss",
    "mcc",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="通用皮肤病分类测试。"
    )

    parser.add_argument(
        "--model",
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
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="测试fold0～fold4并计算均值和标准差。",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        metavar="FOLD",
        help="指定多个折，例如--folds 0 2 3 4。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--tag",
        help="实验标签；未指定时优先从checkpoint恢复。",
    )

    parser.add_argument(
        "--run-root",
        default="runs",
    )
    parser.add_argument(
        "--test-root",
        default="test_results",
    )

    parser.add_argument(
        "--checkpoint",
        default="best",
        help=(
            "测试权重："
            "best=Validation监控指标最优；"
            "best_recall=Validation Macro Recall最优；"
            "best_f1=Validation Macro F1最优；"
            "last=最后一个Epoch。"
        ),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="直接指定一个检查点路径。",
    )

    parser.add_argument(
        "--workers",
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )

    return parser.parse_args()


def resolve_device(name: str) -> torch.device:

    device = torch.device(name)

    if device.type != "cuda":
        return device

    if not torch.cuda.is_available():
        raise RuntimeError(
            "请求使用CUDA，但当前环境无法使用CUDA。"
        )

    index = (
        device.index
        if device.index is not None
        else 0
    )

    if index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA设备不存在：{name}；"
            f"当前可见GPU数量为"
            f"{torch.cuda.device_count()}。"
        )

    return device


def load_checkpoint(path: Path) -> dict:

    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def convert_metrics_to_percent(
    value,
    key=None,
):


    if isinstance(value, dict):
        return {
            sub_key: convert_metrics_to_percent(
                sub_value,
                key=sub_key,
            )
            for sub_key, sub_value
            in value.items()
        }

    if isinstance(value, list):
        return [
            convert_metrics_to_percent(
                item,
                key=key,
            )
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            convert_metrics_to_percent(
                item,
                key=key,
            )
            for item in value
        ]

    if (
        key in PERCENT_METRICS
        and isinstance(value, Number)
    ):
        return float(value) * 100.0

    return value


def format_metric_value(
    metric_key: str,
    value: float,
) -> str:


    if metric_key in RAW_METRICS:
        return f"{float(value):.4f}"

    return f"{float(value):.2f}"


def format_main_metrics(
    metrics: dict,
) -> str:


    name_width = 24
    value_width = 12

    lines = [
        (
            f"{'Metric':<{name_width}} "
            f"{'Value':>{value_width}}"
        ),
        "-" * (
            name_width
            + value_width
            + 1
        ),
    ]

    for key, display_name in (
        METRICS.items()
    ):
        if key not in metrics:
            continue

        label = (
            display_name
            if key in RAW_METRICS
            else f"{display_name} (%)"
        )

        value = format_metric_value(
            key,
            metrics[key],
        )

        lines.append(
            f"{label:<{name_width}} "
            f"{value:>{value_width}}"
        )

    return "\n".join(lines)

def resolve_checkpoint_path(
    args,
    fold: int | None = None,
) -> Path:
    config = {
        "model": args.model,
        "dataset": args.dataset,
        "protocol": args.protocol,
        "fold": fold,
        "seed": args.seed,
    }
    return (
        build_run_directory(
            config=config,
            run_root=args.run_root,
            tag=args.tag or "main",
        )
        / "checkpoints"
        / f"{args.checkpoint}.pth"
    )


def infer_tag(
    checkpoint_path: Path,
    config: dict,
) -> str:

    if config.get("tag"):
        return normalize_tag(
            config["tag"]
        )

    run_name = (
        checkpoint_path
        .parent
        .parent
        .name
    )

    fold = config.get("fold")
    seed = int(
        config["seed"]
    )

    prefix = (
        f"fold{int(fold)}_seed{seed}_"
        if fold is not None
        else f"seed{seed}_"
    )

    if run_name.startswith(prefix):
        return normalize_tag(
            run_name[len(prefix):]
        )

    return "main"


def build_test_report(
    result: dict,
) -> str:


    fold = result["fold"]

    title = f"Fold {fold} Test"

    metrics = result["metrics"]

    lines = [
        (
            f"{'=' * 22} "
            f"{title} "
            f"{'=' * 22}"
        ),
        f"Model: {result['model']}",
        f"Dataset: {result['dataset']}",
        f"Protocol: {result['protocol']}",
        f"Fold: {fold}",
        f"Seed: {result['seed']}",
        f"Tag: {result['tag']}",
        f"Checkpoint: {result['checkpoint']}",
        (
            "Checkpoint epoch: "
            f"{result['checkpoint_epoch']}"
        ),
        (
            "Test samples: "
            f"{result['test_samples']}"
        ),
        (
            "Metric unit: percentage "
            "except Loss and MCC"
        ),
        "",
        "===== Overall Metrics =====",
        "",
        format_main_metrics(
            metrics
        ),
    ]

    per_class = metrics.get(
        "per_class"
    )

    class_names = metrics.get(
        "class_names",
        [],
    )

    if (
        per_class
        and class_names
    ):
        lines.extend(
            [
                "",
                "===== Per-Class Metrics =====",
                "",
                (
                    "Class       Support   "
                    "Precision   Recall      "
                    "Specificity  F1          "
                    "ROC-AUC     AP"
                ),
            ]
        )

        for index, name in enumerate(
            class_names
        ):
            roc_auc = (
                per_class[
                    "roc_auc"
                ][index]
            )

            ap = (
                per_class[
                    "ap"
                ][index]
            )

            lines.append(
                f"{name:<11} "
                f"{per_class['support'][index]:<9d} "
                f"{per_class['precision'][index]:<11.2f} "
                f"{per_class['recall'][index]:<11.2f} "
                f"{per_class['specificity'][index]:<12.2f} "
                f"{per_class['f1'][index]:<11.2f} "
                f"{roc_auc:<11.2f} "
                f"{ap:.2f}"
            )

    confusion = metrics.get(
        "confusion_matrix"
    )

    if confusion is not None:
        lines.extend(
            [
                "",
                "===== Confusion Matrix =====",
                "",
            ]
        )

        lines.extend(
            str(row)
            for row in confusion
        )

    return "\n".join(
        lines
    )

def evaluate_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    test_root: str | Path,
    checkpoint_name: str,
    requested_tag: str | None = None,
    workers: int | None = None,
    batch_size: int | None = None,
) -> dict:

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"检查点不存在："
            f"{checkpoint_path}"
        )

    checkpoint = load_checkpoint(
        checkpoint_path
    )

    if "config" not in checkpoint:
        raise KeyError(
            "检查点中缺少config。"
        )

    config = dict(
        checkpoint["config"]
    )

    if workers is not None:
        config["num_workers"] = int(
            workers
        )

    if batch_size is not None:
        config[
            "evaluation_batch_size"
        ] = int(batch_size)

    config["pretrained"] = False
    config["pin_memory"] = (
        device.type == "cuda"
    )

    config.setdefault(
        "method_epochs",
        config.get(
            "epochs",
            100,
        ),
    )

    tag = normalize_tag(
        requested_tag
        or infer_tag(
            checkpoint_path,
            config,
        )
    )

    config["tag"] = tag

    set_global_seed(
        seed=int(
            config["seed"]
        ),
        deterministic=bool(
            config.get(
                "deterministic",
                True,
            )
        ),
    )

    model_name = checkpoint.get(
        "model_name",
        config.get("model"),
    )

    if not model_name:
        raise KeyError(
            "检查点中缺少模型名称。"
        )

    class_counts = tuple(
        checkpoint.get(
            "class_counts",
            (),
        )
    )

    if not class_counts:
        raise KeyError(
            "检查点中缺少class_counts。"
        )

    from models import get_model_spec

    spec = get_model_spec(
        model_name
    )

    model = spec.build_model(
        config=config,
        class_counts=class_counts,
    )

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    spec.restore_extra(
        model,
        checkpoint,
    )

    model.to(device)

    test_loader = build_loader(
        config=config,
        spec=spec,
        split="test",
    )

    raw_metrics = (
        evaluate_classifier(
            model=model,
            spec=spec,
            data_loader=test_loader,
            device=device,
            num_classes=int(
                config[
                    "num_classes"
                ]
            ),
            class_names=(
                config[
                    "class_names"
                ]
            ),
            include_full_probability_metrics=True,
        )
    )

    metrics = (
        convert_metrics_to_percent(
            raw_metrics
        )
    )

    result = {
        "checkpoint": str(
            checkpoint_path.resolve()
        ),
        "checkpoint_name": (
            checkpoint_name
        ),
        "checkpoint_epoch": int(
            checkpoint["epoch"]
        ),
        "model": model_name,
        "dataset": (
            config["dataset"]
        ),
        "protocol": (
            config["protocol"]
        ),
        "fold": (
            config.get("fold")
        ),
        "seed": int(
            config["seed"]
        ),
        "tag": tag,
        "test_samples": len(
            test_loader.dataset
        ),
        "metric_units": {
            key: (
                "raw"
                if key in RAW_METRICS
                else "percentage"
            )
            for key in METRICS
        },
        "metrics": metrics,
    }

    output_directory = (
        build_test_directory(
            config=config,
            test_root=test_root,
            tag=tag,
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold = config.get("fold")

    experiment_name = (
        f"{model_name}_{tag}"
    )

    output_stem = (
        f"{experiment_name}_"
        f"fold{int(fold)}_"
        f"{checkpoint_name}"
        if fold is not None
        else
        f"{experiment_name}_"
        f"{checkpoint_name}"
    )

    json_path = (
        output_directory
        / f"{output_stem}.json"
    )

    text_path = (
        output_directory
        / f"{output_stem}.txt"
    )

    json_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = (
        build_test_report(
            result
        )
    )

    text_path.write_text(
        report + "\n",
        encoding="utf-8",
    )

    print(
        f"\n{report}"
    )
    print(
        f"\nJSON结果："
        f"{json_path}"
    )
    print(
        f"文本报告："
        f"{text_path}"
    )

    del model
    del test_loader

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def summarize_folds(
    results: list[dict],
    output_directory: Path,
    checkpoint_name: str,
) -> None:

    if len(results) < 2:
        raise ValueError(
            "多折汇总至少需要2组结果。"
        )

    results = sorted(
        results,
        key=lambda item: int(
            item["fold"]
        ),
    )

    summary_metrics = {}

    for key, display_name in (
        METRICS.items()
    ):
        values = [
            float(
                result[
                    "metrics"
                ][key]
            )
            for result in results
        ]

        summary_metrics[key] = {
            "name": display_name,
            "values": values,
            "mean": fmean(values),
            "std": pstdev(values),
        }

    summary = {
        "model": (
            results[0]["model"]
        ),
        "dataset": (
            results[0]["dataset"]
        ),
        "protocol": (
            "cross_5fold"
        ),
        "folds": [
            int(result["fold"])
            for result in results
        ],
        "num_folds": len(results),
        "checkpoint": (
            checkpoint_name
        ),
        "seed": (
            results[0]["seed"]
        ),
        "tag": (
            results[0]["tag"]
        ),
        "standard_deviation": (
            "population"
        ),
        "metric_units": {
            key: (
                "raw"
                if key in RAW_METRICS
                else "percentage"
            )
            for key in METRICS
        },
        "folds": [
            {
                "fold": (
                    result["fold"]
                ),
                "checkpoint_epoch": (
                    result[
                        "checkpoint_epoch"
                    ]
                ),
                "test_samples": (
                    result[
                        "test_samples"
                    ]
                ),
                "checkpoint": (
                    result[
                        "checkpoint"
                    ]
                ),
            }
            for result in results
        ],
        "metrics": summary_metrics,
    }

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_stem = (
        f"{summary['model']}_"
        f"{summary['tag']}_"
        f"summary_{checkpoint_name}"
    )

    json_path = (
        output_directory
        / f"{output_stem}.json"
    )

    csv_path = (
        output_directory
        / f"{output_stem}.csv"
    )

    text_path = (
        output_directory
        / f"{output_stem}.txt"
    )

    json_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            (
                "metric",
                "fold0",
                "fold1",
                "fold2",
                "fold3",
                "fold4",
                "mean",
                "std",
                "unit",
            )
        )

        for key in METRICS:
            item = (
                summary_metrics[key]
            )

            decimals = (
                4
                if key in RAW_METRICS
                else 2
            )

            unit = (
                "raw"
                if key in RAW_METRICS
                else "percentage"
            )

            writer.writerow(
                (
                    item["name"],
                    *[
                        round(
                            float(value),
                            decimals,
                        )
                        for value
                        in item["values"]
                    ],
                    round(
                        float(
                            item["mean"]
                        ),
                        decimals,
                    ),
                    round(
                        float(
                            item["std"]
                        ),
                        decimals,
                    ),
                    unit,
                )
            )


    metric_width = 25
    mean_width = 20
    fold_width = 10

    table_header = (
        f"{'Metric':<{metric_width}} "
        f"{'Mean ± Std':<{mean_width}}"
        + "".join(
            f"{f'Fold{i}':>{fold_width}}"
            for i in range(5)
        )
    )

    table_width = len(
        table_header
    )

    separator = "-" * table_width

    lines = [
        "=" * table_width,
        "Five-Fold Summary".center(
            table_width
        ),
        "=" * table_width,
        (
            f"Model      : "
            f"{summary['model']}"
        ),
        (
            f"Dataset    : "
            f"{summary['dataset']}"
        ),
        (
            f"Protocol   : "
            f"{summary['protocol']}"
        ),
        (
            f"Checkpoint : "
            f"{checkpoint_name}"
        ),
        (
            f"Seed       : "
            f"{summary['seed']}"
        ),
        (
            f"Tag        : "
            f"{summary['tag']}"
        ),
        (
            "Std        : "
            "population (n)"
        ),
        "",
    ]

    metric_groups = (
        (
            "Classification Metrics",
            (
                "loss",
                "accuracy",
                "macro_precision",
                "weighted_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
                "macro_specificity",
                "mcc",
            ),
        ),
        (
            "ROC-AUC Metrics",
            (
                "macro_roc_auc",
                "weighted_roc_auc",
                "micro_roc_auc",
            ),
        ),
        (
            "Average Precision Metrics",
            (
                "macro_ap",
                "weighted_ap",
                "micro_ap",
            ),
        ),
    )

    for (
        group_name,
        metric_keys,
    ) in metric_groups:

        lines.extend(
            [
                group_name,
                separator,
                table_header,
                separator,
            ]
        )

        for key in metric_keys:
            item = summary_metrics[key]

            label = (
                item["name"]
                if key in RAW_METRICS
                else f"{item['name']} (%)"
            )

            mean_text = (
                format_metric_value(
                    key,
                    item["mean"],
                )
            )

            std_text = (
                format_metric_value(
                    key,
                    item["std"],
                )
            )

            mean_std = (
                f"{mean_text} ± "
                f"{std_text}"
            )

            fold_text = "".join(
                f"{format_metric_value(key, value):>{fold_width}}"
                for value in item["values"]
            )

            lines.append(
                f"{label:<{metric_width}} "
                f"{mean_std:<{mean_width}}"
                f"{fold_text}"
            )

        lines.append("")

    report = "\n".join(
        lines
    ).rstrip()

    text_path.write_text(
        report + "\n",
        encoding="utf-8",
    )

    print(
        f"\n{report}"
    )
    print(
        f"\n五折JSON："
        f"{json_path}"
    )
    print(
        f"多折CSV： "
        f"{csv_path}"
    )
    print(
        f"多折文本："
        f"{text_path}"
    )


def validate_args(args) -> None:
    if args.checkpoint_path is not None:
        if args.all_folds or args.folds is not None or args.fold is not None:
            raise ValueError("--checkpoint-path cannot be combined with fold selection")
        return

    if args.all_folds:
        if args.fold is not None or args.folds is not None:
            raise ValueError("--all-folds cannot be combined with --fold or --folds")
        return

    if args.folds is not None:
        if args.fold is not None:
            raise ValueError("--folds cannot be combined with --fold")
        if len(args.folds) < 2:
            raise ValueError("--folds requires at least two fold indices")
        if len(set(args.folds)) != len(args.folds):
            raise ValueError("--folds cannot contain duplicates")
        invalid = [fold for fold in args.folds if fold not in range(5)]
        if invalid:
            raise ValueError(f"Invalid fold indices: {invalid}; expected 0-4")
        return

    if args.fold not in range(5):
        raise ValueError("cross_5fold requires --fold 0-4, --folds, or --all-folds")


def main():
    args = parse_args()


    args.dataset = get_dataset_config(
        args.dataset
    )["name"]

    validate_args(args)

    if args.checkpoint_path is None and not args.model:
        raise ValueError("--model is required when --checkpoint-path is not provided")

    device = resolve_device(
        args.device
    )

    if args.checkpoint_path is not None:
        evaluate_checkpoint(
            checkpoint_path=(
                args.checkpoint_path
            ),
            device=device,
            test_root=(
                args.test_root
            ),
            checkpoint_name=(
                args.checkpoint_path.stem
            ),
            requested_tag=(
                args.tag
            ),
            workers=(
                args.workers
            ),
            batch_size=(
                args.batch_size
            ),
        )

        return

    if args.all_folds or args.folds is not None:
        results = []

        folds = (
            range(5)
            if args.all_folds
            else args.folds
        )

        for fold in folds:
            checkpoint_path = (
                resolve_checkpoint_path(
                    args,
                    fold=fold,
                )
            )

            result = (
                evaluate_checkpoint(
                    checkpoint_path=(
                        checkpoint_path
                    ),
                    device=device,
                    test_root=(
                        args.test_root
                    ),
                    checkpoint_name=(
                        args.checkpoint
                    ),
                    requested_tag=(
                        args.tag
                        or "main"
                    ),
                    workers=(
                        args.workers
                    ),
                    batch_size=(
                        args.batch_size
                    ),
                )
            )

            results.append(
                result
            )

        summary_config = {
            "model": (
                results[0]["model"]
            ),
            "dataset": (
                results[0]["dataset"]
            ),
            "protocol": (
                "cross_5fold"
            ),
            "seed": (
                results[0]["seed"]
            ),
            "tag": (
                results[0]["tag"]
            ),
        }

        output_directory = (
            build_test_directory(
                config=(
                    summary_config
                ),
                test_root=(
                    args.test_root
                ),
                tag=(
                    results[0]["tag"]
                ),
            )
        )

        summarize_folds(
            results=results,
            output_directory=(
                output_directory
            ),
            checkpoint_name=(
                args.checkpoint
            ),
        )

        return

    checkpoint_path = (
        resolve_checkpoint_path(
            args,
            fold=args.fold,
        )
    )

    evaluate_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        device=device,
        test_root=(
            args.test_root
        ),
        checkpoint_name=(
            args.checkpoint
        ),
        requested_tag=(
            args.tag or "main"
        ),
        workers=(
            args.workers
        ),
        batch_size=(
            args.batch_size
        ),
    )


if __name__ == "__main__":
    main()
