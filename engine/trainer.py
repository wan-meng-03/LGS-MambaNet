from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from utils.tensor import move_to_device
from .evaluator import (
    evaluate_classifier,
    format_evaluation,
)


MONITORS = {

    "accuracy",
    "macro_precision",
    "weighted_precision",
    "macro_recall",
    "macro_specificity",
    "macro_f1",
    "weighted_f1",
    "mcc",
    "macro_roc_auc",

}


def _json_default(value):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.Tensor):
        return (
            value.detach()
            .cpu()
            .tolist()
        )

    raise TypeError(
        "无法序列化："
        f"{type(value).__name__}"
    )


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
    spec,
        train_loader,
        valid_loader,
        device: torch.device,
        run_directory: str | Path,
        config: dict,
    ) -> None:
        self.model = model.to(device)
        self.spec = spec
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.config = dict(config)

        self.run_directory = Path(
            run_directory
        )
        self.checkpoint_directory = (
            self.run_directory
            / "checkpoints"
        )
        self.checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.monitor = str(
            config["checkpoint_monitor"]
        )

        if self.monitor not in MONITORS:
            raise ValueError(
                "不支持的监控指标："
                f"{self.monitor}"
            )

        self.logger = self._build_logger()

        self.optimizers = (
            self.spec.build_optimizers(
                self.model,
                self.config,
            )
        )

        self.schedulers = (
            self.spec.build_schedulers(
                self.optimizers,
                self.config,
            )
        )

        self.best_metric = float("-inf")
        self.best_bacc = float("-inf")
        self.best_f1 = float("-inf")

        self.history = []


        self.start_epoch = 1

        self._resume_if_needed()

        with (
            self.run_directory
            / "resolved_config.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.config,
                file,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )

    def _resume_if_needed(self) -> None:
        path = self.config.get(
            "resume_checkpoint"
        )

        if not path:
            return

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        source_counts = tuple(
            checkpoint.get(
                "class_counts",
                (),
            )
        )

        target_counts = tuple(
            self.train_loader
            .dataset
            .class_counts
        )

        if (
            source_counts
            and source_counts
            != target_counts
        ):
            raise ValueError(
                "续训检查点的类别样本数不一致。"
            )

        strict = bool(
            self.config.get(
                "resume_strict",
                True,
            )
        )

        source_model = checkpoint.get(
            "model_name"
        )

        if (
            strict
            and source_model
            != self.spec.name
        ):
            raise ValueError(
                "严格续训要求模型一致："
                f"{source_model} != "
                f"{self.spec.name}"
            )

        incompatible = (
            self.model.load_state_dict(
                checkpoint["model"],
                strict=strict,
            )
        )

        for key, optimizer in (
            self.optimizers.items()
        ):
            if key not in checkpoint[
                "optimizers"
            ]:
                raise KeyError(
                    "检查点缺少优化器："
                    f"{key}"
                )

            optimizer.load_state_dict(
                checkpoint[
                    "optimizers"
                ][key]
            )

        for key, scheduler in (
            self.schedulers.items()
        ):
            if key not in checkpoint[
                "schedulers"
            ]:
                raise KeyError(
                    "检查点缺少调度器："
                    f"{key}"
                )

            scheduler.load_state_dict(
                checkpoint[
                    "schedulers"
                ][key]
            )


            scheduler.step()

        self.spec.restore_extra(
            self.model,
            checkpoint,
        )

        self.start_epoch = (
            int(checkpoint["epoch"])
            + 1
        )

        if not self.config.get(
            "resume_reset_best",
            False,
        ):
            self.best_metric = float(
                checkpoint.get(
                    "best_metric",
                    float("-inf"),
                )
            )

            validation_metrics = (
                checkpoint.get(
                    "validation_metrics",
                    {},
                )
            )

            self.best_bacc = float(
                checkpoint.get(
                    "best_recall",
                    checkpoint.get(
                        "best_bacc",
                        validation_metrics.get(
                            "macro_recall",
                            float("-inf"),
                        ),
                    ),
                )
            )

            self.best_f1 = float(
                checkpoint.get(
                    "best_f1",
                    validation_metrics.get(
                        "macro_f1",
                        float("-inf"),
                    ),
                )
            )

        if self.start_epoch > int(
            self.config["epochs"]
        ):
            raise ValueError(
                "--epochs必须大于检查点Epoch。"
            )

        self.logger.info(
            "Resume: %s | "
            "Source model: %s | "
            "Start epoch: %d | "
            "LR: %s",
            path,
            source_model,
            self.start_epoch,
            {
                key: optimizer
                .param_groups[0]["lr"]
                for key, optimizer
                in self.optimizers.items()
            },
        )

        if not strict:
            self.logger.info(
                "Non-strict load | "
                "Missing: %s | "
                "Unexpected: %s",
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )

    def _build_logger(
        self,
    ) -> logging.Logger:
        self.run_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        logger = logging.getLogger(
            str(
                self.run_directory.resolve()
            )
        )

        logger.setLevel(
            logging.INFO
        )
        logger.propagate = False
        logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        )

        experiment_name = (
            f"{self.spec.name}_"
            f"{self.config.get('tag', 'main')}"
        )

        handlers = (
            logging.StreamHandler(),
            logging.FileHandler(
                self.run_directory
                / f"{experiment_name}_training.log",
                mode="w",
                encoding="utf-8",
            ),
        )

        for handler in handlers:
            handler.setFormatter(
                formatter
            )

            logger.addHandler(
                handler
            )

        return logger

    def _train_epoch(
        self,
        epoch: int,
    ) -> dict[str, Any]:

        self.model.train()

        self.spec.begin_epoch(
            self.model,
            self.optimizers,
            epoch,
            self.config,
        )

        sums = {
            "loss": 0.0,
        }

        latest = {}

        correct = 0
        samples = 0

        for batch_index, batch in enumerate(
            self.train_loader,
            start=1,
        ):
            batch = move_to_device(
                batch,
                self.device,
            )

            self.spec.before_batch(
                self.model,
                self.optimizers,
                self.config,
            )

            result = (
                self.spec.training_step(
                    self.model,
                    batch,
                    epoch,
                    self.config,
                )
            )

            if not torch.isfinite(
                result.loss
            ):
                raise FloatingPointError(
                    f"Epoch {epoch}，"
                    f"Batch {batch_index}"
                    "出现NaN或Inf。"
                )

            self.spec.backward_and_step(
                result,
                self.optimizers,
                self.config,
            )

            batch_size = int(
                result.targets.numel()
            )

            predictions = (
                result.logits.argmax(
                    dim=1
                )
            )

            samples += batch_size

            correct += int(
                (
                    predictions
                    == result.targets
                )
                .sum()
                .item()
            )

            sums["loss"] += (
                float(
                    result.loss
                    .detach()
                    .item()
                )
                * batch_size
            )

            for key, value in (
                result.logs.items()
            ):
                if isinstance(
                    value,
                    (int, float),
                ):
                    sums[key] = (
                        sums.get(
                            key,
                            0.0,
                        )
                        + float(value)
                        * batch_size
                    )
                else:
                    latest[key] = value

            if (
                batch_index == 1
                or batch_index
                % int(
                    self.config[
                        "log_interval"
                    ]
                )
                == 0
                or batch_index
                == len(
                    self.train_loader
                )
            ):
                numeric_logs = " | ".join(
                    f"{key} {value:.4f}"
                    for key, value
                    in result.logs.items()
                    if isinstance(
                        value,
                        (int, float),
                    )
                )

                batch_accuracy = float(
                    (
                        predictions
                        == result.targets
                    )
                    .float()
                    .mean()
                    .item()
                )

                self.logger.info(
                    "Epoch %03d | "
                    "Batch %03d/%03d | "
                    "Loss %.4f | "
                    "ACC %.4f%s",
                    epoch,
                    batch_index,
                    len(
                        self.train_loader
                    ),
                    float(
                        result.loss
                        .detach()
                        .item()
                    ),
                    batch_accuracy,
                    (
                        f" | {numeric_logs}"
                        if numeric_logs
                        else ""
                    ),
                )


        _numeric_names = []
        _numeric_flags = []

        for _name, _param in self.model.named_parameters():
            if _param.grad is not None:
                _numeric_names.append(
                    f"gradient:{_name}"
                )
                _numeric_flags.append(
                    torch.isfinite(
                        _param.grad.detach()
                    ).all().to(self.device)
                )

            _numeric_names.append(
                f"parameter:{_name}"
            )
            _numeric_flags.append(
                torch.isfinite(
                    _param.detach()
                ).all().to(self.device)
            )

        for _name, _buffer in self.model.named_buffers():
            if torch.is_floating_point(_buffer):
                _numeric_names.append(
                    f"buffer:{_name}"
                )
                _numeric_flags.append(
                    torch.isfinite(
                        _buffer.detach()
                    ).all().to(self.device)
                )

        if _numeric_flags:
            _bad_indices = torch.nonzero(
                ~torch.stack(_numeric_flags)
            ).flatten().tolist()

            if _bad_indices:
                _bad_names = [
                    _numeric_names[i]
                    for i in _bad_indices[:20]
                ]

                raise FloatingPointError(
                    f"[NUMERIC AUDIT] Epoch={epoch}, "
                    f"Non-finite state: {_bad_names}"
                )

            self.logger.info(
                "[NUMERIC AUDIT] Epoch %03d: "
                "parameters, gradients and buffers are finite.",
                epoch,
            )

        if samples == 0:
            raise RuntimeError(
                "训练数据集为空。"
            )

        metrics = {
            key: value / samples
            for key, value
            in sums.items()
        }

        metrics.update(
            {
                "accuracy": (
                    correct / samples
                ),
                "sample_count": (
                    samples
                ),
            }
        )

        metrics.update(
            latest
        )

        metrics.update(
            self.spec.end_epoch(
                self.model,
                self.optimizers,
                epoch,
                self.config,
            )
        )

        return metrics


    @staticmethod


    @staticmethod


    def _save_checkpoint(
        self,
        name: str,
        epoch: int,
        validation_metrics: dict,
    ) -> None:


        if name not in {"best", "best_f1"}:
            return

        checkpoint = {
            "model_name": (
                self.spec.name
            ),

            "epoch": int(
                epoch
            ),

            "model": (
                self.model
                .state_dict()
            ),

            "optimizers": {
                key: optimizer
                .state_dict()
                for key, optimizer
                in self.optimizers.items()
            },

            "schedulers": {
                key: scheduler
                .state_dict()
                for key, scheduler
                in self.schedulers.items()
            },

            "best_metric": float(
                self.best_metric
            ),

            "best_recall": float(
                self.best_bacc
            ),


            "best_bacc": float(
                self.best_bacc
            ),

            "best_f1": float(
                self.best_f1
            ),

            "monitor": (
                self.monitor
            ),

            "validation_metrics": (
                validation_metrics
            ),

            "class_counts": list(
                self.train_loader
                .dataset
                .class_counts
            ),

            "config": (
                self.config
            ),

            "extra": (
                self.spec
                .checkpoint_extra(
                    self.model
                )
            ),
        }

        torch.save(
            checkpoint,
            self.checkpoint_directory
            / f"{name}.pth",
        )

    def fit(
        self,
    ) -> None:

        self.logger.info(
            "Model: %s | Device: %s",
            self.spec.name,
            self.device,
        )

        self.logger.info(
            "Model info: %s",
            self.spec.describe(
                self.model
            ),
        )

        for epoch in range(
            self.start_epoch,
            int(
                self.config[
                    "epochs"
                ]
            )
            + 1,
        ):
            train_metrics = (
                self._train_epoch(
                    epoch
                )
            )

            validation_metrics = (
                evaluate_classifier(
                    model=self.model,
                    spec=self.spec,
                    data_loader=(
                        self.valid_loader
                    ),
                    device=(
                        self.device
                    ),
                    num_classes=int(
                        self.config[
                            "num_classes"
                        ]
                    ),
                    class_names=(
                        self.config[
                            "class_names"
                        ]
                    ),
                )
            )

            self.spec.after_validation(
                self.model,
                validation_metrics,
                self.device,
                self.config,
            )

            current_metric = float(
                validation_metrics[
                    self.monitor
                ]
            )

            current_bacc = float(
                validation_metrics[
                    "macro_recall"
                ]
            )

            current_f1 = float(
                validation_metrics[
                    "macro_f1"
                ]
            )

            improved = (
                current_metric
                > self.best_metric
            )

            improved_bacc = (
                current_bacc
                > self.best_bacc
            )

            improved_f1 = (
                current_f1
                > self.best_f1
            )

            if improved:
                self.best_metric = (
                    current_metric
                )

            if improved_bacc:
                self.best_bacc = (
                    current_bacc
                )

            if improved_f1:
                self.best_f1 = (
                    current_f1
                )


            record = {
                "epoch": (
                    epoch
                ),

                "train": (
                    train_metrics
                ),

                "validation": (
                    validation_metrics
                ),

                "monitor": (
                    self.monitor
                ),

                "monitor_value": (
                    current_metric
                ),

                "best_metric": (
                    self.best_metric
                ),

                "best_bacc": (
                    self.best_bacc
                ),

                "best_f1": (
                    self.best_f1
                ),

                "improved": (
                    improved
                ),

                "improved_bacc": (
                    improved_bacc
                ),

                "improved_f1": (
                    improved_f1
                ),

                "learning_rates": {
                    key: float(
                        optimizer
                        .param_groups[0][
                            "lr"
                        ]
                    )
                    for key, optimizer
                    in self.optimizers.items()
                },
            }

            self.history.append(
                record
            )

            self.logger.info(
                "Epoch %03d train: %s",
                epoch,
                train_metrics,
            )

            self.logger.info(
                "\n%s",
                format_evaluation(
                    validation_metrics
                ),
            )


            if improved:
                self._save_checkpoint(
                    "best",
                    epoch,
                    validation_metrics,
                )


            if improved_bacc:
                self._save_checkpoint(
                    "best_recall",
                    epoch,
                    validation_metrics,
                )


            if improved_f1:
                self._save_checkpoint(
                    "best_f1",
                    epoch,
                    validation_metrics,
                )


            self._save_checkpoint(
                "last",
                epoch,
                validation_metrics,
            )


            with (
                self.run_directory
                / "history.json"
            ).open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    self.history,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    default=_json_default,
                )


            for scheduler in (
                self.schedulers.values()
            ):
                scheduler.step()


        self.logger.info(
            "训练完成 | "
            "Best %s = %.6f | "
            "Best Macro Recall = %.6f | "
            "Best Macro F1 = %.6f",
            self.monitor,
            self.best_metric,
            self.best_bacc,
            self.best_f1,
        )
