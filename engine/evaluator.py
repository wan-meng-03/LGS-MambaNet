from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

from utils.tensor import move_to_device
def _divide(
    numerator,
    denominator,
):
    result = np.zeros_like(
        numerator,
        dtype=np.float64,
    )

    np.divide(
        numerator,
        denominator,
        out=result,
        where=denominator != 0,
    )

    return result


def _nanmean(values) -> float:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    valid = np.isfinite(values)

    if not np.any(valid):
        return float("nan")

    return float(
        values[valid].mean()
    )


def _weighted_mean(
    values,
    weights,
) -> float:
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not np.any(valid):
        return float("nan")

    denominator = weights[valid].sum()

    if denominator <= 0:
        return float("nan")

    return float(
        np.sum(
            values[valid]
            * weights[valid]
        )
        / denominator
    )


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    spec,
    data_loader,
    device: torch.device,
    num_classes: int,
    class_names: Sequence[str],
    include_full_probability_metrics: bool = False,
) -> dict[str, Any]:


    model.eval()

    logits_list = []
    labels_list = []

    total_loss = 0.0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(
        data_loader,
        start=1,
    ):
        images = move_to_device(
            images,
            device,
        )

        labels = labels.to(
            device,
            dtype=torch.long,
            non_blocking=True,
        )


        image_finite = torch.isfinite(images)

        if not image_finite.all().item():
            bad_rows = (
                ~image_finite
            ).reshape(images.shape[0], -1).any(dim=1)

            local_ids = (
                bad_rows.nonzero(as_tuple=False)
                .flatten().cpu().tolist()
            )

            raise FloatingPointError(
                f"[VAL INPUT NONFINITE] "
                f"batch={batch_index} "
                f"sample_ids={[total_samples+i for i in local_ids]} "
                f"labels={labels[bad_rows].cpu().tolist()} "
                f"NaN={torch.isnan(images).sum().item()} "
                f"Inf={torch.isinf(images).sum().item()}"
            )


        logits = spec.predict(
            model,
            images,
        )

        finite_logits = torch.isfinite(logits)

        if not finite_logits.all().item():
            bad_rows = (
                ~finite_logits
            ).reshape(logits.shape[0], -1).any(dim=1)

            local_ids = (
                bad_rows.nonzero(as_tuple=False)
                .flatten().cpu().tolist()
            )

            valid_values = logits[finite_logits]

            value_range = (
                (
                    float(valid_values.min().item()),
                    float(valid_values.max().item()),
                )
                if valid_values.numel()
                else None
            )

            raise FloatingPointError(
                f"[VAL LOGITS NONFINITE] "
                f"batch={batch_index} "
                f"sample_ids={[total_samples+i for i in local_ids]} "
                f"labels={labels[bad_rows].cpu().tolist()} "
                f"dtype={logits.dtype} "
                f"NaN={torch.isnan(logits).sum().item()} "
                f"Inf={torch.isinf(logits).sum().item()} "
                f"finite_range={value_range}"
            )


        batch_probabilities = logits.softmax(dim=1)

        if not torch.isfinite(batch_probabilities).all().item():
            raise FloatingPointError(
                f"[VAL SOFTMAX NONFINITE] "
                f"batch={batch_index} "
                f"logits_dtype={logits.dtype}"
            )


        batch_loss = F.cross_entropy(
            logits,
            labels,
            reduction="sum",
        )

        if not torch.isfinite(batch_loss).item():
            raise FloatingPointError(
                f"[VAL LOSS NONFINITE] "
                f"batch={batch_index}"
            )

        total_loss += batch_loss.item()

        total_samples += labels.numel()

        logits_list.append(
            logits.cpu()
        )
        labels_list.append(
            labels.cpu()
        )

    if total_samples == 0:
        raise RuntimeError(
            "评价数据集为空。"
        )

    logits = torch.cat(
        logits_list
    )

    labels = torch.cat(
        labels_list
    ).numpy()

    probabilities = (
        logits.softmax(dim=1)
        .numpy()
    )

    predictions = (
        logits.argmax(dim=1)
        .numpy()
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=range(num_classes),
    )

    tp = np.diag(
        matrix
    ).astype(float)

    support = (
        matrix.sum(axis=1)
        .astype(float)
    )

    predicted = (
        matrix.sum(axis=0)
        .astype(float)
    )

    fp = predicted - tp
    fn = support - tp
    tn = (
        matrix.sum()
        - tp
        - fp
        - fn
    )

    precision = _divide(
        tp,
        predicted,
    )

    recall = _divide(
        tp,
        support,
    )

    specificity = _divide(
        tn,
        tn + fp,
    )

    f1 = _divide(
        2 * precision * recall,
        precision + recall,
    )

    accuracy = float(
        tp.sum()
        / matrix.sum()
    )

    macro_precision = _nanmean(
        precision
    )

    weighted_precision = (
        _weighted_mean(
            precision,
            support,
        )
    )

    macro_recall = _nanmean(
        recall
    )

    macro_f1 = _nanmean(
        f1
    )

    weighted_f1 = _weighted_mean(
        f1,
        support,
    )

    macro_specificity = _nanmean(
        specificity
    )

    mcc = float(
        matthews_corrcoef(
            labels,
            predictions,
        )
    )


    per_class_roc_auc = []

    for index in range(
        num_classes
    ):
        binary = (
            labels == index
        ).astype(int)

        if np.unique(
            binary
        ).size < 2:
            per_class_roc_auc.append(
                float("nan")
            )
            continue

        per_class_roc_auc.append(
            float(
                roc_auc_score(
                    binary,
                    probabilities[
                        :,
                        index,
                    ],
                )
            )
        )

    macro_roc_auc = _nanmean(
        per_class_roc_auc
    )

    per_class = {
        "precision": (
            precision.tolist()
        ),
        "recall": (
            recall.tolist()
        ),
        "specificity": (
            specificity.tolist()
        ),
        "f1": (
            f1.tolist()
        ),
        "roc_auc": (
            per_class_roc_auc
        ),

        "support": (
            support.astype(int)
            .tolist()
        ),
    }

    metrics = {
        "class_names": list(
            class_names
        ),

        "loss": float(
            total_loss
            / total_samples
        ),

        "accuracy": (
            accuracy
        ),

        "macro_precision": (
            macro_precision
        ),
        "weighted_precision": (
            weighted_precision
        ),

        "macro_recall": (
            macro_recall
        ),

        "macro_f1": (
            macro_f1
        ),
        "weighted_f1": (
            weighted_f1
        ),

        "macro_specificity": (
            macro_specificity
        ),

        "mcc": (
            mcc
        ),

        "macro_roc_auc": (
            macro_roc_auc
        ),

        "per_class": (
            per_class
        ),

        "confusion_matrix": (
            matrix.astype(int)
            .tolist()
        ),
    }

    if not (
        include_full_probability_metrics
    ):
        return metrics


    one_hot = np.eye(
        num_classes,
        dtype=np.int64,
    )[labels]

    weighted_roc_auc = (
        _weighted_mean(
            per_class_roc_auc,
            support,
        )
    )

    micro_roc_auc = float(
        roc_auc_score(
            one_hot.ravel(),
            probabilities.ravel(),
        )
    )

    per_class_ap = []

    for index in range(
        num_classes
    ):
        binary = (
            labels == index
        ).astype(int)

        if np.unique(
            binary
        ).size < 2:
            per_class_ap.append(
                float("nan")
            )
            continue

        per_class_ap.append(
            float(
                average_precision_score(
                    binary,
                    probabilities[
                        :,
                        index,
                    ],
                )
            )
        )

    macro_ap = _nanmean(
        per_class_ap
    )

    weighted_ap = _weighted_mean(
        per_class_ap,
        support,
    )

    micro_ap = float(
        average_precision_score(
            one_hot.ravel(),
            probabilities.ravel(),
        )
    )

    metrics.update(
        {
            "weighted_roc_auc": (
                weighted_roc_auc
            ),
            "micro_roc_auc": (
                micro_roc_auc
            ),

            "macro_ap": (
                macro_ap
            ),
            "weighted_ap": (
                weighted_ap
            ),
            "micro_ap": (
                micro_ap
            ),


            "prediction_data": {
                "labels": (
                    labels.astype(int)
                    .tolist()
                ),
                "predictions": (
                    predictions.astype(int)
                    .tolist()
                ),
                "probabilities": (
                    probabilities
                    .astype(float)
                    .tolist()
                ),
            },
        }
    )

    metrics[
        "per_class"
    ][
        "ap"
    ] = per_class_ap

    return metrics


def format_evaluation(
    metrics: dict[str, Any],
) -> str:


    keys = (
        "loss",
        "accuracy",
        "macro_precision",
        "weighted_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "macro_specificity",
        "mcc",
        "macro_roc_auc",
    )

    names = (
        "Loss",
        "ACC",
        "Macro-P",
        "Weighted-P",
        "Macro-R",
        "Macro-F1",
        "Weighted-F1",
        "Macro-SPE",
        "MCC",
        "Macro ROC-AUC",
    )

    lines = [
        " | ".join(
            f"{name} "
            f"{metrics[key]:.4f}"
            for name, key
            in zip(
                names,
                keys,
            )
        )
    ]

    lines.append(
        "Class       PREC     RECALL   "
        "SPE      F1       AUC      N"
    )

    values = metrics[
        "per_class"
    ]

    for index, name in enumerate(
        metrics["class_names"]
    ):
        auc = values[
            "roc_auc"
        ][index]

        lines.append(
            f"{name:<10} "
            f"{values['precision'][index]:.4f}   "
            f"{values['recall'][index]:.4f}   "
            f"{values['specificity'][index]:.4f}   "
            f"{values['f1'][index]:.4f}   "
            f"{auc:.4f}   "
            f"{values['support'][index]}"
        )

    lines.append(
        "Confusion matrix:"
    )

    lines.extend(
        str(row)
        for row in metrics[
            "confusion_matrix"
        ]
    )

    return "\n".join(
        lines
    )
