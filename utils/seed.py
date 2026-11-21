from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(
    seed: int,
    deterministic: bool = True,
) -> None:

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:

    del worker_id

    worker_seed = torch.initial_seed() % (2**32)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_generator(seed: int) -> torch.Generator:

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator
