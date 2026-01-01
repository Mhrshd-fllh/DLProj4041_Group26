import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set seeds for reproducibility across python/numpy/torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Note: full determinism is not always guaranteed across all ops/hardware
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
