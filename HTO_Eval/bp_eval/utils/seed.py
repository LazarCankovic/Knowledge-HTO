"""Global seeding so every run is reproducible end-to-end."""
import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set the seed for every RNG this pipeline touches.

    Called once at the top of train.py, before the split is built or any
    model is instantiated, so a run with the same config+seed is
    byte-for-byte reproducible (modulo non-determinism in optional
    GPU-backed models, which is documented separately in model wrappers).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch  # optional dependency, only needed for LSTM/BERT models

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
