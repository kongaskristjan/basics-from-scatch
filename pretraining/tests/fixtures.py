import random

import numpy as np
import pytest
import torch


@pytest.fixture
def fix_seeds() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
