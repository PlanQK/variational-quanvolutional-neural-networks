import random

import torch
from pennylane import numpy as np


class GeneratorUtils:

    def generate_seeds(size):
        seeds = []
        for n in range(size):
            seeds.append(np.random.randint(low=0, high=10e6))
        return seeds


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
