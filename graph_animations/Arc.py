import numpy as np


class Arc:
    def __init__(self, v1: np.ndarray, v2: np.ndarray):
        self.v1 = v1
        self.v2 = v2

    def at(self, t: float):
        raise NotImplementedError()
