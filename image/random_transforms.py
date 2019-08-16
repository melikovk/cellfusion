import scipy.ndimage as ndimage
import numpy as np

class RandomGamma:
    def __init__(self, min_gamma=.5, max_gamma=1., seed=None):
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.seed = seed

    def __call__(self, img):
        if self.seed:
            np.random.seed(self.seed)
        gamma = np.random.uniform(low=self.min_gamma, high=self.max_gamma)
        return img**gamma
