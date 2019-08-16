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

class RandomContrast:
    def __init__(self, blur_max_sigma = 4, deblur_max_sigma = 4, deblur_k=.6, seed=None):
        self.blur_max_sigma = blur_max_sigma
        self.deblur_max_sigma = deblur_max_sigma
        self.deblur_k = deblur_k
        self.seed = seed

    def __call__(self, img):
        if self.seed:
            np.random.seed(self.seed)
        isblur = np.random.randint(2) == 1
        if isblur:
            sigma = np.random.uniform(low=0, high=self.blur_max_sigma)
            out = ndimage.gaussian_filter1d(img, sigma, axis=-1, mode='constant')
            out = ndimage.gaussian_filter1d(out, sigma, axis=-2, mode='constant')
        else:
            sigma = np.random.uniform(low=0, high=self.deblur_max_sigma)
            out = ndimage.gaussian_filter1d(img, sigma, axis=-1, mode='constant')
            out = ndimage.gaussian_filter1d(out, sigma, axis=-2, mode='constant')
            out = img - out*self.deblur_k/(1-self.deblur_k)
        return out

class AutoContrast:
    def __init__(self, background = 'median', max_percentile=.95):
        self.background = background
        self.max_percentile = max_percentile

    def __call__(self, img):
        if self.background == 'median':
            bkg = np.median(img, axis=(-2,-1))
        else:
            bkg = np.mean(img, axis=(-2,-1))
        max_val = np.quantile(img, self.max_percentile, axis = (-2,-1))
        return (img - bkg)/max_val
