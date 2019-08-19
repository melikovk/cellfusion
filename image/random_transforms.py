import scipy.ndimage as ndimage
import numpy as np

class RandomGamma:
    def __init__(self, min_gamma=.5, max_gamma=1., seed=None):
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.seed = seed

    def __call__(self, img):
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
            out = (img - out*self.deblur_k)/(1-self.deblur_k)
        return out

class RandomBlur:
    def __init__(self, blur_max_sigma = 4, seed=None):
        self.blur_max_sigma = blur_max_sigma
        self.seed = seed

    def __call__(self, img):
        np.random.seed(self.seed)
        sigma = np.random.uniform(low=0, high=self.blur_max_sigma)
        out = ndimage.gaussian_filter1d(img, sigma, axis=-1, mode='constant')
        out = ndimage.gaussian_filter1d(out, sigma, axis=-2, mode='constant')
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

class RandomFlip:
    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, img, boxes):
        np.random.seed(self.seed)
        out_img = img
        out_boxes = boxes
        choice = np.random.randint(3)
        if choice == 0:
            out_img = np.flip(out_img, axis=-2)
            out_boxes[:,0] = out_img.shape[-2]-1-out_boxes[:,0]
        elif choice == 1:
            out_img = np.flip(out_img, axis=-1)
            out_boxes[:,1] = out_img.shape[-1]-1-out_boxes[:,1]
        return out_img, out_boxes

class RandomZoom:
    def __init__(self, min_zoom = .5, max_zoom = 2, keep_aspect = True, seed = None):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.keep_aspect = keep_aspect
        self.seed = seed

    def __call__(self, img, boxes):
        np.random.seed(self.seed)
        if self.keep_aspect:
            x_zoom = y_zoom = np.random.uniform(low=self.min_zoom, high=self.max_zoom)
        else:
            x_zoom = np.random.uniform(low=self.min_zoom, high=self.max_zoom)
            y_zoom = np.random.uniform(low=self.min_zoom, high=self.max_zoom)
        zooms = np.ones(img.ndim)
        zooms[-2:] = (x_zoom, y_zoom)
        out_img = ndimage.zoom(img, zooms)
        x_zoom = out_img.shape[-2]/img.shape[-2]
        y_zoom = out_img.shape[-1]/img.shape[-1]
        out_boxes = boxes*np.array([x_zoom,y_zoom]*2)
        return out_img, out_boxes
