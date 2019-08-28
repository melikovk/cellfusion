import scipy.ndimage as ndimage
import numpy as np
from PIL import Image
import cv2

class RandomGamma:
    def __init__(self, min_gamma=.5, max_gamma=1., seed=None):
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.gen = np.random.RandomState(seed)

    def __call__(self, img):
        gamma = self.gen.uniform(low=self.min_gamma, high=self.max_gamma)
        return cv2.pow(img, gamma)

class RandomContrast:
    def __init__(self, blur_max_sigma = 4, deblur_max_sigma = 4, deblur_k=.6, seed=None):
        self.blur_max_sigma = blur_max_sigma
        self.deblur_max_sigma = deblur_max_sigma
        self.deblur_k = deblur_k
        self.gen = np.random.RandomState(seed)

    def __call__(self, img):
        isblur = self.gen.randint(2) == 1
        if isblur:
            sigma = self.gen.uniform(low=0, high=self.blur_max_sigma)
            # out = ndimage.gaussian_filter1d(img, sigma, axis=-1, mode='constant')
            # out = ndimage.gaussian_filter1d(out, sigma, axis=-2, mode='constant')
            out = cv2.GaussianBlur(img, (0,0), sigma)
        else:
            sigma = self.gen.uniform(low=0, high=self.deblur_max_sigma)
            # out = ndimage.gaussian_filter1d(img, sigma, axis=-1, mode='constant')
            # out = ndimage.gaussian_filter1d(out, sigma, axis=-2, mode='constant')
            out = cv2.GaussianBlur(img, (0,0), sigma)
            out = (img - out*self.deblur_k)/(1-self.deblur_k)
        return out

class RandomSharpen:
    def __init__(self, blur_max_sigma = 4, deblur_max_sigma = 4, deblur_k=.6, seed=None):
        self.deblur_max_sigma = deblur_max_sigma
        self.deblur_k = deblur_k
        self.gen = np.random.RandomState(seed)

    def __call__(self, img):
        sigma = self.gen.uniform(low=0, high=self.deblur_max_sigma)
        out = cv2.GaussianBlur(img, (0,0), sigma)
        out = (img - out*self.deblur_k)/(1-self.deblur_k)
        return out

class RandomBlur:
    def __init__(self, blur_max_sigma = 4, seed=None):
        self.blur_max_sigma = blur_max_sigma
        self.gen = np.random.RandomState(seed)

    def __call__(self, img):
        sigma = self.gen.uniform(low=0, high=self.blur_max_sigma)
        out = cv2.GaussianBlur(img, (0,0), sigma)
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
        self.gen = np.random.RandomState(seed)

    def __call__(self, img, boxes):
        out_img = img
        out_boxes = boxes
        choice = self.gen.randint(3)
        if choice == 0:
            out_img = cv2.flip(out_img, flipCode=0)
            out_boxes[:,0] = out_img.shape[-2]-1-out_boxes[:,0]-out_boxes[:,2]
        elif choice == 1:
            out_img = cv2.flip(out_img, flipCode=1)
            out_boxes[:,1] = out_img.shape[-1]-1-out_boxes[:,1]-out_boxes[:,3]
        return out_img, out_boxes

class RandomZoom:
    def __init__(self, min_zoom = .5, max_zoom = 2, keep_aspect = True, seed = None):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.keep_aspect = keep_aspect
        self.gen = np.random.RandomState(seed)

    def __call__(self, img, boxes):
        if self.keep_aspect:
            fx = fy = self.gen.uniform(low=self.min_zoom, high=self.max_zoom)
        else:
            fx = self.gen.uniform(low=self.min_zoom, high=self.max_zoom)
            fy = self.gen.uniform(low=self.min_zoom, high=self.max_zoom)
        # new_x, new_y = int(img.shape[-2]*x_zoom), int(img.shape[-1]*y_zoom)
        # out_img = np.asarray(Image.fromarray(img).resize((new_y, new_x), Image.BICUBIC))
        out_img = cv2.resize(img, dsize=(0,0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        fx = out_img.shape[-2]/img.shape[-2]
        fy = out_img.shape[-1]/img.shape[-1]
        out_boxes = boxes*np.array([fx,fy]*2)
        return out_img, out_boxes
