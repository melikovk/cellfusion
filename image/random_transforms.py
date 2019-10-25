import numpy as np
import cv2

class RandomGamma:
    """ Apply random Gamma correction to an image
    if min_gamma and max_gamma are scalars the functional object will work
    with images with any number of channels. If min_gamma and max_gamma are
    lists they should have the same length and the instance will work only
    with images with the number of channels equal to the length of min_gamma
    and max_gamma. Multi-channel images shoud be in CxWxH format
    """
    def __init__(self, min_gamma=.5, max_gamma=1., seed=None, channels_independent = False):
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.gen = np.random.RandomState(seed)
        self.independent = channels_independent


    def __call__(self, img, boxes=None):
        if len(img.shape) == 2:
            img = np.expand_dims(img, 0)
        out = np.zeros_like(img)
        if self.independent:
            gammas = self.gen.uniform(low=self.min_gamma, high=self.max_gamma, size=img.shape[0])
        else:
            gammas = [self.gen.uniform(low=self.min_gamma, high=self.max_gamma)]*img.shape[0]
        for c in range(img.shape[0]):
            out[c] = cv2.pow(img[c]-img[c].min(), gammas[c])
        out = np.squeeze(out)
        return  out if boxes is None else (out, boxes)

class RandomContrast:
    def __init__(self, blur_max_sigma = 4, deblur_max_sigma = 4, deblur_k=.6, seed=None, channels_independent = False):
        self.blur_max_sigma = blur_max_sigma
        self.deblur_max_sigma = deblur_max_sigma
        self.deblur_k = deblur_k
        self.gen = np.random.RandomState(seed)
        self.independent = channels_independent

    def __call__(self, img, boxes=None):
        if len(img.shape)==2:
            img = np.expand_dims(img, 0)
        out = np.zeros_like(img)
        if self.independent:
            for c in range(img.shape[0]):
                isblur = self.gen.randint(2) == 1
                if isblur:
                    sigma = self.gen.uniform(low=0, high=self.blur_max_sigma)
                    out[c] = cv2.GaussianBlur(img[c], (0,0), sigma)
                else:
                    sigma = self.gen.uniform(low=0, high=self.deblur_max_sigma)
                    out[c] = cv2.GaussianBlur(img[c], (0,0), sigma)
                    out[c] = (img[c] - out[c]*self.deblur_k)/(1-self.deblur_k)
        else:
            isblur = self.gen.randint(2) == 1
            if isblur:
                sigma = self.gen.uniform(low=0, high=self.blur_max_sigma)
                for c in range(img.shape[0]):
                    out[c] = cv2.GaussianBlur(img[c], (0,0), sigma)
            else:
                sigma = self.gen.uniform(low=0, high=self.deblur_max_sigma)
                for c in range(img.shape[0]):
                    out[c] = cv2.GaussianBlur(img[c], (0,0), sigma)
                    out[c] = (img[c] - out[c]*self.deblur_k)/(1-self.deblur_k)
        out = np.squeeze(out)
        return out if boxes is None else (out, boxes)

class RandomSharpen:
    def __init__(self, deblur_max_sigma = 4, deblur_k=.6, seed=None, channels_independent = False):
        self.deblur_max_sigma = deblur_max_sigma
        self.deblur_k = deblur_k
        self.gen = np.random.RandomState(seed)
        self.independent = channels_independent

    def __call__(self, img, boxes=None):
        if len(img.shape)==2:
            img = np.expand_dims(img, 0)
        out = np.zeros_like(img)
        if self.independent:
            sigmas = self.gen.uniform(low=0, high=self.deblur_max_sigma, size=img.shape[0])
        else:
            sigmas = [self.gen.uniform(low=0, high=self.deblur_max_sigma)]*img.shape[0]
        for c in range(img.shape[0]):
            out[c] = cv2.GaussianBlur(img[c], (0,0), sigmas[c])
            out[c] = (img[c] - out[c]*self.deblur_k)/(1-self.deblur_k)
        out = np.squeeze(out)
        return out if boxes is None else (out, boxes)

class RandomBlur:
    def __init__(self, blur_max_sigma = 4, seed=None, channels_independent = False):
        self.blur_max_sigma = blur_max_sigma
        self.gen = np.random.RandomState(seed)
        self.independent = channels_independent

    def __call__(self, img, boxes=None):
        if len(img.shape)==2:
            img = np.expand_dims(img, 0)
        out = np.zeros_like(img)
        if self.independent:
            sigmas = self.gen.uniform(low=0, high=self.blur_max_sigma, size=img.shape[0])
        else:
            sigmas = [self.gen.uniform(low=0, high=self.blur_max_sigma)]*img.shape[0]
        for c in range(img.shape[0]):
            out[c] = cv2.GaussianBlur(img[c], (0,0), sigmas[c])
        out = np.squeeze(out)
        return out if boxes is None else (out, boxes)

class AutoContrast:
    def __init__(self, background = 'median', max_percentile=.95):
        self.background = background
        self.max_percentile = max_percentile

    def __call__(self, img, boxes=None):
        if self.background == 'median':
            bkg = np.median(img, axis=(-2,-1), keepdims=True)
        else:
            bkg = np.mean(img, axis=(-2,-1))
        max_val = np.quantile(img, self.max_percentile, axis = (-2,-1), keepdims=True).astype(img.dtype)
        out = (img - bkg)/max_val
        return out if boxes is None else (out, boxes)

class RandomFlip:
    def __init__(self, seed=None):
        self.gen = np.random.RandomState(seed)

    def __call__(self, img, boxes = None):
        if len(img.shape)==2:
            img = np.expand_dims(img, 0)
        choice = self.gen.randint(3)
        if choice == 0:
            out_img = np.zeros_like(img)
            if boxes is not None:
                out_boxes = np.copy(boxes)
                out_boxes[:,0] = img.shape[-2]-1-out_boxes[:,0]-out_boxes[:,2]
            for c in range(img.shape[0]):
                out_img[c] = cv2.flip(img[c], flipCode=0)
        elif choice == 1:
            out_img = np.zeros_like(img)
            if boxes is not None:
                out_boxes = np.copy(boxes)
                out_boxes[:,1] = img.shape[-1]-1-out_boxes[:,1]-out_boxes[:,3]
            for c in range(img.shape[0]):
                out_img[c] = cv2.flip(img[c], flipCode=1)
        else:
            out_img = np.copy(img)
            if boxes is not None:
                out_boxes = np.copy(boxes)
        out_img = np.squeeze(out_img)
        return out_img if boxes is None else (out_img, out_boxes)

class RandomZoom:
    def __init__(self, choices = (0.5,1,2), keep_aspect = True, seed = None, interpolation = cv2.INTER_CUBIC):
        self.choices = choices
        self.keep_aspect = keep_aspect
        self.gen = np.random.RandomState(seed)
        self.interpolation = interpolation

    def __call__(self, img, boxes = None):
        if self.keep_aspect:
            fx = fy = self.gen.choice(self.choices)
        else:
            fx = self.gen.choice(self.choices)
            fy = self.gen.choice(self.choices)
        if len(img.shape)==2:
            img = np.expand_dims(img, 0)
        out_img = []
        for c in range(img.shape[0]):
            out_img.append(cv2.resize(img[c], dsize=(0,0), fx=fx, fy=fy, interpolation=self.interpolation))
        out_img = np.squeeze(np.stack(out_img))
        fx = out_img.shape[-2]/img.shape[-2]
        fy = out_img.shape[-1]/img.shape[-1]
        if boxes is not None:
            out_boxes = boxes*np.array([fx,fy]*2)
        return out_img if boxes is None else (out_img, out_boxes)
