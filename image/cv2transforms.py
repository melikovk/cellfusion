import cv2
import numpy as np


class Zoom:
    def __init__(self, zoom, interpolation=cv2.INTER_LINEAR):
        if np.isscalar(zoom):
            self.fx = zoom
            self.fy = zoom
        else:
            self.fx, self.fy = zoom
        self.interpolation = interpolation

    def __call__(self, img, boxes=None):
        if len(img.shape)==2:
            img = np.expand_dims(img, 0)
        out_img = []
        for c in range(img.shape[0]):
            out_img.append(cv2.resize(img[c], dsize=(0,0), fx=self.fx, fy=self.fy, interpolation=self.interpolation))
        out_img = np.squeeze(np.stack(out_img))
        fx = out_img.shape[-2]/img.shape[-2]
        fy = out_img.shape[-1]/img.shape[-1]
        if boxes is not None:
            out_boxes = boxes*np.array([fx,fy]*2)
            return out_img, out_boxes
        else:
            return out_img

class Sharpen:
    def __init__(self, deblur_sigma=3, deblur_k=.6):
        self.deblur_sigma = deblur_sigma
        self.deblur_k = deblur_k

    def __call__(self, img, boxes = None):
        if len(img.shape)==2:
            img = np.expand_dims(img, 0)
        out = np.zeros_like(img)
        for c in range(img.shape[0]):
            out[c] = cv2.GaussianBlur(img[c], (0,0), self.deblur_sigma)
            out[c] = (img[c] - out[c]*self.deblur_k)/(1-self.deblur_k)
        out = np.squeeze(out)
        return out if boxes is None else (out, boxes)

class GaussianBlur:
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __call__(self, img, boxes = None):
        if len(img.shape)==2:
            img = np.expand_dims(img, 0)
        out = np.zeros_like(img)
        for c in range(img.shape[0]):
            out[c] = cv2.GaussianBlur(img[c], (0,0), self.sigma)
        out = np.squeeze(out)
        return out if boxes is None else (out, boxes)

class Gamma:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, boxes = None):
        if len(img.shape)==2:
            img = np.expand_dims(img, 0)
        out = np.zeros_like(img)
        for c in range(img.shape[0]):
            out[c] = cv2.pow(img[c]-img[c].min(), self.gamma)
        out = np.squeeze(out)
        return out if boxes is None else (out, boxes)

class CorrectShading:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img, boxes = None):
        if len(img.shape)==2:
            img = np.expand_dims(img, 0)
        out = np.zeros_like(img)
        for c in range(img.shape[0]):
            out[c] = img[c] - np.min(img[c])
            out[c] = out[c]/cv2.GaussianBlur(out[c], (0,0), self.sigma)
            # out[c] = out[c]/cv2.blur(out[c], (self.sigma, self.sigma))
        out = np.squeeze(out)
        return out if boxes is None else (out, boxes)

def Typecast(newtype):
    return lambda x: x.astype(newtype)

def ToRGB():
    return lambda x: np.stack([x]*3, axis=-1)

# def RandomRotation(angle_range, autocrop=True):
#     if isinstance(angle_range, int):
#         angle_range = (-np.abs(angle_range), np.abs(angle_range))
#     if autocrop:
#         def func(img):
#             alpha = angle_range[0] + np.random.random()*(angle_range[1]-angle_range[0])
#             alphaR = alpha*np.pi/180
#             # Find scaling factor to fit image after rotation
#             h, w = img.shape[:2]
#             if w > h:
#                 scale = w/h*np.sin(np.abs(alphaR))+np.cos(np.abs(alphaR))
#             else:
#                 scale = h/w*np.sin(np.abs(alphaR))+np.cos(np.abs(alphaR))
#             mat = cv2.getRotationMatrix2D((w/2, h/2), alpha, scale)
#             return cv2.warpAffine(img, mat, img.shape[:2])
#     else:
#         def func(img):
#             alpha = angle_range[0] + np.random.random()*(angle_range[1]-angle_range[0])
#             mat = cv2.getRotationMatrix2D((img.shape[0]/2, img.shape[1]/2),alpha, 1)
#             return cv2.warpAffine(img, mat, img.shape[:2])
#     return func
