import cv2
from skimage.exposure import adjust_gamma
import numpy as np

def Resize(newsize, interpolation = cv2.INTER_LINEAR):
    if isinstance(newsize, int):
        newsize = (newsize, newsize)
    return lambda x: cv2.resize(x, newsize, interpolation)

def Typecast(newtype):
    return lambda x: x.astype(newtype)

def ToRGB():
    return lambda x: np.stack([x]*3, axis=-1)

def RandomHflip(prob=.5):
    return lambda x: x if np.random.random(1) > prob else cv2.flip(x, 1)

def RandomVflip(prob=.5):
    return lambda x: x if np.random.random(1) > prob else cv2.flip(x, 0)

def RandomRotation(angle_range, autocrop=True):
    if isinstance(angle_range, int):
        angle_range = (-np.abs(angle_range), np.abs(angle_range))
    if autocrop:
        def func(img):
            alpha = angle_range[0] + np.random.random()*(angle_range[1]-angle_range[0])
            alphaR = alpha*np.pi/180
            # Find scaling factor to fit image after rotation
            h, w = img.shape[:2]
            if w > h:
                scale = w/h*np.sin(np.abs(alphaR))+np.cos(np.abs(alphaR))
            else:
                scale = h/w*np.sin(np.abs(alphaR))+np.cos(np.abs(alphaR))
            mat = cv2.getRotationMatrix2D((w/2, h/2), alpha, scale)
            return cv2.warpAffine(img, mat, img.shape[:2])
    else:
        def func(img):
            alpha = angle_range[0] + np.random.random()*(angle_range[1]-angle_range[0])
            mat = cv2.getRotationMatrix2D((img.shape[0]/2, img.shape[1]/2),alpha, 1)
            return cv2.warpAffine(img, mat, img.shape[:2])
    return func

def Gamma(gamma):
    return lambda x: adjust_gamma(x, gamma)

def AutoContrast():
    return lambda x: (x -x.mean(axis=(0,1))) / x.std(axis=(0,1))
