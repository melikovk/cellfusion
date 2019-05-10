import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
from skimage.io import imshow
import time
import glob

def main():
    img = cv2.imread('/home/fast/Automate/20x/nuclei/image000012.tif', cv2.IMREAD_ANYDEPTH)
    # print(img.shape)
    # img.dtype
    img = (255*(img.astype('float32') - img.min())/(img.max()-img.min())).astype('uint8')
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    mser = cv2.MSER_create(1, 20*20, 60*60,0.5)
    t1 = time.time_ns()
    regions, _ = mser.detectRegions(img)
    print((time.time_ns()-t1)/1e6)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    _ = cv2.polylines(cimg, hulls, 1, (0, 255, 0))
    len(regions)
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    print('OK')

if __name__ == '__main__':
    main()
x = {'test': 1, 'bounds':2}
x
files = glob.glob("/home/storage/kamran/DATA/Kaggle/nuclei/stage1_train/*/masks/*")
len(files)
y = x
y['test']
x['test']=3
y['bounds'] = 1
x['bounds']
y['start'] = 4
x['start']
