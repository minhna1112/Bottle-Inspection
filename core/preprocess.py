import cv2
import numpy as np
import scipy.ndimage as nd


def convert_1(image):
    return 11/27*(image-120)+200

def convert_2(image):
    return 37/18*(image-30)+15

def convert_3(image):
    return 0.5*(image)

def enhance_contrast(img):
    index_1 = np.where((img>=120) & (img<=255))
    index_2 = np.where((img>=30) & (img<120))
    index_3 = np.where(img < 30)

    new_im = np.copy(img)
    new_im[index_1] = convert_1(img[index_1])
    new_im[index_2] = convert_2(img[index_2])
    new_im[index_3] = convert_3(img[index_3])

    return new_im
    
def smoothen(img, filter='median'):
    if filter=='median':
        return cv2.medianBlur(img, 7)
    elif filter=='gaussian':
        return nd.filters.gaussian_filter(img, sigma=1.)
    elif filter=='gaussianCV':
        return cv2.GaussianBlur(img, (3,3), 0)
    else:
        kernel = np.array([[-1,-1, -1],
                      [-1,10, -1],
                      [-1, -1, -1]])
        return cv2.filter2D(img , -1, kernel)

def resize_to_hd(img):
    height, width = img.shape
    if height > width:
        return cv2.resize(img, (1080,1920))
    else:
        return cv2.resize(img, (1920,1080))
        
def binary_thresholding(roi, threshold=127, mode='adaptive', filter_size=3, cons=0):
    if mode=='adaptive':
        thresholded = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,filter_size,cons)
    else:
        thresholded = np.copy(roi)
        thresholded[roi <= threshold] = 0
        thresholded[roi > threshold] = 255

    return thresholded

