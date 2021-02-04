import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from core.preprocess import binary_thresholding
from scipy.ndimage import histogram

def create_circle_mask(image, mask_type='big', background_color='white'):
    if background_color == 'black':
        copy = np.zeros(image.shape, dtype=np.uint8)
        circle_color = (1, 1, 1)
    elif background_color == 'white':
        copy = np.ones(image.shape, dtype=np.uint8)*image.max()
        circle_color = (0, 0, 0)
    r = copy.shape[0] // 2
    radius=r
    if mask_type=='big':
        radius = r-3
    else:
        radius = int(0.7*r)

    circle = cv2.circle(copy, (r, r), radius, circle_color, -1)

    return np.asarray(circle, dtype=np.uint8)

def remove_bottom_border(img, mask_size='big'):
    
    black_mask = create_circle_mask(img, mask_type=mask_size, background_color='black')
    white_mask = create_circle_mask(img, mask_type=mask_size, background_color='white')
    
    img = img * black_mask
    img = img + white_mask
    return img

def visualize_multiple_blobs(img, low, high,step):
    num_step = (high-low)//step
    for i in range(num_step):
        plt.figure(figsize=(20,20))
        plt.subplot(num_step, 1, i+1)
        plt.imshow(binary_thresholding(img, low+i*step), cmap='gray')

def plot_histogram(proc):
    hist = histogram(proc, proc.min(), proc.max(), proc.max())
    plt.plot(hist)

