import cv2
import os
import numpy as np
import time
from core import preprocess
from absl.flags import FLAGS
from absl import app, flags, logging

flags.DEFINE_string('data', './data/bottom', 'directory for raw input images')
flags.DEFINE_string('output', './cropped/bottom_2', 'directory to save cropped images')


def detect_circle(image_path):
    img = cv2.imread(image_path, 0)
    img = preprocess.resize_to_hd(img)
    
    copy = np.copy(img)
    #copy = preprocess.enhance_contrast(copy)
    copy = preprocess.smoothen(copy, filter='gaussian')

    try:
        circles = cv2.HoughCircles(copy, cv2.HOUGH_GRADIENT, minDist=150, dp=1.2, minRadius=50, maxRadius=250)
        circles = np.uint16(np.around(circles))
    except:
        print('Error at: '+ image_path)
    else:
        for (x, y, r) in circles[0]:    
            crop = img[y - r:y + r, x - r:x + r]
            print(f'Succes: {image_path} with ROI size: {crop.shape}')
    
    return crop, image_path.split('/')[-1]
    
def save_crop(crop, file_name, new_dir):
    
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    
    new_path = os.path.join(new_dir, file_name)
    cv2.imwrite(new_path, crop)
    print(f'Save ROI to {new_path}')    
    
    return 1
        

def main(_argv):
    input_dir = FLAGS.data
    output_dir = FLAGS.output
    
    count =0
    
    for file in os.listdir(input_dir):
        path = os.path.join(input_dir, file)
        
        exc_time = 0
        prev_time = time.time()
        
        crop, file_name =  detect_circle(path)
        
        curr_time  = time.time()
        exc_time += curr_time - prev_time
        
        if crop is not None:
            count += save_crop(crop, file_name, output_dir)
        else:
            count = count
        
    print(f"Avergage execution time: {exc_time/len(os.listdir(input_dir))}")    
    print(f"Number of successful detected ROI: {str(count)}, ({100*count/len(os.listdir(input_dir))}%)")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass    
    
