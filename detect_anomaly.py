import os
import cv2
import numpy as np
from core import preprocess, utils
from absl.flags import FLAGS
from absl import app, flags, logging
import time
import math

flags.DEFINE_string('roi', './cropped/bottom_2', 'directory for ROI images')
flags.DEFINE_string('output', './detected/', 'directory to save DETECTED images')
flags.DEFINE_string('process_mode', 'gaussianCV', 'gaussianCV or custom')
flags.DEFINE_string('detect_mode', 'blob', 'blob or find_contours')

def create_blob_detector():
    params = cv2.SimpleBlobDetector_Params()

    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByArea = True
    params.filterByColor = True
    params.blobColor = 0
    params.minArea = 3.14159 * 0.1 * 0.1
    params.maxArea = 3.14159 * 165 * 165*0.7*0.7
    params.minDistBetweenBlobs = 5

    detector = cv2.SimpleBlobDetector_create(params)

    return detector
    
def find_contours(img,copy):
    contours,hierarchy=cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    s=0
    for c in contours:
        rect = cv2.minAreaRect(c) 
        box = cv2.boxPoints(rect) 
        box = np.int0(box) 
        (x,y)=box[1]-box[2]
        a=math.sqrt(x*x+y*y)
        (i,j)=box[2]-box[3]
        b=math.sqrt(i*i+j*j)
        w=min(a,b)
        h=max(a,b)
        draw = cv2.drawContours(copy,[box],-1,(255,255,255),3)
        s+=1
        print(box)
    return draw,s

def detect_blob(surface, copy):
    detector = create_blob_detector()
    keypoints = detector.detect(surface)
    if len(keypoints) ==0:
        print('No anomaly detected!')
    else:
        im_with_keypoints = cv2.drawKeypoints(copy, keypoints, np.array([]), (0, 0, 255),
                                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return im_with_keypoints, len(keypoints)


def detect_single_image(roi_path, preprocess_mode, detect_mode):
    roi = cv2.imread(roi_path, 0)
    copy = np.copy(roi)
    
    roi = preprocess.smoothen(roi, preprocess_mode)
    roi = preprocess.binary_thresholding(roi, mode='adaptive', filter_size=7, cons=-10)
    #roi = utils.remove_bottom_border(roi, 'big')
    
    #center_surface = utils.remove_bottom_border(roi, 'small')
    if detect_mode == 'blob':
        detected, count = detect_blob(roi, copy)
    else:
        detected, count = find_contours(roi, copy)
    #print((detected-roi).max())
    return detected, count
    
def detect_multiple_images(input_dir, output_dir, process_mode, detect_mode):
    file_names = [file_name for file_name in os.listdir(input_dir)]
    for file_name in file_names:
        detected, _ = detect_single_image(os.path.join(input_dir, file_name), process_mode, detect_mode)
        cv2.imwrite(os.path.join(output_dir, file_name), detected)
    
def main(_argv):
    if not os.path.isdir(FLAGS.output):
        os.mkdir(FLAGS.output)
    detect_multiple_images(FLAGS.roi, FLAGS.output, FLAGS.process_mode, FLAGS.detect_mode)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass    
    
