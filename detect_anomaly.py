import os
import cv2
import numpy as np
from core import preprocess, utils
from absl.flags import FLAGS
from absl import app, flags, logging
import time

flags.DEFINE_string('roi', './cropped/bottom_2', 'directory for ROI images')
flags.DEFINE_string('output', './detected/', 'directory to save DETECTED images')

def create_detector(minThres, maxThres, step, repeat_ratio):
    params = cv2.SimpleBlobDetector_Params()

    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByArea = True
    params.filterByColor = True
    params.blobColor = 0
    params.minThreshold = minThres
    params.thresholdStep = step
    params.maxThreshold =   maxThres#int(2.2*proc.max()-300)#
    params.minArea = 3.14159 * 0.1 * 0.1
    params.maxArea = 3.14159 * 165 * 165*0.7*0.7
    params.minRepeatability = int(repeat_ratio*(params.maxThreshold - params.minThreshold) / params.thresholdStep)
    #params.minRepeatability = 1
    params.minDistBetweenBlobs = 5

    detector = cv2.SimpleBlobDetector_create(params)

    return detector

def detect_single_image(roi_path, full_detector, center_detector):
    roi = cv2.imread(roi_path, 0)
    copy = np.copy(roi)
    
    roi = preprocess.smoothen(roi, 'gaussian')
    
    full_surface = utils.remove_bottom_border(roi, 'big')
    center_surface = utils.remove_bottom_border(roi, 'small')

    
    full_keypoints = full_detector.detect(full_surface)
    center_keypoints = center_detector.detect(center_surface)
    
    if len(full_keypoints) + len(center_keypoints) ==0:
        print('No anomaly detected!')
    else:
        im_with_keypoints = cv2.drawKeypoints(copy, center_keypoints, np.array([]), (255, 0, 0),
                                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, full_keypoints, np.array([]), (0, 0, 255),
                                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return im_with_keypoints
    
    

def detect_multiple_images(input_dir, output_dir, full_detector, center_detector):
    file_names = [file_name for file_name in os.listdir(input_dir)]
    for file_name in file_names:
        detected = detect_single_image(os.path.join(input_dir, file_name), full_detector, center_detector)
        cv2.imwrite(os.path.join(output_dir, file_name), detected)
    
def main(_argv):
    full_detector = create_detector(80,100,1,0.75)
    center_detector = create_detector(170,180,1, 0.25) 
    detect_multiple_images(FLAGS.roi, FLAGS.output, full_detector, center_detector)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass    
    
