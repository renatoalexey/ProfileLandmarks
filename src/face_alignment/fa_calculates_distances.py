import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.utils import cfp
from src.utils import core
from .correspondent_fa_type import CorrespondentFaceAlignment
from src.utils.face_type import FaceType

def get_ground_truth_file(base_path, name, file_name ):
    file_name = file_name.split('.')[0]
    return f"{os.path.join(base_path, name)}/profile/{file_name}.txt"

def run():
    image_path_list = cfp.get_images_paths()

    for i, image_path in enumerate(image_path_list):
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ground_truth_points_list = cfp.get_ground_truth_points(image_path)
        
        width = image.shape[1]
        right_side = cfp.verifies_img_side(ground_truth_points_list, width)
    
        if not right_side:
            ground_truth_points_list = [ (width - x, y) for x, y in ground_truth_points_list]
            image = cv2.flip(image, 1)
            
        fa_points_list, face_detected = core.get_face_alignment_points(image)
        distances_list = []
        if face_detected == FaceType.ONE:
            correspondent_points_list = CorrespondentFaceAlignment.CFP.points
            distances_list = core.compare_points(ground_truth_points_list, fa_points_list[0], correspondent_points_list)
        core.writes_euclidean_distances(image_path, face_detected.value, distances_list, "output/cfp_fa_result.txt")

if os.path.exists('output/cfp_fa_result.txt'):
    os.remove('output/cfp_fa_result.txt')  

#if os.path.exists('output/distances.txt'):
 #   os.remove('output/distances.txt')  
run()