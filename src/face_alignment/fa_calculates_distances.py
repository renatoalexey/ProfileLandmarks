import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.utils import cfp
from src.utils import core
from src.utils import save_images 
from .correspondent_fa_type import CorrespondentFaceAlignment
from src.utils.face_type import FaceType

def get_ground_truth_file(base_path, name, file_name ):
    file_name = file_name.split('.')[0]
    return f"{os.path.join(base_path, name)}/profile/{file_name}.txt"

def run():
    image_path_list = cfp.get_images_paths()

    for i, image_path in enumerate(image_path_list):
        
        ground_truth_points_list, image = cfp.get_ground_truth_points(image_path)
            
        fa_points_list, face_detected = core.get_face_alignment_points(image)

        img_suffix = image_path[image_path.index("Images") + 7: len(image_path)].replace("/", "_")
        save_path = f"output/face_alignment/{img_suffix}"

        if face_detected == FaceType.ONE:
            save_images.fiducial_points(image, ground_truth_points_list, fa_points_list, CorrespondentFaceAlignment.CFP.points, save_path)
            distances_list = core.get_euclidean_results(ground_truth_points_list, fa_points_list, CorrespondentFaceAlignment.CFP.points, image)
        elif face_detected == FaceType.MULTIPLE:
            save_images.bounding_boxes(image, fa_points_list, save_path)
        core.writes_euclidean_distances(image_path, face_detected.value, distances_list, "output/cfp_fa_result.txt")

if os.path.exists('output/cfp_fa_result.txt'):
    os.remove('output/cfp_fa_result.txt')  

#if os.path.exists('output/distances.txt'):
 #   os.remove('output/distances.txt')  
run()