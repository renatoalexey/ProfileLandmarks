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
    #image_path_list = ['F:\\Bases\\cfp-dataset\\Data\\Images\\291\\profile\\02.jpg']
    for i, image_path in enumerate(image_path_list):
        
        ground_truth_points_list, image = cfp.get_ground_truth_points(image_path)
            
        fa_points_list, face_detected = core.get_face_alignment_points(image)

        img_suffix = image_path[image_path.index("Images") + 7: len(image_path)].replace("/", "_")
        save_path = f"output/face_alignment/{img_suffix}"
        
        distances_list = []
        
        if face_detected == FaceType.ONE:
            save_images.fiducial_points(image, ground_truth_points_list, fa_points_list, CorrespondentFaceAlignment.CFP.points, save_path)
            distances_list.append(core.get_euclidean_results(ground_truth_points_list, fa_points_list, CorrespondentFaceAlignment.CFP.points, image))
        elif face_detected == FaceType.MULTIPLE:
            save_images.bounding_boxes(image, fa_points_list, save_path)
            for face_points in fa_points_list:
                if teste(image, face_points):
                    distances_list.append(core.get_euclidean_results(ground_truth_points_list, face_points, CorrespondentFaceAlignment.CFP.points, image))
        else:
            distances_list = [[]]

        core.writes_euclidean_distances(image_path, face_detected.value, distances_list, "output/cfp_fa_result.txt")

def teste(image, library_points_list):
    image_height, image_width = image.shape[:2]
    
    x_min = np.min(library_points_list[:, 0])
    y_min = np.min(library_points_list[:, 1])
    x_max = np.max(library_points_list[:, 0])
    y_max = np.max(library_points_list[:, 1])

    width = x_max - x_min
    height = y_max - y_min

    return width * height > 0.2 * ( image_height * image_width )

if os.path.exists('output/cfp_fa_result.txt'):
    os.remove('output/cfp_fa_result.txt')  

#if os.path.exists('output/distances.txt'):
 #   os.remove('output/distances.txt')  
run()