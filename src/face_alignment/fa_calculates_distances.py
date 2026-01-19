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
    #image_path_list = ['F:\\Bases\\cfp-dataset\\Data\\Images\\001\\profile\\01.jpg']
    for i, image_path in enumerate(image_path_list):
        
        ground_truth_points_list, image = cfp.get_ground_truth_points(image_path)
            
        fa_points_list, face_detected = core.get_face_alignment_points(image)

        if not fa_points_list:
            continue
        img_suffix = image_path[image_path.index("Images") + 7: len(image_path)].replace("/", "_")
        save_path = f"output/face_alignment/bounding_box/{img_suffix}"
        
        i_max = core.gets_biggest_bounding_box(fa_points_list[2])
        bounding_box_max = fa_points_list[2][i_max]
        l_accuracy = core.calcs_landmarks_inside(ground_truth_points_list, bounding_box_max)
        l_accuracy = round(l_accuracy, 2)
        #core.writes_landmarks_bb("face_alignment", img_suffix, fa_points_list[0][i_max], bounding_box_max, l_accuracy)
        #save_images.library_bounding_boxes(image, bounding_box_max, save_path)
        fa_points = fa_points_list[0][i_max]
        saves_distances(face_detected, image, fa_points, save_path, ground_truth_points_list, image_path)
        
def saves_distances(face_detected, image, fa_points_list, save_path, ground_truth_points_list, image_path):
    distances_list = []
    correspondent_points = CorrespondentFaceAlignment.CFP.points
    
    #if face_detected == FaceType.MULTIPLE:
       # save_images.bounding_boxes(image, fa_points_list, save_path)
    
    #for fa_points in fa_points_list:
        #if face_detected == FaceType.ONE:
            #save_images.fiducial_points(image, ground_truth_points_list, fa_points, correspondent_points, save_path)
    if face_detected == FaceType.MULTIPLE:
        if not core.valids_bounding_box(image, fa_points_list):
            return
    distances_list.append(core.get_euclidean_results(ground_truth_points_list, fa_points_list, correspondent_points, image))

    core.writes_euclidean_distances(image_path, face_detected.value, distances_list, "result/cfp_fa_result.txt")

#if os.path.exists('result/cfp_fa_result.txt'):
    #os.remove('result/cfp_fa_result.txt')  

#if os.path.exists('output/distances.txt'):
 #   os.remove('output/distances.txt')  
run()