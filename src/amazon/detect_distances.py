import cv2
from src.utils import cfp
from src.utils import core
from .correspondent_amazon_type import CorrespondentAmazon
from src.utils.face_type import FaceType

def run():
    image_path_list = cfp.get_images_paths()

    for image_path in image_path_list:
        
        ground_truth_points_list, image = cfp.get_ground_truth_points(image_path)

        amazon_points_list, face_detected = core.get_amazon_points(image, image_path)

        distances_list = []
        if face_detected == FaceType.ONE:
            correspondent_points_list = CorrespondentAmazon.CFP.points
            distances_list = core.compare_points(ground_truth_points_list, amazon_points_list, correspondent_points_list)
        core.writes_euclidean_distances(image_path, face_detected.value, distances_list, "output/cfp_amazon_result.txt")

        
