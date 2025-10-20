import utils.cfp as cfp
import utils.core as core
from correspondent_amazon_type import CorrespondentAmazon

def run():
    image_path_list = cfp.get_images_paths()

    for image_path in image_path_list:
        amazon_points_list = core.get_amazon_points(image_path)

        face_detected = False
        distances_list = []

        if amazon_points_list:
            face_detected = True
            ground_truth_points_list = cfp.get_ground_truth_points(image_path)
            correspondent_points_list = CorrespondentAmazon.CFP.points
            distances_list = core.compare_points(ground_truth_points_list, amazon_points_list, correspondent_points_list)
        
    core.writes_euclidean_distances(image_path, face_detected, distances_list, "cfp_amazon_result")

        
