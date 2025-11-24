from src.utils import cfp
from src.utils import core
from src.utils import save_images
from .correspondent_amazon_type import CorrespondentAmazon
from src.utils.face_type import FaceType
import os
import numpy  as np

output_amazon_path = "result/cfp_amazon_result.txt"

def run():
    image_path_list = cfp.get_images_paths()
    #image_path_list = ['/home/renatoalexey/Documents/Bases/cfp-dataset/Data/Images/018/profile/01.jpg']
    #image_path_list = ['F:\\Bases\\cfp-dataset\\Data\\Images\\018\\profile\\01.jpg']
    for image_path in image_path_list:
        
        ground_truth_points_list, image = cfp.get_ground_truth_points(image_path)

        amazon_points_list, face_detected = core.get_amazon_points(image, image_path)
        
        img_suffix = image_path[image_path.index("Images") + 7: len(image_path)].replace("/", "_")
        save_path = f"output/amazon/{img_suffix}"

        amazon_points_list = np.array(amazon_points_list)

        distances_list = []
        correspondent_points = CorrespondentAmazon.CFP.points

        if face_detected == FaceType.MULTIPLE:
            save_images.bounding_boxes(image, amazon_points_list, save_path)
        for amazon_points in amazon_points_list:
            if face_detected == FaceType.ONE:
                save_images.fiducial_points(image, ground_truth_points_list, amazon_points, correspondent_points, save_path)
            elif face_detected == FaceType.MULTIPLE:
                if not core.valids_bounding_box(image, amazon_points):
                    continue
            distances_list.append(core.get_euclidean_results(ground_truth_points_list, amazon_points, CorrespondentAmazon.CFP.points, image))

        core.writes_euclidean_distances(image_path, face_detected.value, distances_list, output_amazon_path)

if os.path.exists(output_amazon_path):
    os.remove(output_amazon_path)  

run()