import os
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from .utils import core
from .utils import cfp
from .utils.face_type import FaceType
from .amazon.correspondent_amazon_type import CorrespondentAmazon
from .face_alignment.correspondent_fa_type import CorrespondentFaceAlignment
from PIL import Image

#ground_truth_file_path = '/home/renatoalexey/Documents/Bases/cfp-dataset/Data/Fiducials/446/profile/04.txt'
ground_truth_file_path = 'F:\\Bases\\cfp-dataset\\Data\\Fiducials\\291\\profile\\02.txt'
img_path = core.get_image_path(ground_truth_file_path)

def tests_face_alignment():
    ground_truth_points_list, image = cfp.get_ground_truth_points(img_path)
            
    fa_points_list, face_detected = core.get_face_alignment_points(image)


    correspondent_points = CorrespondentFaceAlignment.CFP.points
    if face_detected == FaceType.ONE:
        distances_list = core.get_euclidean_results(ground_truth_points_list, fa_points_list[0], correspondent_points, image)
    elif face_detected == FaceType.MULTIPLE:
        print('adsf')
  
    color = image.shape[2]
    print(f"Fa: {fa_points_list}  {face_detected}")
        
    #prints_graphic(image, fa_points_list[0], correspondent_points, ground_truth_points_list)
   
def tests_amazon():
    amazon_points = core.get_amazon_points(img_path)
    correspondent_points = CorrespondentAmazon.CFP.points
    if amazon_points:
        prints_graphic(amazon_points, correspondent_points)

def prints_graphic(img, library_points, correspondent_points, ground_truth_points=[]):

    #width, height = img.shape[:2]
    #img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)

    #ground_truth_points = cfp.get_ground_truth_points(img_path)
    #correspondent_points = utils.get_fa_correspondent_points(img, ground_truth_points)
    x_min = np.min(library_points[:, 0])
    y_min = np.min(library_points[:, 1])
    x_max = np.max(library_points[:, 0])
    y_max = np.max(library_points[:, 1])

    if ground_truth_points:     
        fig, ax = plt.subplots()
        ax.imshow(img)   
        plt.imshow(img)
        for i, ground_truth_point in enumerate(ground_truth_points, start=0):
            if correspondent_points.get(i) is not None:
                library_point = library_points[correspondent_points.get(i)]
                x, y = ground_truth_point
                plt.scatter(x, y, color="red", s=10)
                plt.annotate(str(i + 1), (x, y), textcoords="offset points", xytext=(0,5), ha='center', color="red", fontsize=8)
                
                a, b = library_point
                plt.scatter(a, b, color="blue", s=10)
                plt.annotate(str(correspondent_points.get(i) + 1), (a, b), textcoords="offset points", xytext=(0,5), ha='center', color="blue", fontsize=8)
                
        # Cria retângulo
        width = x_max - x_min
        height = y_max - y_min
        
        rect = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )

        ax.add_patch(rect)
        
        plt.imshow(img)
        plt.axis("off")
        #plt.show()

def print_points(fiducial_points, plt, color='red'):
    for i, fiducial_point in enumerate(fiducial_points, start=1):
        x, y = fiducial_point
        x = float(x)
        y = float(y)
        #x = round(x * x_factor, 2)
        #y = round(y * y_factor, 2)
        plt.scatter(x, y, color=color, s=10)
        plt.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,5), ha='center', color="red", fontsize=8)
    return plt 


tests_face_alignment()
#tests_amazon()

