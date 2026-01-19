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
from .mlkit.correspondent_mlkit_type import CorrespondentMLKit
from PIL import Image

#ground_truth_file_path = '/home/renatoalexey/Documents/Bases/cfp-dataset/Data/Fiducials/001/profile/01.txt'
ground_truth_file_path = 'F:\\Bases\\cfp-dataset\\Data\\Fiducials\\001\\profile\\01.txt'
img_path = core.get_image_path(ground_truth_file_path)

def tests_face_alignment():
    img_path = 'F:\\Bases\\cfp-dataset\\renatoalexey.png'
    #ground_truth_points_list, image = cfp.get_ground_truth_points(img_path)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      
    fa_points_list, face_detected = core.get_face_alignment_points(image)
    #fa_points_list =[[124.  , 47.],[126.  , 65.],    [128.  , 83.],    [130.  , 97.],    [136. , 115.],    [152. , 129.],    [168. , 139.],    [186. , 147.],    [196. , 153.],    [202. , 149.],    [208. , 143.],    [208. , 135.],    [210. , 119.],    [212.  , 97.],[204.  , 89.],[202.  , 73.],[198.  , 53.],[178.  , 29.],[186.  , 22.],[194.  , 20.],[198.  , 22.],[200.  , 24.],[202.  , 26.],[200.  , 24.],[198.  , 26.],[194.  , 29.],[184.  , 39.],[202.  , 47.],[208.  , 57.],[216.  , 65.],[218.  , 73.],[202.  , 83.],[206.  , 83.],[208.  , 85.],[208.  , 85.],[204.  , 85.],[186.  , 41.],[192.  , 39.],[194.  , 39.],[192.  , 45.],[192.  , 45.],[190.  , 45.],[192.  , 49.],[196.  , 45.],[198.  , 45.],[190.  , 49.],[196.  , 51.],[196.  , 51.],[200. , 109.],[206. , 101.],[210.  , 95.],[212.  , 97.],[212.  , 95.],[210. , 101.],[204. , 109.],[210. , 113.],[212. , 115.],[212. , 115.],[210. , 115.],[208. , 113.],[200. , 109.],[208. , 103.],[210. , 103.],[210. , 103.],[204. , 109.],[210. , 107.],[210. , 107.],[208. , 107.]] 
    correspondent_points = CorrespondentFaceAlignment.CFP.points
    # if face_detected == FaceType.ONE:
    #distances_list = core.get_euclidean_results(ground_truth_points_list, fa_points_list, correspondent_points, image)
    # elif face_detected == FaceType.MULTIPLE:
    #     print('adsf')
  
    color = image.shape[2]
    #print(f"Fa: {fa_points_list}")
        
    prints_library_graphic(image, fa_points_list[0])
    #prints_graphic(image, fa_points_list[0], correspondent_points, ground_truth_points_list)
    #all_points(image, ground_truth_points_list, fa_points_list[0])
def tests_amazon():
    ground_truth_points_list, image = cfp.get_ground_truth_points(img_path)
    amazon_points, face_detected = core.get_amazon_points(image, img_path)
    correspondent_points = CorrespondentAmazon.CFP.points
    #print(amazon_points)
    amazon_points = np.array(amazon_points[0])
    #amazon_points = []
    #if amazon_points:
    all_points(image, ground_truth_points_list, amazon_points)

def prints_graphic(img, library_points_list, correspondent_points, ground_truth_points=[]):
    for library_points in library_points_list:

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
            
            plt.imshow(img)
            plt.axis("off")
            plt.show()
            
def prints_library_graphic(img, library_points_list):
    fig, ax = plt.subplots()
    ax.imshow(img)   
    plt.imshow(img)
    
    for library_points in library_points_list:
        for library_point in library_points:
            print(f"Library point: {library_point}")    
            a, b = library_point
            plt.scatter(a, b, color="blue", s=10)
            #plt.annotate(str(correspondent_points.get(i) + 1), (a, b), textcoords="offset points", xytext=(0,5), ha='center', color="blue", fontsize=8)
    
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def all_points(image, ground_truth_points, library_points):
    correspondent_points = CorrespondentAmazon.CFP.points
    keys = list(correspondent_points.keys())
    values = list(correspondent_points.values())
    plt.imshow(image)
    for i, ground_truth_point in enumerate(ground_truth_points, start=0):
        if i not in keys:
            continue
        if i != 11 and i != 14 and i != 17 and i != 25:
            continue
        x, y = ground_truth_point
        plt.scatter(x, y, color="red", s=10)
        plt.annotate(str(i + 1), (x, y), textcoords="offset points", xytext=(0,5), ha='center', color="red", fontsize=12)
    for i, library_point in enumerate(library_points, start=0):     
        if i not in values:
            continue
        if i != 10 and i != 9 and i != 18 and i != 3:
            continue
        a, b = library_point
        plt.scatter(a, b, color="blue", s=10)
        plt.annotate(str(i + 1), (a, b), textcoords="offset points", xytext=(0,5), ha='center', color="blue", fontsize=12)
        
        #plt.imshow(img)
    plt.axis("off")
    plt.show()
    #plt.savefig("mlkit_correspondent.png")
    
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

