import os
import math
import cv2
import matplotlib.pyplot as plt
from .utils import core
from .utils import cfp
from .amazon.correspondent_amazon_type import CorrespondentAmazon
from .face_alignment.correspondent_fa_type import CorrespondentFaceAlignment
from PIL import Image

#ground_truth_file_path = '/home/renatoalexey/Documents/Bases/cfp-dataset/Data/Fiducials/009/profile/01.txt'
ground_truth_file_path = 'F:\\Bases\\cfp-dataset\\Data\\Fiducials\\015\\profile\\02.txt'
img_path = core.get_image_path(ground_truth_file_path)

def tests_face_alignment():
    #ground_truth_points_list = cfp.get_ground_truth_points(img_path)
    #image = Image.open(img_path)
    #width = image.size[0]
    #blabla = cfp.verifies_img_side(ground_truth_points_list, width)
    #print(f"Bla: {not blabla}")
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gt_pts = cfp.get_ground_truth_points(img_path)
    width = image.shape[1]
    print(f"Largura: {width}")
    
    color = image.shape[2]
    print(f"Color: {color}")

    right_side = cfp.verifies_img_side(gt_pts, width)

    print(f"Teste ###: {right_side}")
    
    if not right_side:
        gt_pts = [ (width - x, y) for x, y in gt_pts]
        image = cv2.flip(image, 1)
        
    fa_points = core.get_face_alignment_points(image)
    if fa_points is None:
        print("No faces detected")
    #elif len(fa_points) > 1:
     #   print("Mais de um rosto encontrado")
    else:
        correspondent_points = CorrespondentFaceAlignment.CFP.points
        prints_graphic(image, fa_points[0], correspondent_points, gt_pts)
   
def tests_amazon():
    amazon_points = core.get_amazon_points(img_path)
    correspondent_points = CorrespondentAmazon.CFP.points
    if amazon_points:
        prints_graphic(amazon_points, correspondent_points)

def prints_graphic(img, library_points, correspondent_points, ground_truth_points=[]):

    width, height = img.shape[:2]
    #img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)

    #ground_truth_points = cfp.get_ground_truth_points(img_path)
    #correspondent_points = utils.get_fa_correspondent_points(img, ground_truth_points)

    if ground_truth_points:        
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
        #plt.savefig('demon.png')
        plt.show()

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

