import math
from PIL import Image
import os
import face_alignment
import boto3
import cv2

cfp_images_path = "/home/renatoalexey/Documents/Bases/cfp-dataset/Data/Images"
#cfp_images_path = "F:\\Bases\\cfp-dataset\\Data\\Images"

def get_images_paths():
    images_paths = []
    for current_folder in os.listdir(cfp_images_path):
        for i in range(1, 5):
            images_paths.append(f"{cfp_images_path}/{current_folder}/profile/0{i}.jpg")
    return images_paths

def get_ground_truth_path(image_path):
    return image_path.replace("Images", "Fiducials").replace("jpg", "txt")

def get_ground_truth_points(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ground_truth_path = get_ground_truth_path(image_path)
    ground_truth_points_list = []

    if os.path.exists(ground_truth_path):
        with open(ground_truth_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                x, y = line.split(',')
                x = float(x)
                y = float(y)
                ground_truth_points_list.append((x, y))

    width = image.shape[1]
    right_side = verifies_img_side(ground_truth_points_list, width)

    if not right_side:
        ground_truth_points_list = [ (width - x, y) for x, y in ground_truth_points_list]
        image = cv2.flip(image, 1)
   
    return ground_truth_points_list, image

def verifies_img_side(ground_truth_points, width):
    #print(f"Largura: {width}")
    x = ground_truth_points[21][0]
    return x > width/2