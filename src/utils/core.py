import math
from PIL import Image
import face_alignment
import boto3
import cv2
from .face_type import FaceType
import traceback
import numpy as np

rekognition = boto3.client("rekognition", region_name="us-east-1")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

def get_euclidean_results(ground_truth_pts, library_pts, correspondet_points, image, vertical_distance=1, horizontal_distance=1):
    all_distances = []

    height, width = image.shape[:2]
    resolution = height * width

    for i, groud_truth_point in enumerate(ground_truth_pts, start=1):
        if correspondet_points.get(i) is not None:
            try:
                fa_point = library_pts[correspondet_points.get(i)]
                distance = calc_euclidean_distance(groud_truth_point[0], groud_truth_point[1],
                        fa_point[0], fa_point[0], vertical_distance, horizontal_distance)    
                # Dividindo pela área da imagem para normalizar e multiplicando por 1000 para evitar números muito pequenos
                all_distances.append(distance * 1000 / resolution)
            except IndexError as e:
                print(f"Erro: {e}")
                print(f"Valor do i: {i} ### valor do correspondente: {correspondet_points.get(i)}")
                #traceback.print_exc()
                return []
                
    #all_distances = [distance / max_distance for distance in all_distances]
    return all_distances

def calc_euclidean_distance(x1, y1, x2, y2, vertical_distance=1, horizontal_distance=1):
    return round(math.sqrt( ( (x2 - x1) / horizontal_distance ) **2 + ( (y2 - y1) / vertical_distance ) **2), 2)

def writes_euclidean_distances(image_path, face_detected, all_distances, file_path):
    
    with open(file_path, 'a') as file:
        image = Image.open(image_path)
        width, height = image.size
        color = image.mode

        #print(image_path)
        for dists in all_distances:
            img_index = image_path.index("Images")
            msg = (
                f"name: {image_path[img_index + 7:len(image_path)]}, resolution: {width}x{height}, color: {color}, face detected: {face_detected}, "
                f"distances: {dists}, mean: { 0 if len(dists) == 0 else sum(dists) / len(dists)}\n"
            )
            file.write(msg)

def get_image_path(fiducials_file_path):
    return fiducials_file_path.replace("Fiducials", "Images").replace("txt", "jpg")
    #images_index = image_path.index("Images")
    

def get_face_alignment_points(image):
    fa_points_list = []
    try:
        fa_points_list = fa.get_landmarks(image)
        
        if fa_points_list is None:
            return [], FaceType.NONE 
        elif len(fa_points_list) > 1:
            return fa_points_list, FaceType.MULTIPLE
        return fa_points_list[0], FaceType.ONE
        
    except Exception as e:
        print("Erro:", e)
        return fa_points_list, FaceType.NONE

def get_amazon_points(img, image_path):
    with open(image_path, "rb") as img_file:
        img_bytes = img_file.read()

    # --- Chama o Rekognition ---
    response = rekognition.detect_faces(
        Image={'Bytes': img_bytes},
        Attributes=['ALL']
    )
    
    faceDetails = response["FaceDetails"]
    amazon_pts = []
    faces_detected = FaceType.ONE
    
    if len(faceDetails) == 0:
        faces_detected = FaceType.NONE
        return [[]], faces_detected
    elif len(faceDetails) > 1:
        faces_detected = FaceType.MULTIPLE

    h, w, _ = img.shape

    for i, face in enumerate(faceDetails):
        points_temp = []
        for landmark in face["Landmarks"]:
            x = float(landmark["X"] * w)
            y = float(landmark["Y"] * h)
            points_temp.append((x, y))
        amazon_pts.append(points_temp)

    return amazon_pts, faces_detected

def save_results_library(ground_truth_points_list, libray_points_list, correspondent_points_list,
                         face_detected, image, image_path):
    if face_detected == FaceType.ONE:
        distances_list = get_euclidean_results(ground_truth_points_list, libray_points_list, correspondent_points_list, image)
    writes_euclidean_distances(image_path, face_detected.value, distances_list, "output/cfp_fa_result.txt")

def valids_bounding_box(image, library_points_list):
    image_height, image_width = image.shape[:2]
    
    x_min = np.min(library_points_list[:, 0])
    y_min = np.min(library_points_list[:, 1])
    x_max = np.max(library_points_list[:, 0])
    y_max = np.max(library_points_list[:, 1])

    width = x_max - x_min
    height = y_max - y_min

    return width * height > 0.2 * ( image_height * image_width )