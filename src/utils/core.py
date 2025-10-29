import math
from PIL import Image
import face_alignment
import boto3
import cv2
from .face_type import FaceType

rekognition = boto3.client("rekognition", region_name="us-east-1")
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

def compare_points(ground_truth_pts, library_pts, correspondet_points, vertical_distance=1, horizontal_distance=1):
    all_distances = []

    for i, groud_truth_point in enumerate(ground_truth_pts, start=1):
        if correspondet_points.get(i) is not None:
            try:
                fa_point = library_pts[correspondet_points.get(i)]
                distance = calc_euclidean_distance(groud_truth_point[0], groud_truth_point[1],
                        fa_point[0], fa_point[0], vertical_distance, horizontal_distance)    
                all_distances.append(distance)
            except IndexError as e:
                print(f"Erro: {e}")
                print(f"Valor do i: {i} ### valor do correspondente: {correspondet_points.get(i)}")
                
    max_distance = max(all_distances)
    all_distances = [distance / max_distance for distance in all_distances]
    return all_distances

def calc_euclidean_distance(x1, y1, x2, y2, vertical_distance=1, horizontal_distance=1):
    return round(math.sqrt( ( (x2 - x1) / horizontal_distance ) **2 + ( (y2 - y1) / vertical_distance ) **2), 2)

def writes_euclidean_distances(image_path, face_detected, all_distances, file_path):
    
    with open(file_path, 'a') as file:
        image = Image.open(image_path)
        width, height = image.size
        color = image.mode

        #print(image_path)
        img_index = image_path.index("Images")
        msg = (
            f"name: {image_path[img_index + 7:len(image_path)]}, resolution: {width}x{height}, color: {color}, face detected: {face_detected}, "
            f"distances: {all_distances}, mean: { 0 if len(all_distances) == 0 else sum(all_distances) / len(all_distances)}\n"
        )

        file.write(msg)

def get_image_path(fiducials_file_path):
    return fiducials_file_path.replace("Fiducials", "Images").replace("txt", "jpg")
    #images_index = image_path.index("Images")
    

def get_face_alignment_points(image):
    fa_points_list = []
    try:
        face_detected = FaceType.ONE
        fa_points_list = fa.get_landmarks(image)
        
        if fa_points_list is None:
            face_detected = FaceType.NONE 
        elif len(fa_points_list) > 1:
            face_detected = FaceType.MULTIPLE
        return fa_points_list, face_detected
        
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
    face_detected = FaceType.ONE

    if len(faceDetails) == 0:
        face_detected = FaceType.NONE
    elif len(faceDetails) > 1:
        face_detected = FaceType.MULTIPLE

    amazon_pts = []
    h, w, _ = img.shape

    for face in faceDetails:
        for landmark in face["Landmarks"]:
            x = float(landmark["X"] * w)
            y = float(landmark["Y"] * h)
            amazon_pts.append((x, y))

    return amazon_pts, face_detected