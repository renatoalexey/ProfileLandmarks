import numpy as np
from skimage import io
from scipy.io import loadmat
from pathlib import Path
from graphic import graphic_bar
import face_alignment
import cv2
import json
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
from enums.tecniques import Techs
from enums.tecniques_resize import TechsResize
from enums.bright_type import Brights
from enums.combine_type import MedianBright
from enums.resize_type import Sizes
from PIL import Image, ImageEnhance

results_path = 'output/results.txt'

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

height_factor = 0
width_factor = 0

points_file = 'output/points_not_found.txt'

def calcEuclideanDistance(x1, y1, x2, y2):
    return round(math.sqrt((x2 - x1*width_factor)**2 + (y2 - y1*height_factor)**2), 2)

all_distances = {}
all_points_distances = {} 

def initialize_distances(pipeline):
    global all_distances, all_points_distances
    all_distances = {}
    all_points_distances = {} 
    for tech in pipeline:
        all_distances[tech] = []
        all_points_distances[tech] = []
        
def readData(key, pipeline):
    folder_image_path = Path('images')
    folder_landmarks_path = 'landmarks/'

    initialize_distances(pipeline)

    cont = 0
    for file_image_path in folder_image_path.iterdir():
        #if cont == 50:
           # break
        if(cont == 100 or cont == 500 or cont == 1000):
            print(f"Imagem n {cont}")
        cont +=1
        if file_image_path.is_file():
            file_name = os.path.splitext(os.path.basename(file_image_path))[0]
            file_landmarks_path = f'{folder_landmarks_path}{file_name}_pts.mat'

            all_distances, all_points_distances = calcPointsDiffs(file_image_path, file_landmarks_path, pipeline)

    #printGraph(all_points_distances)
    #print(all_points_distances)
    print(f"Cont: {cont}")

    with open('output/results.txt', 'a') as file:
        file.write(json.dumps({str(k): v for k, v in all_distances.items()}))
        file.write(json.dumps({str(k): v for k, v in all_points_distances.items()}))
        #file.write(json.dumps(all_points_distances))

    graphic_bar.printGraphics(f"output/{key}", all_distances, all_points_distances)

def calcPointsDiffs(file_image_path, file_landmarks_path, pipeline):
    data = loadmat(file_landmarks_path)
    data_points = data['pts_2d']
    img = cv2.imread(file_image_path)
    gray_image = Techs.GRAY.getTech(img)
    #print_landmarks(img, data_points)

    return getTechsResults(file_image_path, data_points, img, gray_image, pipeline)

def getTechsResults(file_image_path, data_points, normal_image, gray_image, pipeline):

    for tech in pipeline:
        #entry_image = gray_image
        entry_image = normal_image
        if tech == Techs.NORMAL or tech == Techs.GRAY:
            entry_image = normal_image
        
        format_image = tech.getTech(entry_image)
        
        prediction_points = fa.get_landmarks(format_image)
        
        global height_factor
        height_factor = normal_image.shape[0] / format_image.shape[0]
        global width_factor 
        width_factor = normal_image.shape[1] / format_image.shape[1]
        
        if prediction_points is not None:
            print_landmarks(format_image, data_points, prediction_points[0], tech.f_name)
            euclidean_distances, euclidean_mean = getEuclideanMetrics(file_image_path, data_points, prediction_points)
            if euclidean_distances is not None:
                all_points_distances[tech].append(euclidean_distances)
                all_distances[tech].append(euclidean_mean)
            else:
                all_points_distances[tech].append(-1)
                all_distances[tech].append(-1)
                writesPointsNotFound(file_image_path, tech, "euclidean")
        else:
            all_points_distances[tech].append(-1)
            all_distances[tech].append(-1)
            writesPointsNotFound(file_image_path, tech, "")
                
    return all_distances, all_points_distances

def writesPointsNotFound(file_image_path, tech, suffix):
    with open(points_file, 'a') as file:
        #formatted_tech = tech.f_name.replace("\n", "")
        file.write(f"tech: {tech.name} image: {file_image_path} - {suffix} \n")

def getEuclideanMetrics(file_image_path, data_points, prediction_points):
    try: 
        for i in range(0, len(prediction_points)):
            euclidean_distances = getDistances(data_points, prediction_points, i) 
            euclidean_mean = np.mean(euclidean_distances)

            if euclidean_mean <= 50:
                return euclidean_distances, euclidean_mean

            if euclidean_mean > 50 and i == len(prediction_points):
                print(f"{file_image_path} - {euclidean_mean}")
    except Exception as e:  # Handle any other exception
        print(f"An error occurred: {e}, Prediction points: {prediction_points}")
    return None, None

def print_landmarks(img, points, points_face, title):
    
    #for i, point in enumerate(points):
     #  x = round(point[0] / width_factor)
      # y = round(point[1] / height_factor)
      # cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

    for i, point in enumerate(points_face, start=1):
        if i != 60 and i != 38 and i != 32:
            continue
        x = round(point[0])
        y = round(point[1])
        cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

        # Coloca o número ao lado
        cv2.putText(
            img,
            str(i),                 # texto (índice)
            (x + 5, y - 5),         # posição (um pouco deslocado do ponto)
            cv2.FONT_HERSHEY_SIMPLEX, # fonte
            0.2,                    # tamanho da fonte
            (255, 0, 0),            # cor (vermelho em BGR)
            1,                      # espessura
            cv2.LINE_AA             # suavização
        )

    cv2.imshow("", img)
    cv2.waitKey(0)

def getDistances(data_points, prediction_points, l):
    euclidean_distances = []
    for i, pred_point in enumerate(prediction_points[l]):
        euclidean_distances.append( calcEuclideanDistance(pred_point[0], 
                    pred_point[1], data_points[i][0], data_points[i][1]))
            
    return euclidean_distances

def sum_points_diffs(all_points_distances, euclidean_distances, type):
    if not all_points_distances[type]:
        all_points_distances[type] = euclidean_distances
    else:
        temp = []
        for a, b in zip(all_points_distances[type], euclidean_distances):
            temp.append(a + b)
        all_points_distances[type] = temp
    return all_points_distances

def sample(key, pipeline):

    file_name = "02.jpg"
    points = "landmarks/IBUG_image_003_1_6_pts.mat"

    initialize_distances(pipeline)
    all_distances, all_points = calcPointsDiffs(file_name, points, pipeline)
    #graphic_bar.printGraphics(f"output/{key}", all_distances, all_points)

pipeline1 = [Techs.NORMAL, Techs.GRAY, Techs.BRIGHT_MINUS, Techs.BRIGHT_PLUS, Techs.MEAN, Techs.MEDIAN, Techs.HIST, Techs.BORDER]
pipeline2 = [Techs.NORMAL, Brights.BRIGHT_1, Brights.BRIGHT_2, Brights.BRIGHT_3, Brights.BRIGHT_4, Brights.BRIGHT_5]
pipeline3 = [Techs.NORMAL, Sizes.SIZE_900, Sizes.SIZE_700, Sizes.SIZE_300, Sizes.SIZE_150]  
pipeline4 = [Techs.NORMAL, MedianBright.MEDIAN_BRIGHT, MedianBright.S_MEDIAN_BRIRHT, MedianBright.BORDER_BRIGHT, MedianBright.S_BORDER_BRIGHT]
pipeline5 = [Techs.NORMAL, Brights.BRIGHT_1, Techs.MEDIAN, Techs.HIST, Techs.BORDER,
            MedianBright.S_MEDIAN_BRIRHT, Sizes.SIZE_300]
pipeline6 = [TechsResize.NORMAL, TechsResize.BRIGHT_MINUS, TechsResize.BRIGHT_PLUS, 
             TechsResize.MEAN, TechsResize.MEDIAN, TechsResize.HIST, TechsResize.BORDER]

#pipelines = {"pip5": pipeline5}
#pipelines = {"pip7": [Sizes.SIZE_450]}
pipelines = {"pip1": [Techs.NORMAL]}

if os.path.exists(points_file):
    os.remove(points_file)

if os.path.exists('output/sums.txt'):
    os.remove('output/sums.txt')  
    
if os.path.exists(results_path):
    os.remove(results_path)

for key, pipeline in pipelines.items():
    with open(points_file, 'a') as file:
        file.write(f'pipeline: {key} \n')
    sample(key, pipeline)
    #readData(key, pipeline)