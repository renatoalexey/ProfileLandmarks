import math
import cv2
import numpy as np
from .gets_via_gt import gets_gt
from .gets_via_gt import gets_all_gt
from src.utils import core
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from glob import glob
from statistics import fmean
import os

def library_bounding_boxes(image, bounding_box_points, library_points, output_path=None):
    fig, ax = plt.subplots()
    ax.imshow(image)

    for i, mlkit_point in enumerate(library_points):
        a, b = mlkit_point

        ax.scatter(a, b, color="blue", s=10)
        ax.annotate(
            str(i + 1),
            (a, b),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            color="blue",
            fontsize=8
        )

    x_min = bounding_box_points[0]
    y_min = bounding_box_points[1]
    x_max = bounding_box_points[2]
    y_max = bounding_box_points[3]

    rectangle = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor="r",
        facecolor="none"
    )

    ax.add_patch(rectangle)

    ax.axis("off")

    fig.savefig(output_path)
    plt.close(fig)

def print_bar(first_notes, rep_notes):
    categorias = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']

    x = np.arange(len(categorias))
    width = 0.35

    plt.bar(x - width/2, first_notes, width, label='Repetição')
    plt.bar(x + width/2, rep_notes, width, label='ML Kit')

    plt.xticks(x, categorias)
    plt.legend()
    plt.xlabel('Landmarks')
    plt.ylabel('Média de erro')
    plt.savefig('bar_copy.png')
    plt.show()

def pritn_boxplot(distances):

    boxplot_data = list(zip(*distances.values()))

    plt.figure(figsize=(10, 6))

    plt.boxplot(boxplot_data)

    plt.xlabel("Landmarks")
    plt.ylabel("Média de erro")
    #plt.title("Box Plot por posição")
    plt.savefig('via_output/box_plot.png')
    plt.show()

def calc_euclidean_distance(x1, y1, x2, y2, vertical_distance=1, horizontal_distance=1):
    return round(math.sqrt( ( (x2 - x1) / horizontal_distance ) **2 + ( (y2 - y1) / vertical_distance ) **2), 2)

vertical_point_a = 11
vertical_point_b = 8
horizontal_point_a = 0
horizontal_point_b = 18

def normalize_img(img_path):
    image = cv2.imread(img_path)
    height, width, _ = image.shape
    return height, width

def get_euclidean_results(all_mlkit_pts):
    all_distances = {} 

    all_ground_truth_pts = gets_all_gt()

    for i, img_key in enumerate(all_mlkit_pts, start=0):
        image = cv2.imread(f"./via_image/{img_key[29:].replace('_copy', '')}")
        height, width, _ = image.shape
        all_distances[img_key] = []
        for gt_points in all_ground_truth_pts[img_key]:
            img_dists = []
            for j, gt_point in enumerate(gt_points, start=0):
                try:
                    mlkit_pts = all_mlkit_pts[img_key][j]
                    distance = calc_euclidean_distance(gt_point[0], gt_point[1],
                            mlkit_pts[0], mlkit_pts[1], height, width)    
                    # Dividindo pela área da imagem para normalizar e multiplicando por 1000 para evitar números muito pequenos
                    img_dists.append(distance)
                except IndexError as e:
                    print(f"Erro: {e}")
                    #traceback.print_exc()
                    return []
            all_distances[img_key].append(img_dists)
                
    #all_distances = [distance / max_distance for distance in all_distances]
    all_means = {}
    for img_key in all_distances:
        medias = [round(sum(coluna) / len(coluna), 4) for coluna in zip(*all_distances[img_key])]
        all_means[img_key] = medias
    return all_means

def compare_copy():
    ground_truth_pts = gets_gt()
    all_distances = []
    keys = ['http://localhost:8080/Images/renatoalexey@gmail.com-renato_alexey-13-5-2026_2','http://localhost:8080/Images/renatoalexey@gmail.com-renato_alexey-13-5-2026_4']

    for i, key in enumerate(keys, start=0):
        distances = []
        height, width = normalize_img(f"./via_image/{key[29:]}.png")
        for j, first_point in enumerate(ground_truth_pts[f"{key}.png"], start=0):
            second_point = ground_truth_pts[f"{key}_copy.png"][j]
            distance = calc_euclidean_distance(first_point[0], first_point[1],
                    second_point[0], second_point[1], height, width)
            distances.append(distance)
        all_distances.append(distances)
    #print(f"{key} | {distances}")
    #print(np.mean(distances))
    print(all_distances)
    return all_distances

def compare_all_copy(all_distances_copy):
    copy_means = [round((x + y)/2, 2) for x, y in zip(all_distances_copy[0], all_distances_copy[1])]
    mlkit_distances = get_euclidean_results(all_mlkit_points)
    mlkit_means = [fmean(valores) for valores in zip(*mlkit_distances.values())]
    print_bar(copy_means, mlkit_means)

def compare_bb(all_mlkit_points):
    for image_path in glob("./via_image/*.png"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fa_points_list, face_detected = core.get_face_alignment_points(image)

        if not fa_points_list:
            return
        
        i_max = core.gets_biggest_bounding_box(fa_points_list[2])
        bounding_box_max = fa_points_list[2][i_max]
        image_name = image_path[12:]
        mlkit_points =  all_mlkit_points[f"http://localhost:8080/Images/{image_name}"][:4]
        l_accuracy = core.calcs_landmarks_inside(mlkit_points, bounding_box_max)
        library_bounding_boxes(image, bounding_box_max, mlkit_points, f"via_image/output/{image_name}")
        l_accuracy = round(l_accuracy, 2)
        print(l_accuracy)

all_mlkit_points = {'http://localhost:8080/Images/renatoalexey@gmail.com-maria_luiza-21-5-2026_1.png': [(240.0, 433.0), (175.0, 471.0), (124.0, 491.0), (72.0, 490.0), (48.0, 425.0), (85.0, 416.0), (51.0, 399.0), (42.0, 382.0), (75.0, 362.0), (27.0, 339.0), (77.0, 257.0), (93.0, 224.0), (142.0, 226.0), (191.0, 252.0), (131.0, 276.0), (155.0, 282.0), (127.0, 286.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-maria_luiza-21-5-2026_3.png': [(298.0, 405.0), (350.0, 440.0), (393.0, 455.0), (440.0, 451.0), (454.0, 392.0), (420.0, 384.0), (452.0, 369.0), (458.0, 354.0), (432.0, 332.0), (472.0, 311.0), (432.0, 231.0), (415.0, 202.0), (376.0, 208.0), (338.0, 234.0), (385.0, 250.0), (367.0, 260.0), (390.0, 263.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-gabriel_de_castro_michelassi-13-5-2026_1.png': [(266, 537),(196, 589),(139, 610),(79, 606),(57, 511),(102, 506),(60, 491),(52, 472),(89, 445),(39, 419),(89, 322),(106, 283),(154, 279),(206, 305),(145, 332),(169, 343),(141, 349)], 'http://localhost:8080/Images/renatoalexey@gmail.com-gabriel_de_castro_michelassi-13-5-2026_2.png': [(230, 414), (284, 456), (333, 478), (394, 481), (410, 416),(413, 373),(407, 390),(362, 403),(379, 349),(427, 330),(391, 260),(369, 236),(319, 235),(272, 252),(305, 280),(329, 275),(333, 282)], 'http://localhost:8080/Images/renatoalexey@gmail.com-renato_alexey-13-5-2026_2.png': [(192.0, 386.0), (130.0, 414.0), (80.0, 423.0), (32.0, 414.0), (27.0, 350.0), (69.0, 351.0), (33.0, 338.0), (28.0, 325.0), (58.0, 309.0), (18.0, 288.0), (70.0, 231.0), (86.0, 208.0), (122.0, 210.0), (161.0, 231.0), (112.0, 244.0), (129.0, 252.0), (108.0, 253.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-renato_alexey-13-5-2026_4.png': [(240.0, 336.0), (289.0, 366.0), (329.0, 379.0), (371.0, 376.0), (382.0, 322.0), (348.0, 318.0), (380.0, 307.0), (385.0, 295.0), (360.0, 277.0), (396.0, 259.0), (361.0, 202.0), (351.0, 180.0), (315.0, 181.0), (278.0, 200.0), (321.0, 213.0), (303.0, 219.0), (325.0, 222.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-danilo-11-5-2026_2.png': [(232, 475),(163, 499),(110, 503),(57, 487),(48, 418),(92, 423),(55, 398),(50, 380),(86, 368),(42, 334),(105, 260),(130, 232),(176, 249),(219, 287),(158, 293),(178, 307),(151, 302)], 'http://localhost:8080/Images/renatoalexey@gmail.com-danilo-11-5-2026_1.png': [(295.0, 438.0), (368.0, 465.0), (424.0, 470.0), (482.0, 452.0), (493.0, 378.0), (444.0, 383.0), (490.0, 357.0), (495.0, 338.0), (458.0, 319.0), (508.0, 282.0), (442.0, 206.0), (416.0, 176.0), (368.0, 193.0), (324.0, 233.0), (385.0, 240.0), (364.0, 255.0), (394.0, 251.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-sarah_-14-5-2026.png': [(156.0, 326.0), (112.0, 342.0), (78.0, 346.0), (41.0, 335.0), (37.0, 299.0), (69.0, 301.0), (43.0, 285.0), (41.0, 274.0), (67.0, 268.0), (39.0, 249.0), (84.0, 201.0), (106.0, 186.0), (138.0, 198.0), (161.0, 221.0), (121.0, 223.0), (133.0, 233.0), (113.0, 230.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-sarah_-14-5-2026_4.png': [(170.0, 265.0), (212.0, 286.0), (245.0, 293.0), (278.0, 287.0), (285.0, 247.0), (254.0, 244.0), (280.0, 232.0), (284.0, 222.0), (263.0, 211.0), (291.0, 195.0), (258.0, 146.0), (245.0, 126.0), (217.0, 130.0), (191.0, 149.0), (226.0, 160.0), (213.0, 167.0), (231.0, 167.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-renato_alexey-13-5-2026_2_copy.png': [(192.0, 386.0), (130.0, 414.0), (80.0, 423.0), (32.0, 414.0), (27.0, 350.0), (69.0, 351.0), (33.0, 338.0), (28.0, 325.0), (58.0, 309.0), (18.0, 288.0), (70.0, 231.0), (86.0, 208.0), (122.0, 210.0), (161.0, 231.0), (112.0, 244.0), (129.0, 252.0), (108.0, 253.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-renato_alexey-13-5-2026_4_copy.png': [(240.0, 336.0), (289.0, 366.0), (329.0, 379.0), (371.0, 376.0), (382.0, 322.0), (348.0, 318.0), (380.0, 307.0), (385.0, 295.0), (360.0, 277.0), (396.0, 259.0), (361.0, 202.0), (351.0, 180.0), (315.0, 181.0), (278.0, 200.0), (321.0, 213.0), (303.0, 219.0), (325.0, 222.0)]}

#pritn_boxplot(get_euclidean_results2(all_mlkit_points))
compare_all_copy(compare_copy())
#compare_bb(all_mlkit_points)
#compare_copy(all_mlkit_points)