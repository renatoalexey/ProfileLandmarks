import math
import cv2
import numpy as np
from .gets_via_gt import gets_gt
import matplotlib.pyplot as plt

def pritn_bp(distances):

    boxplot_data = list(zip(*distances.values()))

    plt.figure(figsize=(10, 6))

    plt.boxplot(boxplot_data)

    plt.xlabel("Posição no array")
    plt.ylabel("Valores")
    plt.title("Box Plot por posição")

    plt.show()

def calc_euclidean_distance(x1, y1, x2, y2, vertical_distance=1, horizontal_distance=1):
    return round(math.sqrt( ( (x2 - x1) / horizontal_distance ) **2 + ( (y2 - y1) / vertical_distance ) **2), 2)

vertical_point_a = 11
vertical_point_b = 8
horizontal_point_a = 0
horizontal_point_b = 18

def get_euclidean_results(mlkit_pts, vertical_distance=1, horizontal_distance=1):
    all_distances = {} 

    # vertical_distance = calc_euclidean_distance(ground_truth_pts[vertical_point_a][0], ground_truth_pts[vertical_point_a][1],
    #                                           ground_truth_pts[vertical_point_b][0], ground_truth_pts[vertical_point_b][1])
    # horizontal_distance = calc_euclidean_distance(ground_truth_pts[horizontal_point_a][0], ground_truth_pts[horizontal_point_a][1],
    #                                           ground_truth_pts[horizontal_point_b][0], ground_truth_pts[horizontal_point_b][1])

    ground_truth_pts = gets_gt()

    for i, img_key in enumerate(ground_truth_pts, start=0):
            try:
                img_mlkit_pts = mlkit_pts[img_key]
                img_dists = []
                for j, mlkit_point in enumerate(img_mlkit_pts, start=0):
                    gt_point = ground_truth_pts[img_key][j]
                    distance = calc_euclidean_distance(gt_point[0], gt_point[1],
                            mlkit_point[0], mlkit_point[1])    
                    # Dividindo pela área da imagem para normalizar e multiplicando por 1000 para evitar números muito pequenos
                    img_dists.append(distance)
                all_distances[img_key] = img_dists
            except IndexError as e:
                print(f"Erro: {e}")
                #traceback.print_exc()
                return []
                
    #all_distances = [distance / max_distance for distance in all_distances]
    return all_distances

def compare_copy(mlkit_points):
    ground_truth_pts = gets_gt()

    keys = ['http://localhost:8080/Images/renatoalexey@gmail.com-renato_alexey-13-5-2026_2.png','http://localhost:8080/Images/renatoalexey@gmail.com-renato_alexey-13-5-2026_2_copy.png','http://localhost:8080/Images/renatoalexey@gmail.com-renato_alexey-13-5-2026_4.png','http://localhost:8080/Images/renatoalexey@gmail.com-renato_alexey-13-5-2026_4_copy.png'] 

    for key in keys:
        img_mlkit_pts = mlkit_points[key]
        distances = []
        for i, mlkit_point in enumerate(img_mlkit_pts, start=0):
                    gt_point = ground_truth_pts[key][i]
                    distance = calc_euclidean_distance(gt_point[0], gt_point[1],
                            mlkit_point[0], mlkit_point[1])
                    distances.append(distance)
        print(f"{key} | {distances}")
        print(np.mean(distances))

mlkit_points = {'http://localhost:8080/Images/renatoalexey@gmail.com-maria_luiza-21-5-2026_1.png': [(240.0, 433.0), (175.0, 471.0), (124.0, 491.0), (72.0, 490.0), (48.0, 425.0), (85.0, 416.0), (51.0, 399.0), (42.0, 382.0), (75.0, 362.0), (27.0, 339.0), (77.0, 257.0), (93.0, 224.0), (142.0, 226.0), (191.0, 252.0), (131.0, 276.0), (155.0, 282.0), (127.0, 286.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-maria_luiza-21-5-2026_3.png': [(298.0, 405.0), (350.0, 440.0), (393.0, 455.0), (440.0, 451.0), (454.0, 392.0), (420.0, 384.0), (452.0, 369.0), (458.0, 354.0), (432.0, 332.0), (472.0, 311.0), (432.0, 231.0), (415.0, 202.0), (376.0, 208.0), (338.0, 234.0), (385.0, 250.0), (367.0, 260.0), (390.0, 263.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-gabriel_de_castro_michelassi-13-5-2026_1.png': [(266, 537),(196, 589),(139, 610),(79, 606),(57, 511),(102, 506),(60, 491),(52, 472),(89, 445),(39, 419),(89, 322),(106, 283),(154, 279),(206, 305),(145, 332),(169, 343),(141, 349)], 'http://localhost:8080/Images/renatoalexey@gmail.com-gabriel_de_castro_michelassi-13-5-2026_2.png': [(230, 414), (284, 456), (333, 478), (394, 481), (410, 416),(413, 373),(407, 390),(362, 403),(379, 349),(427, 330),(391, 260),(369, 236),(319, 235),(272, 252),(305, 280),(329, 275),(333, 282)], 'http://localhost:8080/Images/renatoalexey@gmail.com-renato_alexey-13-5-2026_2.png': [(192.0, 386.0), (130.0, 414.0), (80.0, 423.0), (32.0, 414.0), (27.0, 350.0), (69.0, 351.0), (33.0, 338.0), (28.0, 325.0), (58.0, 309.0), (18.0, 288.0), (70.0, 231.0), (86.0, 208.0), (122.0, 210.0), (161.0, 231.0), (112.0, 244.0), (129.0, 252.0), (108.0, 253.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-renato_alexey-13-5-2026_4.png': [(240.0, 336.0), (289.0, 366.0), (329.0, 379.0), (371.0, 376.0), (382.0, 322.0), (348.0, 318.0), (380.0, 307.0), (385.0, 295.0), (360.0, 277.0), (396.0, 259.0), (361.0, 202.0), (351.0, 180.0), (315.0, 181.0), (278.0, 200.0), (321.0, 213.0), (303.0, 219.0), (325.0, 222.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-danilo-11-5-2026_2.png': [(232, 475),(163, 499),(110, 503),(57, 487),(48, 418),(92, 423),(55, 398),(50, 380),(86, 368),(42, 334),(105, 260),(130, 232),(176, 249),(219, 287),(158, 293),(178, 307),(151, 302)], 'http://localhost:8080/Images/renatoalexey@gmail.com-danilo-11-5-2026_1.png': [(295.0, 438.0), (368.0, 465.0), (424.0, 470.0), (482.0, 452.0), (493.0, 378.0), (444.0, 383.0), (490.0, 357.0), (495.0, 338.0), (458.0, 319.0), (508.0, 282.0), (442.0, 206.0), (416.0, 176.0), (368.0, 193.0), (324.0, 233.0), (385.0, 240.0), (364.0, 255.0), (394.0, 251.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-sarah_-14-5-2026.png': [(156.0, 326.0), (112.0, 342.0), (78.0, 346.0), (41.0, 335.0), (37.0, 299.0), (69.0, 301.0), (43.0, 285.0), (41.0, 274.0), (67.0, 268.0), (39.0, 249.0), (84.0, 201.0), (106.0, 186.0), (138.0, 198.0), (161.0, 221.0), (121.0, 223.0), (133.0, 233.0), (113.0, 230.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-sarah_-14-5-2026_4.png': [(170.0, 265.0), (212.0, 286.0), (245.0, 293.0), (278.0, 287.0), (285.0, 247.0), (254.0, 244.0), (280.0, 232.0), (284.0, 222.0), (263.0, 211.0), (291.0, 195.0), (258.0, 146.0), (245.0, 126.0), (217.0, 130.0), (191.0, 149.0), (226.0, 160.0), (213.0, 167.0), (231.0, 167.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-renato_alexey-13-5-2026_2_copy.png': [(192.0, 386.0), (130.0, 414.0), (80.0, 423.0), (32.0, 414.0), (27.0, 350.0), (69.0, 351.0), (33.0, 338.0), (28.0, 325.0), (58.0, 309.0), (18.0, 288.0), (70.0, 231.0), (86.0, 208.0), (122.0, 210.0), (161.0, 231.0), (112.0, 244.0), (129.0, 252.0), (108.0, 253.0)], 'http://localhost:8080/Images/renatoalexey@gmail.com-renato_alexey-13-5-2026_4_copy.png': [(240.0, 336.0), (289.0, 366.0), (329.0, 379.0), (371.0, 376.0), (382.0, 322.0), (348.0, 318.0), (380.0, 307.0), (385.0, 295.0), (360.0, 277.0), (396.0, 259.0), (361.0, 202.0), (351.0, 180.0), (315.0, 181.0), (278.0, 200.0), (321.0, 213.0), (303.0, 219.0), (325.0, 222.0)]}

#pritn_bp(get_euclidean_results(mlkit_points))
compare_copy(mlkit_points)