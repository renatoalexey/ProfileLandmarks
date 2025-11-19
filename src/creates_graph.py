import re
import numpy as np
import matplotlib.pyplot as plt
import os
from src.face_alignment.correspondent_fa_type import CorrespondentFaceAlignment
from src.mlkit.correspondent_mlkit_type import CorrespondentMLKit
from src.amazon.correspondent_amazon_type import CorrespondentAmazon

file_library_name = {"cfp_fa_result.txt": "Face Alignment", "cfp_mlkit_result.txt": "ML Kit", "cfp_amazon_result.txt": "Amazon Rekognition"}

def print_graph(means, graph_name, positions=None, x_label=None):
    plt.figure(figsize=(12, 7))
    plt.boxplot(means, positions, showmeans=True, meanline=True, tick_labels=positions, whis=2.5)
    plt.xlabel(x_label)
    plt.ylabel('Média das distâncias normalizadas entre os pontos (groud truth e biblioteca)')
    plt.savefig(f'{graph_name}.png')

def get_image_name(line):
    match = re.search(r"name:[^,]*", line)
    return match.group().split(":")[1].strip()

def get_face_detected(line):
    match = re.search(r"face detected:[^,]*", line)
    return match.group().split(":")[1].strip()

def get_resolution_area(line): 
    match = re.search(r"resolution:[^,]*", line)
    #return match.group()
    resolution_sum = match.group().split(":")[1].split("x")
    return int(resolution_sum[0].strip()) * int(resolution_sum[1].strip())
    
def get_distances(line):
    match = re.search(r"distances:\s*\[(.*?)\]", line)
    if match and match != []:
        values = match.group(1).split(",")   # separa por vírgula
        #print(values)
        if values[0] == '':
            return []
        distances = [float(v.strip()) for v in values]  # converte pra float
        return distances
        #print(distances)

def get_result_files():

    output_path = 'result'
    files_names = os.listdir(output_path)

    return list(map(lambda f_name: f"{output_path}/{f_name}", files_names))

def get_file_lines(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def all_distances_boxplot():
    result_path = 'result'
    distances_tools_list = []
    for result_file in os.listdir(result_path):
        if os.path.isdir(os.path.join(result_path, result_file)):
            continue
        distance_means = []
        result_file_full_path = os.path.join(result_path, result_file)
        with open(result_file_full_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line == '':
                    continue
                distances = get_distances(line)
                if not distances:
                    continue
                dist_mean = np.mean(distances)
                distance_means.append(dist_mean)
                
                #monta estrutura de distancia por resolucao
                soma = get_resolution_area(line)
                soma_key = int( soma/100 + 1) * 100
            distances_tools_list.append(distance_means)
            
    print_graph(distances_tools_list, "distances", ["Face Alignment", "ML Kit", "Amazon Rekognition"], "Bibliotecas")

def distances_per_point(file_path, keys):
    lines = get_file_lines(file_path)
    #keys = [4, 7, 9, 10, 11, 12, 13, 16, 17, 21, 24, 25, 26]
    #keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 25, 28, 26, 29]
    point_distances_list = []
    for line in lines:
        if line == '':
            continue
        distances = get_distances(line)
        if not distances:
            continue
        
        for i, distance in enumerate(distances):
            if len(point_distances_list) == i:
                point_distances_list.append([])
            point_distances_list[i].append(distance)

    label = get_library_from_path(file_path)
    print_graph(point_distances_list, f"points_distances_{label}", keys, "Points")

def distances_per_resolution_area(file_path):
    resolution_distances = {}
    lines = get_file_lines(file_path)

    resolution_area_list = []
    distances_area_list = []
    for line in lines:
        if line == '':
            continue
        distances = get_distances(line)
        if not distances:
            continue

        dist_mean = np.mean(distances)
    
        #monta estrutura de distancia por resolucao
        resolution_area = get_resolution_area(line)
        
        distances_area_list.append((dist_mean, resolution_area))

    sorted_distances_area_list = sorted(distances_area_list, key=lambda x: x[1])
    distances_temp = []
    values = []
    labels = []
    for i, sorted_dist in enumerate(sorted_distances_area_list, start=1):
        distances_temp.append(sorted_dist[0])
        if i % 200 == 0:
            labels.append(sorted_dist[1])
            values.append(distances_temp)
            distances_temp = []
        
    #print(f"Resolution min: {min(resolution_area_list)}")
    #print(f"Resolution max: {max(resolution_area_list)}")
    #print(f"Resolution distances: {resolution_distances}")
    print(list(values))
    print(labels)
    label = get_library_from_path(file_path)
    print_graph(values, f"resolutions_distances_{label}", labels, "Resolutions")

def get_rel_interval(file_path):
    resolution_area_list = []
  
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line == '':
                continue
            #monta estrutura de distancia por resolucao
            resolution_area = get_resolution_area(line)
            area_key = int( resolution_area/40000 + 1) * 40000
            
            if area_key not in resolution_area_list:
                resolution_area_list.append(area_key)
    return sorted(resolution_area_list)
    #print(len(resolution_area_list))

def teste():
    file_path = "result/cfp_amazon_result.txt"
    lines = get_file_lines(file_path)
    amazon_names = []
    for line in lines:
        actual_img = get_image_name(line)
        resolution_area = get_resolution_area(line)
        amazon_names.append((actual_img, resolution_area))

    file_path = "result/cfp_mlkit_result.txt"
    lines = get_file_lines(file_path)
    fa_names = []
    for line in lines:
        actual_img = get_image_name(line)
        fa_names.append(actual_img)

    for az in amazon_names:
        if az[0] not in fa_names:
            print(f"name: {az[0]}, {az[1]}, color: RGB, face detected: Multiple, distances: [], mean: 0")
    
def teste_2(): 
    all_files = ["result/cfp_mlkit_result.txt"]
    rels_interval_list = get_rel_interval(all_files[0])
    x_keys = []
    for file_path in all_files:
        detected_area_list = []
        lines = get_file_lines(file_path)

        resolution_distance_list = {}
        teste = {}
        before_img = ''
        for line in lines:
            actual_img = get_image_name(line)
            if line == '' or before_img == actual_img:
                continue
            before_img = actual_img
            #monta estrutura de distancia por resolucao
            resolution_area = get_resolution_area(line)
            face_detected = get_face_detected(line)

            detected_area_list.append((face_detected, resolution_area))
            
        sorted_distances_area_list = sorted(detected_area_list, key=lambda x: x[1])
        detected_temp = []
        values = []
        labels = []
        for i, sorted_dist in enumerate(sorted_distances_area_list, start=1):
            detected_temp.append(sorted_dist[0])
            if i % 200 == 0 :
                labels.append(str(sorted_dist[1]))
                values.append((detected_temp.count('True') + detected_temp.count('Multiple'))*100/len(detected_temp))
                detected_temp = []
            #if i == len(sorted_distances_area_list):
             #   labels.append(f"> {labels[len(labels) - 1]}")
            
        img_name = get_image_name(line)
        #plt.plot(list(map(lambda x: str(int(x/10000)), x_keys)), arrei, label=file_path, marker='o')
        label = get_library_from_path(file_path)
        plt.plot(labels, values, label=label, marker='o')
    
    #plt.xticks(range(len(blabla)), blabla)
    print(labels)
    plt.xticks(labels)
    #plt.xticks(blabla)
    plt.xlabel("Eixo X")
    plt.ylabel("Percentual de acerto")
    plt.legend()
    #plt.grid(True)
    # Mostra o gráfico
    plt.show()
    #plt.savefig('accuracy_face_3.png')
    
def accuracy_face_per_resolution():

    all_files = get_result_files()
    #all_files = ["result/cfp_fa_result.txt"]
    rels_interval_list = get_rel_interval(all_files[0])
    x_keys = []
    for file_path in all_files:
        detected_area_list = []
        lines = get_file_lines(file_path)

        resolution_distance_list = {}
        teste = {}
        before_img = ''
        for line in lines:
            actual_img = get_image_name(line)
            if line == '' or before_img == actual_img:
                continue
            before_img = actual_img
            #monta estrutura de distancia por resolucao
            resolution_area = get_resolution_area(line)
            face_detected = get_face_detected(line)

            detected_area_list.append((face_detected, resolution_area))
            
        sorted_distances_area_list = sorted(detected_area_list, key=lambda x: x[1])
        detected_temp = []
        values = []
        labels = []
        for i, sorted_dist in enumerate(sorted_distances_area_list, start=1):
            detected_temp.append(sorted_dist[0])
            if i % 200 == 0:
                labels.append(str(round(sorted_dist[1]/1000, 1)))
                values.append((detected_temp.count('True') + detected_temp.count('Multiple'))*100/len(detected_temp))
                detected_temp = []
            
        img_name = get_image_name(line)
        #plt.plot(list(map(lambda x: str(int(x/10000)), x_keys)), arrei, label=file_path, marker='o')
        label = get_library_from_path(file_path)
        plt.plot(labels, values, label=label, marker='o')

    #plt.xticks(range(len(blabla)), blabla)
    print(labels)
    plt.xticks(labels)
    plt.xlabel("Eixo X")
    plt.ylabel("Percentual de acerto")
    plt.legend()
    # Mostra o gráfico
    #plt.show()
    plt.savefig('accuracy_face.png')

def get_library_from_path(file_path):
    file_name = file_path[file_path.find("/") + 1:]
    label = file_library_name[file_name]
    return label

def run():
    resolution_distances = {}
    cfp_resolutions_path = "output/cfp_fa_result.txt" 
    distance_means = []
    with open(cfp_resolutions_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line == '':
                continue
            distances = get_distances(line)
            if not distances:
                continue
            dist_mean = np.mean(distances)
            distance_means.append(dist_mean)
            
            #monta estrutura de distancia por resolucao
            resolution_area = get_resolution_area(line)
            area_key = int( resolution_area/10000 + 1) * 10000
            
            if area_key in resolution_distances:
                resolution_distances[area_key].append(dist_mean)
            else:
                resolution_distances[area_key] = [dist_mean]
        
        print_graph(distance_means, "distances")
        print_graph(list(resolution_distances.values()), "resolutions_distances", list(resolution_distances.keys()))

#get_resolution_sum("name: 001/profile\01.jpg, resolution: 158x157, color: RGB, face detected: True, distances: [12.85, 17.62, 27.88, 31.04, 28.19, 24.89, 20.72, 17.03, 7.39, 63.9, 81.49, 94.78, 57.51, 62.88, 56.66, 63.94, 74.62, 74.0, 71.01, 52.4, 47.84, 27.59, 38.01, 25.89], mean: 45.00541666666667")

file_path = "result/cfp_amazon_result.txt"

#distances_per_resolution_area(file_path)
distances_per_point(file_path, list(CorrespondentAmazon.CFP.points.keys()))
#all_distances_boxplot()
#accuracy_face_per_resolution()
#teste()
#teste_2()
#get_match_result("name: 083\profile\02.jpg, resolution: 1336x1220, color: RGB, face detected: Multiple, distances: [], mean: 0")