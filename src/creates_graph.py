import re
import numpy as np
import matplotlib.pyplot as plt
import os

def print_graph(means, graph_name, positions=None):
    plt.figure(figsize=(12, 7))
    plt.boxplot(means, positions, showmeans=True, meanline=True, tick_labels=positions)
    plt.xlabel('Bibiotecas')
    plt.ylabel('Média das distâncias normalizadas entre os pontos (groud truth e biblioteca)')
    plt.savefig(f'{graph_name}.png')

def get_face_detected(line):
    match = re.search(r"face detected:[^,]*", line)
    return match.group().split(":")[1].strip()

def get_resolution_area(line): 
    match = re.search(r"resolution:[^,]*", line)
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

def get_output_files():

    output_path = 'output'
    distances_tools_list = []
    files_names = os.listdir(output_path)

    return list(map(lambda f_name: f"{output_path}/{f_name}", files_names))

def get_file_lines(file_path):
    with open(file_path, 'r') as file:
        return file.readlines()

def all_distances_boxplot():
    output_path = 'output'
    distances_tools_list = []
    for result_file in os.listdir(output_path):
        distance_means = []
        result_file_full_path = os.path.join(output_path, result_file)
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
            
    print_graph(distances_tools_list, "distances_4", ["Face Alignment", "ML Kit"])

def distances_per_point(file_path):
    lines = get_file_lines(file_path)
    keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 25, 28, 26, 29]
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

    print_graph(point_distances_list, "points_distances", keys)

def distances_per_resolution_area(file_path):
    resolution_distances = {}
    lines = get_file_lines(file_path)

    resolution_area_list = []
    for line in lines:
        if line == '':
            continue
        distances = get_distances(line)
        if not distances:
            continue

        dist_mean = np.mean(distances)
        
        #monta estrutura de distancia por resolucao
        resolution_area = get_resolution_area(line)
        area_key = int( resolution_area/40000 + 1) * 40000
        
        resolution_area_list.append(resolution_area)

        if area_key in resolution_distances:
            resolution_distances[area_key].append(dist_mean)
        else:
            resolution_distances[area_key] = [dist_mean]

    print(f"Resolution min: {min(resolution_area_list)}")
    print(f"Resolution max: {max(resolution_area_list)}")
    print(f"Resolution distances: {resolution_distances}")
    print_graph(list(resolution_distances.values()), "resolutions_distances_new", list(resolution_distances.keys()))

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

def accuracy_face_per_resolution():

    all_files = get_output_files()
    rels_interval_list = get_rel_interval(all_files[0])
    x_keys = []
    for file_path in all_files:
        
        lines = get_file_lines(file_path)

        resolution_distances = {}
        for line in lines:
            if line == '':
                continue
            
            #monta estrutura de distancia por resolucao
            resolution_area = get_resolution_area(line)
            face_detected = get_face_detected(line)
            area_key = int( resolution_area/200000 + 1) * 200000
            
            if area_key in resolution_distances:
                resolution_distances[area_key].append(face_detected)
            else:
                resolution_distances[area_key] = [face_detected]

        arrei = []
        for teste in list(resolution_distances.values()):
            count_truth = teste.count('True')
            arrei.append( round(count_truth * 100 / len(teste), 2))

        x_keys = sorted(list(resolution_distances.keys()))
        #plt.plot(list(map(lambda x: str(int(x/10000)), x_keys)), arrei, label=file_path, marker='o')
        plt.plot(x_keys, arrei, label=file_path, marker='o')
    blabla = list(map(lambda x: str(int(x/10000)), x_keys))
    #plt.xticks(range(len(blabla)), blabla)
    plt.xticks(x_keys)
    #plt.xticks(blabla)
    plt.xlabel("Eixo X")
    plt.ylabel("Percentual de acerto")
    plt.legend()
    #plt.grid(True)
    # Mostra o gráfico
    #plt.show()
    plt.savefig('accuracy_face.png')

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

file_path = "output/cfp_fa_result.txt"

#distances_per_resolution_area(file_path)
#distances_per_point(file_path)
#all_distances_boxplot()
accuracy_face_per_resolution()
#get_match_result("name: 083\profile\02.jpg, resolution: 1336x1220, color: RGB, face detected: Multiple, distances: [], mean: 0")