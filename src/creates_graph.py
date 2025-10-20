import re
import numpy as np
import matplotlib.pyplot as plt
import os

def print_graph(means, graph_name, positions=None):
    plt.figure(figsize=(12, 7))
    plt.boxplot(means, positions)
    plt.xlabel('Pontos fiduciais')
    plt.ylabel('Média das distâncias normalizadas entre os pontos (gt e biblioteca)')
    plt.savefig(f'{graph_name}.png')

def get_match_result(str, line):
    return re.search(r"distances:\s*\[(.*?)\]", str)

def get_resolution_sum(line): 
    match = re.search(r"resolution:[^,]*", line)
    resolution_sum = match.group().split(":")[1].split("x")
    return int(resolution_sum[0].strip()) + int(resolution_sum[0].strip())
    
def get_distances(line):
    match = re.search(r"distances:\s*\[(.*?)\]", line)
    if match and match != []:
        values = match.group(1).split(",")   # separa por vírgula
        print(values)
        if values[0] == '':
            return []
        distances = [float(v.strip()) for v in values]  # converte pra float
        return distances
        #print(distances)

def teste():
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
                soma = get_resolution_sum(line)
                soma_key = int( soma/100 + 1) * 100
            distances_tools_list.append(distance_means)
            
            print_graph(distances_tools_list, "distances_3", ["Face Alignment", "ML Kit"])

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
            soma = get_resolution_sum(line)
            soma_key = int( soma/100 + 1) * 100
            
            if soma_key in resolution_distances:
                resolution_distances[soma_key].append(dist_mean)
            else:
                resolution_distances[soma_key] = [dist_mean]
        
        print_graph(distance_means, "distances")
        print_graph(list(resolution_distances.values()), "resolutions_distances", list(resolution_distances.keys()))

#get_resolution_sum("name: 001/profile\01.jpg, resolution: 158x157, color: RGB, face detected: True, distances: [12.85, 17.62, 27.88, 31.04, 28.19, 24.89, 20.72, 17.03, 7.39, 63.9, 81.49, 94.78, 57.51, 62.88, 56.66, 63.94, 74.62, 74.0, 71.01, 52.4, 47.84, 27.59, 38.01, 25.89], mean: 45.00541666666667")
teste()