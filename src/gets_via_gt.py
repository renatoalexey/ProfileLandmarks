import os
import json

json_folder = "via_json"

def run():
    all_images_points = []
    pontos_por_imagem = {}
    json_count = 0
    for json_name in os.listdir(json_folder):
        json_fullpath = os.path.join(json_folder, json_name)

        with open(json_fullpath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        for i, item in enumerate(data['metadata'].values(), start=0):
            vid = item['vid']
            x, y = item['xy'][1], item['xy'][2]
            
            if vid not in pontos_por_imagem:
                pontos_por_imagem[vid] = []
            if (pontos_por_imagem[vid] and i == 0):
                x += pontos_por_imagem[vid][i][0]
                y += pontos_por_imagem[vid][i][1]
                pontos_por_imagem[vid][i] = (x,y)
            else:
                pontos_por_imagem[vid].append((x, y))
        
        #all_images_points.append(pontos_por_imagem)
        json_count += 1

    #print(pontos_por_imagem)
    ground_truth_points =  {
        key: [(x / 2, y / 2) for x, y in points]
        for key, points in pontos_por_imagem.items()
    }

    print(ground_truth_points)    

run()