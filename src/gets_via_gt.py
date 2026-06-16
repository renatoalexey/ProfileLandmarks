import os
import json

json_folder = "via_json"

def gets_all_gt():

    all_images_points = []
    pontos_por_imagem = {}
    json_count = 0
    for json_name in os.listdir(json_folder):
        json_fullpath = os.path.join(json_folder, json_name)

        with open(json_fullpath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        json_info = {}

        for item in data['file'].values():
            json_info[item['fid']] = item['fname']
        
        vid = -1
        i = 0
        landmarks = []
        img_name = ''
        for item in data['metadata'].values():
            if vid != item['vid']:
                i = 0
                vid = item['vid']
                if landmarks:
                    pontos_por_imagem[img_name].append(landmarks)
                    landmarks = []
            x, y = item['xy'][1], item['xy'][2]

            img_name = json_info[vid] 
            
            if img_name not in pontos_por_imagem:
                pontos_por_imagem[img_name] = []
            landmarks.append((x, y))
            i += 1
        
        #all_images_points.append(pontos_por_imagem)
        json_count += 1

    return pontos_por_imagem
    #print(json_count)

    #print(ground_truth_points)    

def gets_gt():
    all_images_points = []
    pontos_por_imagem = {}
    json_count = 0
    for json_name in os.listdir(json_folder):
        json_fullpath = os.path.join(json_folder, json_name)

        with open(json_fullpath, 'r', encoding='utf-8') as file:
            data = json.load(file)

        json_info = {}

        for item in data['file'].values():
            json_info[item['fid']] = item['fname']
        
        vid = -1
        i = 0
        for item in data['metadata'].values():
            if vid != item['vid']:
                i = 0
                vid = item['vid']
            x, y = item['xy'][1], item['xy'][2]

            img_name = json_info[vid] 
            
            if img_name not in pontos_por_imagem:
                pontos_por_imagem[img_name] = [(0, 0)]
            if (i < len(pontos_por_imagem[img_name])):
                x += pontos_por_imagem[img_name][i][0]
                y += pontos_por_imagem[img_name][i][1]
                pontos_por_imagem[img_name][i] = (x,y)
            else:
                pontos_por_imagem[img_name].append((x, y))
            i += 1
        
        #all_images_points.append(pontos_por_imagem)
        json_count += 1

    #print(pontos_por_imagem)
    #print(json_count)
    ground_truth_points =  {
        key: [(x / json_count, y / json_count) for x, y in points]
        for key, points in pontos_por_imagem.items()
    }

    #print(ground_truth_points)    
    return ground_truth_points

#gets_gt()
#teste()