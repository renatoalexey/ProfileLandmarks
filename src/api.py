from flask import Flask, jsonify, request, send_file
import os
import json
import ast
import sys
from src.utils import cfp
from src.utils import core
from src.utils import save_images 
from src.utils.face_type import FaceType
from io import BytesIO
from PIL import Image
import numpy as np

#sys.path.append(os.path.dirname(__file__))


correspondet_points = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 10, 10: 18, 11: 20, 12: 23, 13: 37, 15: 38, 17: 40, 18: 48, 19: 28, 20: 29, 21: 30, 22: 31, 25: 32, 28: 53, 26: 49, 29: 13} 
vertical_point_a = 11
vertical_point_b = 8
horizontal_point_a = 0
horizontal_point_b = 18

vertical_distance = 1.0
horizontal_distance = 1.0

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"mensagem": "Servidor Python funcionando 🚀"})

@app.route("/soma", methods=["GET"])
def soma():
    a = int(request.args.get("a", 0))
    b = int(request.args.get("b", 0))
    return jsonify({"resultado": a + b})

@app.route("/image", methods=["GET"])
def get_ground_truth_points():
    image_path = str(request.args.get("image_path"))
    #print(f"Fiducials folder: {fiducials_folder}")
    ground_truth_points_list, image = cfp.get_ground_truth_points(image_path)   

    # Converte o array NumPy em um objeto PIL
    pil_img = Image.fromarray(image)

    # Salva em memória como PNG
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)

    # Retorna como resposta HTTP
    #print(f"Image path: {image_path}")
    #print(f"Ground truth: {ground_truth_points_list}")
    return send_file(buffer, mimetype="image/png")

    #return jsonify({"ground_truth_pt": ground_truth_pts})

@app.route("/compare/points", methods=["GET"])
def get_compare_results():
    image_path = request.args.get("image_path")
    library_pts = request.args.get("library_pts")
    #print(f"Fiducial points: {fiducials_file_path}")

    face_detected = False
    all_distances = []
    
    library_pts = ast.literal_eval(library_pts.strip("'"))
    #print(f"Library points: {library_pts}")
    
    ground_truth_points_list, image = cfp.get_ground_truth_points(image_path)

    img_suffix = image_path[image_path.index("Images") + 7: len(image_path)].replace("/", "_")
    save_path = f"output/ml_kit/{img_suffix}"
    
    distances_list = []
    size = len(library_pts)
    print(f"Image path compare: {image_path}")
    face_detected = FaceType.ONE
    if  size == 1:
        print("Uma face")
        save_images.fiducial_points(image, ground_truth_points_list, library_pts[0], correspondet_points, save_path)
        distances_list.append(core.get_euclidean_results(ground_truth_points_list, library_pts[0], correspondet_points, image))
    elif size > 1:
        print("Múltiplas face")
        #print(library_pts)
        face_detected = FaceType.MULTIPLE
        #library_pts = [np.array(lib) for lib in library_pts]
        library_pts = np.array(library_pts)
        print(library_pts)
        save_images.bounding_boxes(image, library_pts, save_path)
        for face_points in library_pts:
            pts = np.array(face_points)
            #x_min, x_max = pts[:,0].min(), pts[:,0].max()
            #y_min, y_max = pts[:,1].min(), pts[:,1].max()
            face_points_list = [list(p) for p in face_points]
            #print(face_points_list)
            if core.valids_bounding_box(image, face_points):
                distances_list.append(core.get_euclidean_results(ground_truth_points_list, face_points, correspondet_points, image))
    else: 
        print("Nenhuma face")
        distances_list = [[]]
        face_detected = FaceType.NONE

    core.writes_euclidean_distances(image_path, face_detected.value, distances_list, "output/cfp_mlkit_result.txt")
    
    return jsonify({"teste": 123})

def get_gt_points(fiducials_file_path):

    ground_truth_pts = []

    #print(f"Fiducials folder: {fiducials_folder}")
    
    if os.path.exists(fiducials_file_path):
        with open(fiducials_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                x, y = line.split(',')
                x = float(x)
                y = float(y)
                ground_truth_pts.append((x, y))

    return ground_truth_pts

file_path = "output/cfp_mlkit_result.txt" 
if os.path.exists(file_path):
    os.remove(file_path)    
if __name__ == "__main__":
    # 0.0.0.0 = acessível de outros dispositivos da rede
    app.run(host="0.0.0.0", port=5000, debug=True)
