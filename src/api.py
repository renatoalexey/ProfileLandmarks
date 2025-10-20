from flask import Flask, jsonify, request
import os
import json
import ast
import utils
import sys

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

@app.route("/ground/truth/points", methods=["GET"])
def get_ground_truth_points():
    fiducials_folder = str(request.args.get("fiducials_folder"))
    ground_truth_pts = []

    #print(f"Fiducials folder: {fiducials_folder}")
    
    if os.path.exists(fiducials_folder):
        with open(fiducials_folder, 'r') as file:
            lines = file.readlines()
            for line in lines:
                x, y = line.split(',')
                x = float(x)
                y = float(y)
                ground_truth_pts.append((x, y))

    return jsonify({"ground_truth_pt": ground_truth_pts})

@app.route("/compare/points", methods=["GET"])
def get_compare_results():
    fiducials_file_path = request.args.get("fiducials_folder")
    library_pts = request.args.get("library_pts")
    #print(f"Fiducial points: {fiducials_file_path}")

    image_path = utils.get_image_path(fiducials_file_path)
    face_detected = False
    all_distances = []
    
    library_pts = ast.literal_eval(library_pts.strip("'"))
    #print(f"Library points: {library_pts}")
    
    if library_pts:
        face_detected = True
        ground_truth_pts = get_gt_points(fiducials_file_path)
        
        vertical_distance = utils.calc_euclidean_distance(ground_truth_pts[vertical_point_a][0], ground_truth_pts[vertical_point_a][1],
                                              ground_truth_pts[vertical_point_b][0], ground_truth_pts[vertical_point_b][1])
        horizontal_distance = utils.calc_euclidean_distance(ground_truth_pts[horizontal_point_a][0], ground_truth_pts[horizontal_point_a][1],
                                              ground_truth_pts[horizontal_point_b][0], ground_truth_pts[horizontal_point_b][1])
        
        all_distances = utils.compare_points(ground_truth_pts, library_pts, correspondet_points, vertical_distance, horizontal_distance)
    
    utils.writes_euclidean_distances(image_path, face_detected, all_distances, file_path)
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

file_path = "profile_results/cfp_mlkit_result.txt" 
if os.path.exists(file_path):
    os.remove(file_path)    
if __name__ == "__main__":
    # 0.0.0.0 = acessível de outros dispositivos da rede
    app.run(host="0.0.0.0", port=5000, debug=True)
