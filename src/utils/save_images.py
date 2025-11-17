import matplotlib
matplotlib.use("Agg")   # backend sem interface gráfica
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from . import cfp
from . import core
from src.face_alignment.correspondent_fa_type import CorrespondentFaceAlignment
from .face_type import FaceType

def fiducial_points(image, ground_truth_points, library_points, correspondent_points, save_path):
    plt.clf()
    plt.imshow(image)
    for i, ground_truth_point in enumerate(ground_truth_points, start=0):
        if correspondent_points.get(i) is not None:
            library_point = library_points[correspondent_points.get(i)]
            x, y = ground_truth_point
            plt.scatter(x, y, color="red", s=10)
            plt.annotate(str(i + 1), (x, y), textcoords="offset points", xytext=(0,5), ha='center', color="red", fontsize=8)
            
            a, b = library_point
            plt.scatter(a, b, color="blue", s=10)
            plt.annotate(str(correspondent_points.get(i) + 1), (a, b), textcoords="offset points", xytext=(0,5), ha='center', color="blue", fontsize=8)
        
        #plt.imshow(img)
        plt.axis("off")
        plt.savefig(save_path)
        
def bounding_boxes(image, library_points_list, output_path=None):
    plt.clf()
    fig, ax = plt.subplots()
    ax.imshow(image)   
    #plt.imshow(image)
    for library_points in library_points_list:
        x_min = np.min(library_points[:, 0])
        y_min = np.min(library_points[:, 1])
        x_max = np.max(library_points[:, 0])
        y_max = np.max(library_points[:, 1])

                
        # Cria retângulo
        width = x_max - x_min
        height = y_max - y_min

        rectangle = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )

        ax.add_patch(rectangle)
    
    plt.axis("off")
    plt.savefig(output_path)
    #plt.show()
    plt.close()

#ground_truth_file_path = 'F:\\Bases\\cfp-dataset\\Data\\Fiducials\\446\\profile\\04.txt'
#ground_truth_file_path = 'F:\\Bases\\cfp-dataset\\Data\\Fiducials\\085\\profile\\01.txt'
#img_path = core.get_image_path(ground_truth_file_path)

def tests_face_alignment():
    output_path = "output/face_alignment/"
    #ground_truth_file_path = 
    images_paths = ['F:\\Bases\\cfp-dataset\\Data\\Images\\085\\profile\\01.jpg']
    
    for img_path in images_paths:
        ground_truth_points_list, image = cfp.get_ground_truth_points(img_path)
                
        fa_points_list, face_detected = core.get_face_alignment_points(image)

        correspondent_points = CorrespondentFaceAlignment.CFP.points
        img_suffix = img_path[img_path.index("Images") + 7: len(img_path)].replace("/", "_")
        print(f"{output_path}{img_suffix}")

        if face_detected == FaceType.ONE:
            fiducial_points(image, ground_truth_points_list, fa_points_list, correspondent_points, f"{output_path}{img_suffix}")
        elif face_detected == FaceType.MULTIPLE:
            for fa_p in fa_points_list:
                if teste(image, fa_p):
                    bounding_boxes(image, fa_points_list)

def teste(image, library_points_list):
    image_height, image_width = image.shape[:2]
    
    x_min = np.min(library_points_list[:, 0])
    y_min = np.min(library_points_list[:, 1])
    x_max = np.max(library_points_list[:, 0])
    y_max = np.max(library_points_list[:, 1])

    width = x_max - x_min
    height = y_max - y_min

    return width * height > 0.2 * ( image_height * image_width )
#tests_face_alignment()