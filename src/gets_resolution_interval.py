import cv2
from src.utils import cfp

def run():
    image_path_list = cfp.get_images_paths()

    resolution_area_list = []
    rel_max = 0
    blbla = ""
    for i, image_path in enumerate(image_path_list):
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width  = image.shape[:2]
        resolution_area = height * width
        resolution_area_list.append(resolution_area)

        if resolution_area > rel_max:
            rel_max = resolution_area
            blbla = image_path
    
    print(f"Maior: {blbla},  {rel_max}")
    print(f"Resolution min: {min(resolution_area_list)}")
    print(f"Resolution max: {max(resolution_area_list)}")

#run()
    
