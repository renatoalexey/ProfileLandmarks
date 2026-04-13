from PIL import Image

#image_name = 'resolution_distances.png'
#image_path_list = ['resolutions_distances_Amazon Rekognition.png', 'resolutions_distances_Face Alignment.png', 'resolutions_distances_ML Kit.png']

image_name_path_list = {'points_distances.png': ['points_distances_Amazon Rekognition.png', 'points_distances_Face Alignment.png', 'points_distances_ML Kit.png']  , 'resolution_distances.png': ['resolutions_distances_Amazon Rekognition.png', 'resolutions_distances_Face Alignment.png', 'resolutions_distances_ML Kit.png']}

for image_name in image_name_path_list.keys():
    image_path_list = image_name_path_list[image_name]
    imgs = [Image.open(f) for f in image_path_list]

    w, h = imgs[0].size
    final = Image.new('RGB', (w, h*3))

    for i, img in enumerate(imgs):
        final.paste(img, (0, i*h))

    final.save(image_name)