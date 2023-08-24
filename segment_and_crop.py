# This file is used later in bb_infer.py file
from rembg import remove
import os
import cv2
import numpy as np
import json

IMAGE_PATH = '/root/data/apple/original/images/'
DEST_PATH = '/root/data/apple/cropped-apple-bb/crop-images-bb/'

def segment_crop(img_path):
    """
    Perform segmentation and center crop on an image.
    
    Args:
        img_path (str): Path to the input image file.
    
    Returns:
        numpy.ndarray: Cropped and resized image.
    """

    # Read the image and get its dimensions
    image = cv2.imread(img_path)
    img_height, img_width = image.shape[:2]
    
    # Preprocess the image
    image = remove(image)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to obtain a binary image
    thresh = 1
    im_bw = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(im_bw.astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding rectangle coordinates of the largest contour
    (x, y, w, h) = cv2.boundingRect(largest_contour)
    
    # Calculate the center coordinate
    xc = x + w // 2
    yc = y + h // 2

    # Determine the crop size
    if w > h:
        crop_size = (w, w)
    else:
        crop_size = (h, h)
    
    # Calculate crop boundaries
    x_min = max(0, xc - crop_size[0] // 2)
    y_min = max(0, yc - crop_size[1] // 2)
    x_max = min(img_width, x_min + crop_size[0])
    y_max = min(img_height, y_min + crop_size[1])

    # Perform center crop
    cropped_image = image[y_min:y_max, x_min:x_max]
    
    # Resize the cropped image to the desired size
    desired_size = (1024, 1024)
    result = cv2.resize(cropped_image, desired_size)
    
    return result, x_min / img_width, y_min / img_height, img_width / crop_size[0], img_height / crop_size[1]

if __name__ == '__main__':
    JSON_FILE = './ground-truth-bbox/inference_modified.json'
    OUTPUT_JSON_FILE = './ground-truth-bbox/inference_modified_210623.json'
    data = []
    img_path = [os.path.join(IMAGE_PATH, item) for item in os.listdir(IMAGE_PATH)]

    import time
    
    for img in img_path:
        start = time.time()
        basename = os.path.basename(img)
        cropped, x_crop, y_crop, ratio_crop_w, ratio_crop_h = segment_crop(img)
        cv2.imwrite(f'{DEST_PATH}{basename}', cropped)
        
        item = {
            'name': basename,
            'image_path': os.path.join(IMAGE_PATH, basename),
            'cropped_image_path': f'{DEST_PATH}{basename}',
            'class': '',
            'state': '',
            'coordinates': [],
            'modified_coords': [],
            'cropped_info': {'x_crop': x_crop,
                            'y_crop': y_crop,
                            'ratio_crop_w': ratio_crop_w,
                            'ratio_crop_h': ratio_crop_h} 
        }
        
        end = time.time()
        print(f'{end-start:.2f} s')
        data.append(item)

    with open(JSON_FILE, 'r') as json_file:
        existing_data = json.load(json_file)
    
    for item, existing_item in zip(data, existing_data):
        item['class'] = existing_item['class']
        item['state'] = existing_item['state']
        item['coordinates'] = existing_item['coordinates']
        item['modified_coords'] = existing_item['modified_coords']
        
    with open(OUTPUT_JSON_FILE, 'w') as json_output:
        json.dump(data, json_output, indent=4)
