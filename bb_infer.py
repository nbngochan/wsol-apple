import cv2
import json

CROPPED_PATH_BB = 'D:/mnt/data_source/cropped-apple-bb/crop-images-bb/'
MODIFILED_JSON_210623 = 'D:/lab/dataset/HelloAppleWorld/Yolo-type/classification/inference_modified_210623.json'
MODIFILED_JSON_2106355 = 'D:/lab/dataset/HelloAppleWorld/Yolo-type/classification/inference_modified_2106355.json'


with open(MODIFILED_JSON_210623, 'r') as json_file:
    data = json.load(json_file)

for item in data:
    name = item['name']
    cropped_info = item['cropped_info']
    image = cv2.imread(item['image_path'])
    image_crop = cv2.imread(item['cropped_image_path'])

    crop_coordinates_ratio = []
    
    for coord in item['coordinates']:
        x_crop = cropped_info['x_crop']
        y_crop = cropped_info['y_crop']
        ration_crop_w = cropped_info['ratio_crop_w']
        ration_crop_h = cropped_info['ratio_crop_h']
        
        xc, yc, width, height = coord[0], coord[1], coord[2], coord[3]
        x_new, y_new, width, height = xc - x_crop, yc - y_crop, width, height  # old ratio
        x_new, y_new, width, height = x_new*ration_crop_w, y_new*ration_crop_h, width*ration_crop_w, height*ration_crop_h  # from old ratio to new ratio (after crop)
        
        crop_coordinates_ratio.append((x_new, y_new, width, height))
    
    item['crop_coordinates_ratio'] = crop_coordinates_ratio

with open(MODIFILED_JSON_2106355, 'w') as json_output:
    json.dump(data, json_output, indent=4)
    

with open(MODIFILED_JSON_2106355, 'r') as json_file:
    data = json.load(json_file)
    
     # Loop through each item in the JSON data
    for item in data:
        name = item['name']
        image = cv2.imread(item['cropped_image_path'])
        height, width = image.shape[:2]
        
        # Iterate over modified coordinates and draw bounding boxes on the image
        for coord in item['crop_coordinates_ratio']:
            x, y, w, h = coord[0], coord[1], coord[2], coord[3]
            
            x1 = int((x - w / 2) * width)
            y1 = int((y - h / 2) * height)
            x2 = int((x + w / 2) * width)
            y2 = int((y + h / 2) * height)
            
            # Draw the bounding box rectangle on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Save the image with bounding boxes
        cv2.imwrite(f'{CROPPED_PATH_BB}{name}_crop.jpg', image)