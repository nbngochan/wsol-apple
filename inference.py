import torch
import cv2
import os
import torchvision.transforms as transforms
from train_classifier import SimpleModel
from PIL import Image
import numpy as np
from torch.nn import functional as F
import json
from collections import Counter
from sklearn.metrics import confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import pickle


# CHECKPOINT_PATH = 'D:/lab/project/weakly-segmentation/classification/pytorch-image-classification/results/tb_logs/lightning_logs/version_45/checkpoints/best_model_012-0.1648-0.94.ckpt'
# OUTPUT_DIR = 'D:/lab/dataset/HelloAppleWorld/Yolo-type/classification/2206/scale0.25/'
# JSON_FILE = 'D:/lab/dataset/HelloAppleWorld/Yolo-type/classification/inference_modified_2106355.json'
JSON_FILE = './ground-truth-bbox/inference_modified.json'
OUTPUT_DIR = '/root/data/apple/cropped-apple-bb/image/'
CHECKPOINT_PATH = './results/tb_logs/lightning_logs/version_50/checkpoints/best_model_012-0.1877-0.94.ckpt'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ])

checkpoint_path = CHECKPOINT_PATH
model = SimpleModel.load_from_checkpoint(checkpoint_path)
model = model.to(device)
model = model.eval()
target_layers = model.model.layer4  


def predict_test():
    results = {}
    scales = [0.25, 0.5, 1]
    
    for scale in scales:
        results[f'scale{scale}'] = {'results': [], 'cam': [], 'image_path': []}
        
        with open(JSON_FILE, 'r') as json_file:
            inference_data = json.load(json_file)
        
        for item in inference_data[:500]:
            image_path = item['cropped_image_path']
            print(scale, os.path.basename(image_path))
            label, cam = predict(image_path, scale=scale)
            print(label)
            
            results[f'scale{scale}']['results'].append(label)
            results[f'scale{scale}']['cam'].append(cam)
            results[f'scale{scale}']['image_path'].append(image_path)
    
    # Save with pickle to preserve the original structure and type
    with open('results.pickle', 'wb') as pickle_file:
        pickle.dump(results, pickle_file)
        
    return results
    
def predict(image_path, scale):
    classes = ['defective', 'normal']
    
    image = Image.open(image_path)
    width, height = image.size
    new_width, new_height = int(width * scale), int(height * scale)
    image_scaled = image.resize((new_width, new_height))

    input_tensor = transform(image_scaled)
    input_tensor = input_tensor.unsqueeze(0).to(device)

    output = model(input_tensor)
    h_x = F.softmax(output, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()

    cam_algorithm = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    with cam_algorithm as cam:
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]
        grayscale_cam_rz = cv2.resize(grayscale_cam, (width, height))
        
    # Delete variables and tensors to release memory
    del image, image_scaled, input_tensor, output, h_x, cam_algorithm, grayscale_cam
    torch.cuda.empty_cache()

    return classes[idx[0]], grayscale_cam_rz


def infer():
    with open(JSON_FILE, 'r') as json_file:
        inference_data = json.load(json_file)
    
    y_true = [int(label['class']) for label in inference_data]
    y_pred = []
    classes = ['defective', 'normal']
    class_list = []
    
    for item in inference_data:
        img = item['cropped_image_path']
        # img = item['image_path']
        basename = os.path.basename(img)
        rgb_img = cv2.imread(img, 1)[:, :, ::-1]
        
        height, width = rgb_img.shape[:2]
        height, width = int(height), int(width)
        
        rgb_img = np.float32(rgb_img) / 255.0
        rgb_scale = cv2.resize(rgb_img, (width, height))

        image = Image.open(img)
        image = image.resize((width, height))
        
        input_tensor = transform(image)
        input_tensor = input_tensor.unsqueeze(0).to(device)
        
        output = model(input_tensor)
        h_x = F.softmax(output, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()
        
        y_pred.append(idx[0])
        class_list.append(classes[idx[0]])
            
        cam_algorithm = GradCAM
        with cam_algorithm(model=model, target_layers=target_layers, use_cuda=True) as cam:
            grayscale_cam = cam(input_tensor=input_tensor)
            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam_rz = cv2.resize(grayscale_cam, (width, height))
            
            cam_image = show_cam_on_image(rgb_scale, grayscale_cam_rz, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            
        y = 40
        line_height = 30
        for i in range(len(classes)):
            print(f'{probs[i]:.3f} -> {classes[idx[i]]}')
            text = f'{classes[idx[i]]}: {probs[i]:.3f}'
            cv2.putText(cam_image, text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y += line_height
        
        # Write all the folder
        cv2.imwrite(f'{OUTPUT_DIR}/CAM/CAM_{basename}', cam_image)
        cv2.imwrite(f'{OUTPUT_DIR}/GRAYSCALE/grayscale_heatmap_{basename}', grayscale_cam_rz * 255.0)

    # Print confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Save the confusion matrix to a file
    np.savetxt(f'{OUTPUT_DIR}/confusion_matrix.txt', cm, fmt='%d')
     
    with open(f'{OUTPUT_DIR}/output.json', 'w') as file:
        count = Counter(class_list)
        json_obj = json.dumps(count)
        file.write(json_obj)

if __name__ == '__main__':
    infer()
    # han = predict_test()

