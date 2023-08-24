import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torch.nn import functional as F
import json
import torch
import os
from pytorch_grad_cam import GradCAM
from train_classifier import SimpleModel
import pickle
import time
import pandas as pd
from utils import calculate_mean_iou, non_max_suppression
from tqdm import tqdm
from vit import VitGenerator, visualize_predict
# from skimage.segmentation import slic
from skimage.segmentation import watershed
from cuda_slic.slic import slic as slic
from skimage.measure import regionprops
import selective_search


def im2double(img):
    min_val = np.min(img.ravel())
    max_val = np.max(img.ravel())
    out = (img.astype("float") - min_val) / (max_val - min_val)
    return out

def show_normalized_image(img):
    plt.imshow(img.astype("uint8"))
    plt.axis("off")
    plt.show()
    

class CombineCam:
    """The Class containing the information needed to combine multiple scale of CAM images
    Input:
    - threshold: int
    - model: str
    - json: str
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        # transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])
    
    def __init__(self, threshold, checkpoint_path, json_file):
        self.vit_name = 'vit_small'
        self.patch_size = 8
        self.threshold = threshold
        self.json_file = json_file
        self.cnn_model = SimpleModel.load_from_checkpoint(checkpoint_path).to(self.device).eval()
        self.vit_model = VitGenerator(self.vit_name, self.patch_size, self.device, checkpoint_path=None, evaluate=True, random=False, verbose=True)
        self.target_layers = self.cnn_model.model.layer4
        self.cam_algorithm = GradCAM(model=self.cnn_model, target_layers=self.target_layers, use_cuda=True)
        
    
    def _predict(self, image_path, scale):
        """Predict the label and CAM image for an image at a given scale."""
        classes = ['defective', 'normal']
        image = Image.open(image_path)
        width, height = image.size
        new_width, new_height = int(width * scale), int(height * scale)
        image_scaled = image.resize((new_width, new_height))
    
        input_tensor = self._preprocess_image(image_scaled)
        output = self.cnn_model(input_tensor)
        h_x = F.softmax(output, dim=1).data.squeeze()
        _, idx = h_x.sort(0, True)
        label = classes[idx[0]]

        cam_method = self.cam_algorithm
            
        grayscale_cam = cam_method(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam.squeeze(0)
        grayscale_cam_rz = cv2.resize(grayscale_cam, (width, height))
    
        return label, grayscale_cam_rz

    def _preprocess_image(self, image):
        input_tensor = self.transform(image)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        return input_tensor

    def predict(self, image_path, scale):
        """Predict the label and CAM image for an image at a given scale."""
        label, cam = self._predict(image_path, scale)
        return label, cam
    
    
    def get_fusedcam_watershed(self, fused_cam):
        # Apply threshold to obtain binary mask
        binary_mask = np.zeros_like(fused_cam)
        binary_mask[im2double(fused_cam) >= self.threshold] = 255

        # Perform morphological operations to clean up the binary mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = binary_mask.astype(np.uint8)
        
        # Perform distance transform on the binary mask
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)

        # Apply threshold to obtain markers for watershed
        markers = np.zeros_like(binary_mask)
        markers[dist_transform > 0.7 * dist_transform.max()] = 255

        # Perform watershed segmentation
        markers = cv2.connectedComponents(markers)[1]
        labels = watershed(-dist_transform, markers, mask=binary_mask)
        
        bounding_boxes = []
        for label in np.unique(labels):
            if label == 0:
                continue
            mask = labels == label
            nonzero_pixels = np.argwhere(mask)
            min_y, min_x = np.min(nonzero_pixels, axis=0)
            max_y, max_x = np.max(nonzero_pixels, axis=0)
            bounding_boxes.append([min_x, min_y, max_x, max_y])
        return bounding_boxes
    
    
    def get_fusedcam_slic(self, image_path, fused_cam, n_segments):
        image = cv2.imread(image_path)[:, :, ::-1]
        bg_indices = np.where((image == [0, 0, 0]).all(axis=2))
        white_img = image.copy()
        white_img[bg_indices] = [255, 255, 255]
        
        # Apply SLIC for superpixel segmentation
        segments = slic(white_img, n_segments=n_segments)
        
        slic_bbox = []

        props = regionprops(segments)
        for prop in props:
            minr, minc, maxr, maxc = prop.bbox
            slic_bbox.append([minc, minr, maxc, maxr])

        # Calculate average CAM value within each superpixel
        # cam_avg_values = []
        
        # cam = im2double(fused_cam)
        
        # for segment in np.unique(segments):
        #     mask = segments == segment
        #     avg_value = np.mean(cam[mask])
            
        #     cam_avg_values.append(avg_value)
        
        # # Select superpixels with high average CAM values as potential defective regions
        # potential_defective_segments = np.where(np.array(cam_avg_values) > self.threshold)[0]

        # # Extract bounding boxes for potential defective regions
        # bounding_boxes = [slic_bbox[segment-1] for segment in potential_defective_segments]

        score_map = np.zeros_like(fused_cam)
        cam = im2double(fused_cam)

        for segment in np.unique(segments):
            mask = segments == segment
            avg_value = np.mean(cam[mask])
            score_map[mask] = avg_value

        # Select superpixels with high average CAM values as potential defective regions
        potential_defective_segments = np.where(score_map > self.threshold)
        row_indices = potential_defective_segments[0]
        column_indices = potential_defective_segments[1]

        # Extract bounding boxes for potential defective regions
        bounding_boxes = []
        for row, col in zip(row_indices, column_indices):
            segment_label = segments[row, col]
            if segment_label < len(slic_bbox):
                bounding_boxes.append(slic_bbox[segment_label])
        # import pdb; pdb.set_trace()
        return bounding_boxes
    

    def get_fusedcam_selectivesearch(self, image_path, fused_cam, topN):
        desire_size = (512, 512)
        image = cv2.imread(image_path)
        # image = cv2.resize(image, desire_size)
        height, width = image.shape[:2]
        
        # Convert the fused CAM image to grayscale
        fused_cam = cv2.resize((fused_cam*255).astype(np.uint8), desire_size)
        
        # Apply Selective Search for region proposal
        rgb_fusedcam = cv2.cvtColor(fused_cam, cv2.COLOR_GRAY2RGB)
        regions = selective_search.selective_search(rgb_fusedcam, mode='fast')
        boxes_filter = selective_search.box_filter(regions, min_size=40, topN=topN)
        
        # Convert Selective Search results to bounding boxes
        bounding_boxes = []
        for x1, y1, x2, y2 in boxes_filter:
            bbox_ratio = [x1 / desire_size[1], y1 / desire_size[0], x2 / desire_size[1], y2 / desire_size[0]]
            bounding_boxes.append(bbox_ratio)
        
        # Refine the bounding boxes using FusedCAM
        refined_bounding_boxes = []
        
        for bbox_ratio in bounding_boxes:
            # Convert bounding box coordinates to (minc, minr, maxc, maxr) format (in terms of pixels)
            minc = int(bbox_ratio[0] * width)
            minr = int(bbox_ratio[1] * height)
            maxc = int(bbox_ratio[2] * width)
            maxr = int(bbox_ratio[3] * height)
            
            # Extract the region within the bounding box from the image
            roi = rgb_fusedcam[minr:maxr, minc:maxc]
            
            # Check if the roi is empty
            if roi.size == 0:
                continue
            
            # Calculate the average CAM value within the region
            avg_value = np.mean(im2double(roi))
            
            # If the average CAM value is above the threshold, consider it a potential defective region
            if avg_value > self.threshold:
                refined_bounding_boxes.append([minc, minr, maxc, maxr])
            
        
        # Apply Non-Maximum Suppression (NMS) to merge overlapping bounding boxes
        nms_bboxes = non_max_suppression(refined_bounding_boxes, rgb_fusedcam)
        
        # # Draw the final bounding boxes on the image
        # for bb in nms_bboxes:
        #     x1, y1, x2, y2 = bb
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # image_name = os.path.basename(image_path)
        # cv2.imwrite(f'./selective-search/{image_name}', image)
        return nms_bboxes
    
    
    def predict_scales(self, scales = [0.25, 0.5, 1], file_path='cropped_image_path'):
        """Predict CAM images at different scales for all images in inference_data.

        Args:
            scales (list of float): List of scales to use for prediction.

        Returns:
            dict: Dictionary containing the predicted results for each image and scale.
                The structure is {'image_name': {'scale': {'label': [], 'cam': []}}}.
        """
                
        results = {}
        with open(self.json_file, 'r') as file:
            inference_data = json.load(file)
            
        image_paths = [item[file_path] for item in inference_data]
        
        
        total_images = len(image_paths)
        progress_bar = tqdm(total=total_images, desc='Processing Images')

        start_all = time.time()
        for image in image_paths:
            start = time.time()
            image_name = os.path.basename(image).split('.')[0]
            image_results = {'scale0.25': {'label': None, 'cam': None},
                            'scale0.5': {'label': None, 'cam': None},
                            'scale1': {'label': None, 'cam': None},
                            'vit': {'attention_map': None}}
            
            for scale in scales:
                # Predict the label and CAM image for the current image and scale
                label, cam = self.predict(image, scale)
                
                # Append the label and CAM image to the corresponding scale in the image_results
                image_results[f'scale{scale}']['label'] = label
                image_results[f'scale{scale}']['cam'] = cam
            
            image_pil = Image.open(image)
            factor_reduce = 2
            img_size = tuple(np.array(image_pil.size[::-1]) // factor_reduce)
            attention_map = visualize_predict(self.vit_model, image_pil, img_size, self.patch_size, self.device)
            attention_map = im2double(np.mean(attention_map, axis=0))
            attention_map = cv2.resize(attention_map, (image_pil.size[0], image_pil.size[1]))
            # print(attention_map.shape)
            image_results['vit']['attention_map'] = attention_map
            
            results[image_name] = image_results

            end = time.time()
            elapsed_time = end - start
            progress_bar.set_postfix({'Time': f'{elapsed_time:.2f}s'})
            progress_bar.update(1)
            

            print(f'Processed {image_name} in {end - start:.2f} seconds.')
            
        progress_bar.close()
        end_all  = time.time()
        process_time = (end_all - start_all) / 60

        print(f'Processed {len(image_paths)} in {process_time:.2f} minutes.')
        
        # Save with pickle to preserve the original structure and type
        with open('./multiscale-dict/with-vit/multiscale_vit_version2.pickle', 'wb') as pickle_file:
            pickle.dump(results, pickle_file)
           
        return results
    
    
    def combine_cam_images(self, *cams, weights=None):
        """Combine multiple CAM images using element-wise operations.

        Args:
            *cams: Variable number of CAM images to be combined.
            weights (list or None, optional): Weights for the weighted sum operation. 
                                            If None, equal weights will be used for all CAM images. 
                                            Defaults to None.

        Returns:
            tuple: A tuple containing the combined CAM images using different strategies:
                - add (ndarray): Element-wise sum of the CAM images.
                - mean (ndarray): Element-wise mean of the CAM images.
                - max (ndarray): Element-wise maximum of the CAM images.
                - weighted (ndarray): Weighted sum of the CAM images.
        """
        
        valid_cams = [cam for cam in cams if cam is not None]
        
        if len(valid_cams) == 0:
            return None
        
        # Add element-wise
        add = np.sum(valid_cams)
        
        # Mean element-wise
        mean = np.mean(valid_cams, axis=0)
        
        # Max element-wise
        max = np.maximum.reduce(valid_cams)
        
        # Weighted sum
        if weights is None:
            weights = [0.5] * len(valid_cams)

        weighted = np.average(valid_cams, axis=0, weights=weights)
        
        return add, mean, max, weighted
    
    def localize_defective(self, image, cam, threshold):
        threshold = threshold
        min_box_area = 1600  # minimum pixel of 40x40

        # Threshold the CAM to obtain binary mask
        # import pdb; pdb.set_trace()
        binary_mask = np.zeros_like(cam)
        binary_mask[cam >= threshold] = 255

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

        # Extract bounding box coordinates
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
        bounding_boxes = [bb for bb in bounding_boxes if bb[2]*bb[3] > min_box_area]

        # Create a copy of the image for drawing bounding boxes
        img_with_boxes = image.copy()

        # Draw bounding boxes on the original image
        proposed_boxes = []
        
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            x1, y1, x2, y2 = x, y, x + w, y + h
            proposed_boxes.append((x1, y1, x2, y2))
        
        # show_normalized_image(img_with_boxes)
        
        return proposed_boxes
    
    def get_ground_truth(self, image_path, file_path='cropped_image_path', coord='crop_coordinates_ratio'):
        with open(self.json_file, 'r') as json_file:
            data = json.load(json_file)
        
        for item in data:
            if item[file_path] == image_path:
                image = cv2.imread(image_path)
                height, width = image.shape[:2]
                bounding_box = item[coord]
                
                ground_truth = []
                for bb in bounding_box:
                    x, y, w, h = bb[0], bb[1], bb[2], bb[3]
                    
                    if isinstance(x, int) and isinstance(y, int) and isinstance(w, int) and isinstance(h, int):
                        x1 = x
                        y1 = y
                        x2 = w
                        y2 = h   
                        ground_truth.append((x1, y1, x2, y2))
                    
                    else:
                        x1 = int((x - w / 2) * width)
                        y1 = int((y - h / 2) * height)
                        x2 = int((x + w / 2) * width)
                        y2 = int((y + h / 2) * height)
                        
                        # Perform boundary checks
                        x1 = max(0, min(x1, width - 1))
                        y1 = max(0, min(y1, height - 1))
                        x2 = max(0, min(x2, width - 1))
                        y2 = max(0, min(y2, height - 1))
                        
                        ground_truth.append((x1, y1, x2, y2))
        
        return ground_truth
    
    
    def get_cams(self, image_results, option=None):
        labels = [image_results[f'scale{scale}']['label'] for scale in [0.25, 0.5, 1]]
        cams = [image_results[f'scale{scale}']['cam'] for scale in [0.25, 0.5, 1]]
        attention_map = image_results['vit']['attention_map']
        
        defective_cams = [cam for label, cam in zip(labels, cams) if label == 'defective']
        
        "The following code is used to return the CAM image at SINGLE different scales"
        # Return the CAM image at scale 0.25
        if option == 'scale0.25':
            if 'defective' in labels:
                cam = cams[0]
            else :
                cam = None
            return cam
        
        # Return the CAM image at scale 0.5
        if option is None:
            if 'defective' in labels:
                cam = cams[1]
            else :
                cam = None
            return cam
        
        # Return the CAM image at scale 1.0
        if option == 'scale1':
            if 'defective' in labels:
                cam = cams[2]
            else :
                cam = None
            return cam
        
        "The following code is used to return the CAM image at multiple scales"
        if len(defective_cams) >= 2 or all(label == 'defective' for label in labels):
            if any(np.array_equal(cams[1], cam) for cam in defective_cams):
                weights = [2.0 if np.array_equal(cams[1], cam) else 0.5 for cam in defective_cams]
            else:
                weights = None
                
            _, multi_cam, _, _ = self.combine_cam_images(*defective_cams, weights=weights)
            
            # Return the fused cam image at scale 0.25 and 0.5
            if option == 'scale0.25_0.5':
                fused_cam = self.combine_cam_images(cams[0], cams[1])
                return fused_cam[1]
            
            # Return the fused cam image at 3 scales
            if option in ['multiscale', 'multi_slic', 'multi_watershed', 'multi_selectivesearch']:
                return im2double(multi_cam)
            
            # Return the only attention map
            if option == 'attention':
                return attention_map
            
            # Return the fused cam image at 3 scales with attention map
            if option == 'multi_attention':
                multi_cam = im2double(multi_cam)
                defective_cams_with_attention = self.combine_cam_images(multi_cam, attention_map)
                fused_cam = defective_cams_with_attention[1]
                return fused_cam
            
        else:
            return None
    
    
    def get_iou(self, image, image_path, fused_cam, threshold, file_path, coord, has_proposed_boxes=False, segmentation_method=None, n_segments=None, topN=None):
        if has_proposed_boxes:
            if segmentation_method == 'multi_slic':
                proposed_boxes = self.get_fusedcam_slic(image_path, fused_cam, n_segments)
            elif segmentation_method == 'multi_watershed':
                proposed_boxes = self.get_fusedcam_watershed(fused_cam)
            elif segmentation_method == 'multi_selectivesearch':
                proposed_boxes = self.get_fusedcam_selectivesearch(image_path, fused_cam, topN)
            else:
                raise ValueError("Invalid segmentation_method specified.")
        
        else:  
            proposed_boxes = self.localize_defective(image, fused_cam, threshold)
        
        gt_boxes = self.get_ground_truth(image_path, file_path, coord)
        
        # Calculating mean IoU per image
        mean_iou = calculate_mean_iou(proposed_boxes, gt_boxes)
        
        return mean_iou, proposed_boxes, gt_boxes
    
    
    def evaluate_all(self, image_dir, result_dict=None,
                     file_path='cropped_image_path', coord='crop_coordinates_ratio',
                     option=None, img_name=None):
        ious = []
        ious_dict = {}
        
        if result_dict is None:
            results = self.predict_scales(file_path=file_path)
        else:
            results = pd.read_pickle(result_dict)
        
        progress_bar = tqdm(total=len(results), desc='Processing Images')
        start_all = time.time()
        
        for image_name, image_results in results.items():
            start = time.time()
            
            if img_name is not None:
                image_name = img_name
                image_results = results[image_name]
                
            image_path = os.path.join(image_dir, f'{image_name}.jpg')
            image = cv2.imread(image_path)[:, :, ::-1]
            fused_cam = self.get_cams(image_results, option=option)
            
            if fused_cam is not None:
                # Calculate mean IoU per image
                if option in ['multi_slic', 'multi_watershed', 'multi_selectivesearch']:
                    iou, proposed_boxes, gt_boxes = self.get_iou(image, image_path, fused_cam,
                                                                 self.threshold, file_path, coord,
                                                                 has_proposed_boxes=True,
                                                                 segmentation_method=option,
                                                                 n_segments=800,
                                                                 topN=400)
                    
                else:
                    iou, proposed_boxes, gt_boxes = self.get_iou(image, image_path, fused_cam, self.threshold, file_path, coord)
                    
                # Save IoU score, predicted boxes and ground truth for each image
                image_results = {'IoU': iou,
                                'proposed_boxes': proposed_boxes,
                                'ground_truth': gt_boxes}
                
                ious_dict[image_name] = image_results
        
                ious.append(iou)
                    
                end = time.time()
                elapsed_time = end - start
                progress_bar.set_postfix({'Time': f'{elapsed_time:.2f}s'})
                progress_bar.update(1)
            
                print(f'Processed {image_name} in {end - start:.2f} seconds.')
        
        # Calculate the overall IoU score
        overall_iou = np.mean(ious)
        
        # Save the results
        progress_bar.close()
        end_all  = time.time()
        process_time = (end_all - start_all) / 60

        print(f'Processed {len(results)} images in {process_time:.2f} minutes.')
        
        return overall_iou, ious_dict
    
    
def proposed_approach():
    # Initialize the class 
    threshold = 0.5
    localizer = CombineCam(threshold=threshold, 
                           checkpoint_path='./results/tb_logs/lightning_logs/version_45/checkpoints/best_model_012-0.1648-0.94.ckpt',
                           json_file='./ground-truth-bbox/inference_modified_original.json')
    
    # List of available options
    options = {1: None,  # scale at the MR image size (512)
               2: 'scale0.25',
               3: 'scale1',
               4: 'scale0.25_0.5',
               5: 'attention',
               6: 'multiscale',
               7: 'multi_attention',
               8: 'multi_slic',
               9: 'multi_watershed',
               10: 'multi_selectivesearch'}
    
    # Evaluate the proposed approach by using different options with IoU score
    avg_iou, _ = localizer.evaluate_all(image_dir='/root/data/apple/cropped-apple-bb/images/',
                                        result_dict='./multiscale-dict/with-vit/multiscale_vit_version2.pickle',
                                        file_path='cropped_image_path',
                                        coord='crop_coordinates_ratio',
                                        option=options[7],
                                        img_name=None)
    
    print(avg_iou)
    print(f'Average IoU score: {avg_iou:.4f}')

if __name__ == '__main__':
    proposed_approach()
    
    