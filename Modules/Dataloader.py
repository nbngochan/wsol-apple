from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torchvision.transforms._presets import ObjectDetection
from functools import partial
import numpy as np
import os
import json
import PIL.Image as Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


def normalize_image(image):
    xmin = np.min(image)
    xmax = np.max(image)
    return (image - xmin)/(xmax - xmin + 10e-6)

def collate_fn(batch):
    return tuple(zip(*batch))

class Standardize(object):
    """ Standardizes a 'PIL Image' such that each channel
        gets zero mean and unit variance. """
    def __call__(self, img):
        return (img - img.mean(dim=(1,2), keepdim=True)) \
            / torch.clamp(img.std(dim=(1,2), keepdim=True), min=1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class base_dataset(Dataset):
    '''
    mode: 'train', 'val', 'test'
    data_path: path to data folder
    imgsize: size of image
    transform: transform function
    '''
    def __init__(self, mode, data_path, imgsize=224, transform=None):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.img_list = None
        self.label_list = None

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        '''
        return tran_image, target, original image
        '''
        image = self.img_list[index]
        label = self.label_list[index]
        trans_img = self.transform(image)
        return trans_img, label, image
      
       
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, mode, data_path, imgsize=224, transform=None):
        self.root = data_path
        self.transform = transform
        # load all image files, sorting them to
        # ensure that they are aligned
        imgs = list(sorted(os.listdir(os.path.join(data_path, "PNGImages"))))
        masks = list(sorted(os.listdir(os.path.join(data_path, "PedMasks"))))

        n = len(imgs)
        if(mode == 'train'):
            self.imgs = imgs[:int(n*0.8)]
            self.masks = masks[:int(n*0.8)]
        elif(mode == 'val'):
            self.imgs = imgs[int(n*0.8):]
            self.masks = masks[int(n*0.8):]
        elif(mode == 'test'):
            self.imgs = imgs[int(n*0.8):]
            self.masks = masks[int(n*0.8):]

        if(self.transform is None):
            self.transform = partial(ObjectDetection)()
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        trans_img = self.transform(img)

        return trans_img, target, transforms.ToTensor()(img)
       
class AppleRead(Dataset):
    def __init__(self, mode, data_path, imgsize=224, transform=None):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        # read json file
        with open(os.path.join(data_path, 'inference_modified_2106355.json'), 'r') as f:
            json_data = json.load(f)

        # remove images with class 1
        json_data = [x for x in json_data if x['class'] != 1]

        n = len(json_data)
        if(mode == 'train'):
            self.dataset = json_data[:int(n*0.8)]
        elif(mode == 'val'):
            self.dataset = json_data[int(n*0.8):]
        elif(mode == 'test'):
            self.dataset = json_data[int(n*0.8):]

        if(self.transform is None):
            self.transform = partial(ObjectDetection)()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):

        img_path = os.path.join(self.data_path, 'images', self.dataset[index]['name'])
        img = Image.open(img_path).convert("RGB")

        # process labels
        temp_boxes = self.dataset[index]['crop_coordinates_ratio']
        num_objs = len(temp_boxes)
        boxes = []
        # convert from [x_center, y_center, width, height] to [xmin, ymin, xmax, ymax]
        # and convert from ratio to absolute value
        for box in temp_boxes:
            x_center, y_center, width, height = box
            xmin = int((x_center - width/2) * img.size[0])
            xmax = int((x_center + width/2) * img.size[0])
            ymin = int((y_center - height/2) * img.size[1])
            ymax = int((y_center + height/2) * img.size[1])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64) # all labels are 1
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        trans_img = self.transform(img)

        return trans_img, target, transforms.ToTensor()(img)
    
class DataModule(LightningDataModule):
    '''
    Data Module for Train/Val/Test data loadding
    Args: 
        data_settings, training_settings: hyperparameter settings
        transform: data augmentation
    Returns:
        Train/Test/Val data loader
    '''
    def __init__(self, data_settings, training_settings, transform=[None, None]):
        super().__init__()

        self.dataset = data_settings['name']
        self.root_dir = data_settings['path']
        self.img_size = data_settings['img_size']
        self.batch_size = training_settings['n_batch']
        self.num_workers = training_settings['num_workers']

        self.data_class = {
            "PennFudan": PennFudanDataset,
            "Apple": AppleRead
        }
        self.class_list = None
        self.collate_fn = None
        if(self.dataset in ["Apple", "PennFudan"]):
            self.collate_fn = collate_fn

        self.train_transform, self.val_transform = transform
        
    def setup(self, stage: str):

        if stage == "fit":
            self.Train_dataset = self.data_class[self.dataset](mode="train", data_path=self.root_dir,
                                                imgsize=self.img_size, transform=self.train_transform)
            self.Val_dataset = self.data_class[self.dataset](mode="val", data_path=self.root_dir,
                                                imgsize=self.img_size, transform=self.val_transform)
                
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.Test_dataset = self.data_class[self.dataset](mode="test", data_path=self.root_dir,
                                                imgsize=self.img_size, transform=self.val_transform)
           
    def train_dataloader(self):
        return DataLoader(self.Train_dataset, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.Val_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.Test_dataset, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, collate_fn=self.collate_fn)
