# wsol-apple
>This project explores weakly supervised object localization techniques on a class-label dataset of apple images. 

**Table of Contents**
- **About**
- **Getting Started**
- **Usage**
- **Localization Visualization**

## **About**
This work utilizes a convolutional neural network (CNN) pretrained on image-level labels to generate class activation maps in a multi-scale manner to highlight discriminative regions. Additionally, a vision transformer (ViT) pretrained was treated to produce multi-head attention maps as an auxiliary detector. By integrating the CNN based CAMs and attention maps, our approach localizes defective regions without requiring bounding box or pixel-level supervision during training.

![Proposed method: fused map (MS-CA) obtained by fusing a multiscale CAM (MS-C) with an attention map obtained by crossing one of the multi-scale inputs through an auxiliary branch](/assets/images/main_figure.png)


## **Getting Started**
All the dependencies and required libraries are included in the *requirements.txt* file

**Prequisite**
- Python=3.10
- CUDA: 11
- Pytorch framework
- Others: numpy, pandas, scipy, open-cv
- Dashboard: tensorboard, neptune AI, gradio


**Installation**
1. Clone the repo
```
$ git clone https://github.com/nbngochan/wsol-apple.git
```

2. Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html), change your directory to the clone repo and create new conda Python virtual enviroment *'myenv'*
```
$ conda create -n ENV_NAME python=3.10
$ conda activate ENV_NAME
```

3. Run the followibg command in your Terminal/Command Prompt to install the libraries required
 ```
 $ conda install pytorch==2.0.9 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.7 -c pytorch
 $ pip install -r requirements.txt
 ```

Or install through **Docker:**
```
docker-compose build
docker-compose run --rm dev
```
Please see [Dockerfile](/Dockerfile) and [requirements.txt](/requirements.txt)


**Custom dataset**

The dataset used in this project is the Surface Defective Apple (SDA) dataset involved taking high-resolution images of apples with an industrial-grade camera from an orchard in Jangseong-gun, South Korea. It contains 12,000 2448x2048 color images split into 6 classes (normal, physiological, scratch, malformation, blight, and others). The dataset is split into 2 small datasets: 
- (1st) Class-label dataset: use for classifier training, only binary label.
- (2nd) Instance-label dataset: use for evaluate the localization process.


| Tables        | Class-label           | Instance-label  |
| ------------- |:-------------:| -----:|
| Normal      | 6,201  | 255 |
| Defective      | 3,799      |   823 |
| Total | 10,000       |    1,078 |


Links to download the datasets are provided in *assets/data_sda.txt.*


## **Usage**
### [**Object Detection Model**](https://github.com/Ka0Ri/Pytorch-pretrained-models)


### **Weakly Supvervised Defects Localization**
**PIPEPLINE**

TODO

**_Object Classification Module_**
Implements image classification for a SDA dataset based on [PyTorch Lightning]() framework and [timm](). The model used for classifier training:
- ResNet50
- EfficientNet-v2
- GoogLeNet-v4

**1. Data Preparation**
Organize the original data to be structured as:

```
dataset/
  - train/
    - train_images/
      - img1.jpg
      - img2.jpg
      ...
    - train.csv
  - test/  
    - test_images/
      - img1.jpg
      - img2.jpg
      ...
    - test.csv
```
The dataset contains CSV files for the training and test sets - <span style='color: red;'>train.csv</span> and <span style='color: red;'>test.csv</span>. These CSV files contain the following columns:
- name - The image file name
- label - The class label (0 for normal, 1 for defective)
- state - The class name (normal or defective)
- file_path - The path to the segmented image file
- file_raw_path - The path to the original raw image file

Below is an example of what the CSV file looks like:
```
name,label,state,file_path,file_raw_path
8501,1,defective,mnt/dataset/defective-fruit/test/test_segmented/8501_segmented.jpg,mnt/dataset/defective-fruit/test/test/8501.jpg
2878,0,normal,mnt/dataset/defective-fruit/test/test_segmented/2878_segmented.jpg,mnt/dataset/defective-fruit/test/test/2878.jpg 
```

**2. Training classification**
```
python train.py --csv-train /root/data/apple/cropped-apple-bb/defective-fruit/train.csv --csv-val /root/data/apple/cropped-apple-bb/defective-fruit/test.csv
```
You can use the most of models in [timm](https://huggingface.co/docs/timm/index) by specifying `--mode-name` directly
```
usage: train.py [-h] --dataset DATASET [--outdir OUTDIR]
                [--model-name MODEL_NAME] [--img-size IMG_SIZE]
                [--epochs EPOCHS] [--save-interval SAVE_INTERVAL]
                [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS]
                [--gpu-ids GPU_IDS [GPU_IDS ...] | --n-gpu N_GPU]
                [--seed SEED]

Train classifier.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET, -d DATASET
                        Root directory of dataset
  --outdir OUTDIR, -o OUTDIR
                        Output directory
  --model-name MODEL_NAME, -m MODEL_NAME
                        Model name (timm)
  --img-size IMG_SIZE, -i IMG_SIZE
                        Input size of image
  --epochs EPOCHS, -e EPOCHS
                        Number of training epochs
  --save-interval SAVE_INTERVAL, -s SAVE_INTERVAL
                        Save interval (epoch)
  --batch-size BATCH_SIZE, -b BATCH_SIZE
                        Batch size
  --num-workers NUM_WORKERS, -w NUM_WORKERS
                        Number of workers
  --gpu-ids GPU_IDS [GPU_IDS ...]
                        GPU IDs to use
  --n-gpu N_GPU         Number of GPUs
  --seed SEED           Seed
```

**Tensorboard Logging**
```
tensorboard --logdir ./results
```


**_Defects Localization Module_**

**1. Data Preprocessing**
This part using another type of dataset: instance-label so we need to perform 

**2. Localization phrase**


## **Localization Visualization**




