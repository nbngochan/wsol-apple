# wsol-apple
This project explores weakly supervised object localization techniques on a class-label dataset of apple images. 

**Table of Contents**
- **About**
- **Getting Started**
- **Usage**


### **About**
This work utilizes a convolutional neural network (CNN) pretrained on image-level labels to generate class activation maps in a multi-scale manner to highlight discriminative regions. Additionally, a vision transformer (ViT) pretrained was treated to produce multi-head attention maps as an auxiliary detector. By integrating the CNN based CAMs and attention maps, our approach localizes defective regions without requiring bounding box or pixel-level supervision during training.

![Proposed method: fused map (MS-CA) obtained by fusing a multiscale CAM (MS-C) with an attention map obtained by crossing one of the multi-scale inputs through an auxiliary branch](/assets/images/main_figure.png)


### **Getting Started**
All the dependencies and required libraries are included in the *requirements.text* file

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

**Custom dataset**

The dataset used in this project is the Surface Defective Apple (SDA) dataset involved taking high-resolution images of apples with an industrial-grade camera. It contains 12,000 2448x2048 color images split into 6 classes (normal, physiological, scratch, malformation, blight, and others) with 6000 images per class. The dataset is split into 50,000 training images and 10,000 test images.

| Tables        | Class-label           | Instance-label  |
| ------------- |:-------------:| -----:|
| Normal      | right-aligned | $1600 |
| Defective      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |


Links to download the datasets are provided in *assets/data_sda.txt.*


### **Usage**
[**Object Detection Model**](https://github.com/Ka0Ri/Pytorch-pretrained-models)

**Object Classification Model**

**Localization**
- fd
- gf
- fgah






