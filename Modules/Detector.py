import torch.nn as nn

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.ssd import SSDHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import _utils
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_resnet50_fpn, \
                                        retinanet_resnet50_fpn, retinanet_resnet50_fpn_v2, \
                                        ssd300_vgg16, ssdlite320_mobilenet_v3_large, \
                                        maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_MobileNet_V3_Large_FPN_Weights,\
                                        RetinaNet_ResNet50_FPN_Weights, RetinaNet_ResNet50_FPN_V2_Weights, \
                                        SSD300_VGG16_Weights, SSDLite320_MobileNet_V3_Large_Weights, \
                                        MaskRCNN_ResNet50_FPN_V2_Weights, MaskRCNN_ResNet50_FPN_Weights

MODEL = {
    "fasterrcnn-s": fasterrcnn_mobilenet_v3_large_fpn, "fasterrcnn-m": fasterrcnn_resnet50_fpn, "fasterrcnn-l": fasterrcnn_resnet50_fpn_v2,
    "retinanet-m": retinanet_resnet50_fpn, "retinanet-l": retinanet_resnet50_fpn_v2,
    "ssd-s": ssdlite320_mobilenet_v3_large, "ssd-m": ssd300_vgg16,

    "maskrcnn-s": maskrcnn_resnet50_fpn, "maskrcnn-m": maskrcnn_resnet50_fpn_v2
}

WEIGHTS = {
    "fasterrcnn-s": FasterRCNN_MobileNet_V3_Large_FPN_Weights, "fasterrcnn-m": FasterRCNN_ResNet50_FPN_Weights, "fasterrcnn-l": FasterRCNN_ResNet50_FPN_V2_Weights,
    "retinanet-m": RetinaNet_ResNet50_FPN_Weights, "retinanet-l": RetinaNet_ResNet50_FPN_V2_Weights,
    "ssd-s": SSDLite320_MobileNet_V3_Large_Weights, "ssd-m": SSD300_VGG16_Weights,

    "maskrcnn-s": MaskRCNN_ResNet50_FPN_Weights, "maskrcnn-m": MaskRCNN_ResNet50_FPN_V2_Weights
}


    
class WrappingDetector(nn.Module):
    '''
    Wrapping Detection model with customized head,
    The given models can be used in reference mode (is_full = True)
    with pretrained weights (is_pretrained = True)
    or fine tuning (is_freeze = False) with explicit the number of classes 

    - Supported models: fasterrcnn, ssd, retinanet [-option s: small, m: medium, l: large]
    Authors: dtvu1707@gmail.com
    '''
    def __init__(self, model_configs: dict) -> None:

        super(WrappingDetector, self).__init__()
        # Parse parameters
        self.backbone_name = model_configs['backbone']['name']
        self.is_full = model_configs['backbone']['is_full']
        self.is_pretrained = model_configs['backbone']['is_pretrained']
        self.is_freeze = model_configs['backbone']['is_freeze']
        self.n_cls = model_configs['n_cls']

        self._model_selection()
            
    def _model_selection(self):

        # Load pretrained model
        name = self.backbone_name
        assert name in MODEL.keys(), "Model %s not found" % name

        if self.is_pretrained:
            base_model = MODEL[name](weights=WEIGHTS[name].COCO_V1)
        else:
            base_model = MODEL[name](WEIGHTS=None)
        self.preprocess = WEIGHTS[name].COCO_V1.transforms()
        self.meta = WEIGHTS[name].COCO_V1.meta
        
        # turn off gradient
        if self.is_freeze:
            for param in base_model.parameters():
                param.requires_grad = False
        
        if not self.is_full:
            # Modify detection head
            if "fasterrcnn" in name:
                num_ftrs = base_model.roi_heads.box_predictor.cls_score.in_features
                base_model.roi_heads.box_predictor = FastRCNNPredictor(num_ftrs, self.n_cls)   
            elif "retinanet" in name:
                num_ftrs = base_model.backbone.out_channels
                num_anchors = base_model.anchor_generator.num_anchors_per_location()
                base_model.head = RetinaNetHead(num_ftrs, num_anchors[0], self.n_cls)
            elif "ssd" in name:
                num_ftrs = _utils.retrieve_out_channels(base_model.backbone, (320, 320))
                num_anchors = base_model.anchor_generator.num_anchors_per_location()
                base_model.head = SSDHead(num_ftrs, num_anchors, self.n_cls)
            elif "maskrcnn" in name:
                num_ftrs = base_model.roi_heads.box_predictor.cls_score.in_features
                base_model.roi_heads.box_predictor = FastRCNNPredictor(num_ftrs, self.n_cls)
                mask_num_ftrs = base_model.roi_heads.mask_predictor[0].in_channels
                base_model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_num_ftrs, 256, self.n_cls)
       
        self.base_model = base_model
    
    def forward(self, x, y=None):

        images = list(image for image in x)
        if y is not None:
            targets = [{k: v for k, v in t.items()} for t in y]
            output = self.base_model(images, targets)
        else:
            output = self.base_model(images)
        
        return output