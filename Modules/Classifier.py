import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet152, \
                                densenet121, densenet161, densenet201, \
                                efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s, \
                                vgg11_bn, vgg13_bn, vgg19_bn, \
                                wide_resnet50_2, wide_resnet101_2, \
                                inception_v3, \
                                mobilenet_v3_large, mobilenet_v3_small, \
                                shufflenet_v2_x0_5, shufflenet_v2_x2_0, shufflenet_v2_x1_0, \
                                convnext_tiny, convnext_base, convnext_large, \
                                resnext50_32x4d, resnext101_64x4d, resnext101_32x8d, \
                                vit_b_16, vit_l_16, vit_h_14, \
                                swin_t, swin_b, swin_s
                                
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet152_Weights, \
                                DenseNet121_Weights, DenseNet161_Weights, DenseNet201_Weights, \
                                EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights, \
                                VGG11_BN_Weights, VGG13_BN_Weights, VGG19_BN_Weights, \
                                Wide_ResNet101_2_Weights, Wide_ResNet50_2_Weights, \
                                Inception_V3_Weights, \
                                MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights, \
                                ShuffleNet_V2_X0_5_Weights, ShuffleNet_V2_X1_0_Weights, ShuffleNet_V2_X2_0_Weights, \
                                ConvNeXt_Tiny_Weights, ConvNeXt_Base_Weights, ConvNeXt_Large_Weights, \
                                ResNeXt101_32X8D_Weights, ResNeXt50_32X4D_Weights, ResNeXt101_64X4D_Weights, \
                                ViT_B_16_Weights, ViT_L_16_Weights, ViT_H_14_Weights, \
                                Swin_S_Weights, Swin_T_Weights, Swin_B_Weights
      


MODEL = {"resnet-s": resnet18, "resnet-m": resnet50, "resnet-l": resnet152,
        "densenet-s": densenet121, "densenet-m": densenet161, "densenet-l": densenet201,
        "efficientnet-s": efficientnet_v2_s, "efficientnet-m": efficientnet_v2_m, "efficientnet-l": efficientnet_v2_l,
        "vgg-s": vgg11_bn, "vgg-m": vgg13_bn, "vgg-l": vgg19_bn,
        "wideresnet-s": wide_resnet50_2, "wideresnet-m": wide_resnet101_2,
        "inception": inception_v3,
        "mobilenet-s": mobilenet_v3_small, "mobilenet-l": mobilenet_v3_large,
        "shufflenet-s": shufflenet_v2_x0_5, "shufflenet-m": shufflenet_v2_x1_0, "shufflenet-l": shufflenet_v2_x2_0,
        "convnext-s": convnext_tiny, "convnext-m": convnext_base, "convnext-l": convnext_large,
        "resnext-s": resnext50_32x4d, "resnext-m": resnext101_32x8d, "resnext-l": resnext101_64x4d,
        "vit-s": vit_b_16, "vit-m": vit_l_16, "vit-l": vit_h_14,
        "swin-s": swin_t, "swin-m": swin_s, "swin-l": swin_b
        }
WEIGHTS = {"resnet-s": ResNet18_Weights, "resnet-m": ResNet50_Weights, "resnet-l": ResNet152_Weights,
           "densenet-s": DenseNet121_Weights, "densenet-m": DenseNet161_Weights, "densenet-l": DenseNet201_Weights,
           "efficientnet-s": EfficientNet_V2_S_Weights, "efficientnet-m": EfficientNet_V2_M_Weights, "efficientnet-l": EfficientNet_V2_L_Weights,
           "vgg-s": VGG11_BN_Weights, "vgg-m": VGG13_BN_Weights, "vgg-l": VGG19_BN_Weights, \
           "wideresnet-s": Wide_ResNet50_2_Weights, "wideresnet-m":  Wide_ResNet101_2_Weights,
           "inception": Inception_V3_Weights,
           "mobilenet-s": MobileNet_V3_Small_Weights, "mobilenet-l": MobileNet_V3_Large_Weights,
           "shufflenet-s": ShuffleNet_V2_X0_5_Weights, "shufflenet-m": ShuffleNet_V2_X1_0_Weights, "shufflenet-l": ShuffleNet_V2_X2_0_Weights,
           "convnext-s": ConvNeXt_Tiny_Weights, "convnext-m": ConvNeXt_Base_Weights, "convnext-l": ConvNeXt_Large_Weights,
           "resnext-s": ResNeXt50_32X4D_Weights, "resnext-m": ResNeXt101_32X8D_Weights, "resnext-l": ResNeXt101_64X4D_Weights,
           "vit-s": ViT_B_16_Weights, "vit-m": ViT_L_16_Weights, "vit-l": ViT_H_14_Weights,
           "swin-s": Swin_T_Weights, "swin-m": Swin_S_Weights, "swin-l": Swin_B_Weights
        }

class WrappingClassifier(nn.Module):
    '''
    Wrapping Classification with customized headers
    The given models can be used in reference mode (is_full = True)
    with pretrained weights (is_pretrained = True)
    or fine tuning (is_freeze = False) with explicit the number of classes
    - Supported models: resnet, shuffle, inception, resnext, vgg, mobile, , 
                        vit, swin, wideresnet, dense, convnext, efficient
    [-option s: small, m: medium, l: large]
    Authors: dtvu1707@gmail.com
    '''
    def __init__(self, model_configs=None):

        super(WrappingClassifier, self).__init__()
        # Parse parameters
        self.backbone_name = model_configs['backbone']['name']
        self.is_full = model_configs['backbone']['is_full']
        self.is_pretrained = model_configs['backbone']['is_pretrained']
        self.is_freeze = model_configs['backbone']['is_freeze']
        self.n_cls = model_configs['n_cls']

        self._model_selection()
        
        if not self.is_full:
            self.classifier = nn.Linear(self.num_ftrs, self.n_cls)

    def _model_selection(self):

        # Load pretrained model
        name = self.backbone_name
        assert name in MODEL.keys(), "Model %s not found" % name

        if self.is_pretrained: #init with pretrained weights
            base_model = MODEL[name](weights=WEIGHTS[name].IMAGENET1K_V1)
        else:
            base_model = MODEL[name](weights=None)
        self.preprocess = WEIGHTS[name].IMAGENET1K_V1.transforms()
        self.meta = WEIGHTS[name].IMAGENET1K_V1.meta

        # turn off gradient
        if self.is_freeze:
            for param in base_model.parameters():
                param.requires_grad = False
        
        if not self.is_full:
            # Remove Classification head
            if any([x in name for x in ["resnet", "inception", "resnext", "wideresnet", "shufflenet"]]):
                if "inception" in name: 
                    base_model.aux_logits=False
                self.num_ftrs = base_model.fc.in_features
                base_model.fc = nn.Identity()
            elif any([x in name for x in ["vgg", "mobile", "efficient"]]):
                if "vgg" in name: 
                    self.num_ftrs = base_model.features[-4].out_channels
                else: 
                    self.num_ftrs = base_model.features[-1].out_channels
                base_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                base_model.classifier = nn.Identity()
            elif any([x in name for x in ["densenet", "convnext"]]):
                if "convnext" in name: 
                    self.num_ftrs = base_model.classifier[2].in_features
                else: 
                    self.num_ftrs = base_model.classifier.in_features
                base_model.classifier = nn.Identity()
            elif "vit" in name:
                self.num_ftrs = base_model.heads[-1].in_features
                base_model.heads = nn.Identity()
            elif "swin" in name:
                self.num_ftrs = base_model.head.in_features
                base_model.head = nn.Identity()

        self.backbone = base_model
    
    def forward(self, x, y=None):

        feats = self.backbone(x)
        if self.is_full:
            return feats
        else:
            feats = torch.flatten(feats, start_dim=1)
            return self.classifier(feats)