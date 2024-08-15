from collections import OrderedDict

import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc

from .utils import AnchorGenerator
from .rpn import RPNHead, RegionProposalNetwork
from .pooler import RoIAlign
from .roi_heads import RoIHeads
from .transform import Transformer

from torchvision.models import mobilenet_v2
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.ops import misc
from .convolutions import DepthwiseSeparableConv

class MaskRCNN(nn.Module):
    """
    Implements Mask R-CNN.

    The input image to the model is expected to be a tensor, shape [C, H, W], and should be in 0-1 range.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensor, as well as a target (dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [xmin, ymin, xmax, ymax] format, with values
          between 0-H and 0-W
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

    The model returns a Dict[Tensor], containing the classification and regression losses 
    for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensor, and returns the post-processed
    predictions as a Dict[Tensor]. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [xmin, ymin, xmax, ymax] format, 
          with values between 0-H and 0-W
        - labels (Int64Tensor[N]): the predicted labels
        - scores (FloatTensor[N]): the scores for each prediction
        - masks (FloatTensor[N, H, W]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)
        
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
        num_classes (int): number of output classes of the model (including the background).
        
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_num_samples (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors during training of the RPN
        rpn_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_num_samples (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals during training of the 
            classification head
        box_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_num_detections (int): maximum number of detections, for all classes.
        
    """
    
    def __init__(self, backbone, num_classes,
                 # RPN parameters
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=256, rpn_positive_fraction=0.5,
                 rpn_reg_weights=(1., 1., 1., 1.),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 # RoIHeads parameters
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_num_samples=512, box_positive_fraction=0.25,
                 box_reg_weights=(10., 10., 5., 5.),
                 box_score_thresh=0.1, box_nms_thresh=0.6, box_num_detections=100):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels
        
        #------------- RPN --------------------------
        anchor_sizes = (128, 256, 512)
        anchor_ratios = (0.5, 1, 2)
        num_anchors = len(anchor_sizes) * len(anchor_ratios)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        rpn_head = RPNHead(out_channels, num_anchors)
        
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
             rpn_anchor_generator, rpn_head, 
             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
             rpn_num_samples, rpn_positive_fraction,
             rpn_reg_weights,
             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        
        #------------ RoIHeads --------------------------
        box_roi_pool = RoIAlign(output_size=(7, 7), sampling_ratio=2)
        
        resolution = box_roi_pool.output_size[0]
        # in_channels = out_channels * resolution ** 2
        in_channels = out_channels
        # mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, num_classes)
        
        self.head = RoIHeads(
             box_roi_pool, box_predictor,
             box_fg_iou_thresh, box_bg_iou_thresh,
             box_num_samples, box_positive_fraction,
             box_reg_weights,
             box_score_thresh, box_nms_thresh, box_num_detections)
        
        self.head.mask_roi_pool = RoIAlign(output_size=(14, 14), sampling_ratio=2)
        
        layers = (256, 256, 256, 256)
        dim_reduced = 256
        self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)
        
        #------------ Transformer --------------------------
        self.transformer = Transformer(
            min_size=800, max_size=1333, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225])
        
    def forward(self, image, target=None):
        ori_image_shape = image.shape[-2:]
        
        image, target = self.transformer(image, target)
        image_shape = image.shape[-2:]
        feature = self.backbone(image)
        
        proposal, rpn_losses = self.rpn(feature, image_shape, target)
        result, roi_losses = self.head(feature, proposal, image_shape, target)
        
        if self.training:
            return dict(**rpn_losses, **roi_losses)
        else:
            result = self.transformer.postprocess(result, image_shape, ori_image_shape)
            return result
        
# Global Average Pooling (GAP) + Single Fully Connected Layer Implementation instead of 2 FCs     
class FastRCNNPredictor(nn.Module):  # Box Head
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.fc = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)  # Box Predictor
        
    def forward(self, x):
        # print(f"Input shape before GAP: {x.shape}")
        x = self.gap(x)
        # print(f"Shape after GAP: {x.shape}")
        x = x.flatten(start_dim=1)  # Apply GAP and then flatten
        # print(f"Shape after flatten: {x.shape}") 
        score = self.fc(x)
        bbox_delta = self.bbox_pred(x)
        return score, bbox_delta

# Alternate Implementation: 1x1 Convolutions instead of 2 FCs

# class FastRCNNPredictor1x1(nn.Module):
#     def __init__(self, in_channels, mid_channels, num_classes):
#         super().__init__()
#         self.conv1x1_1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
#         self.conv1x1_2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
#         self.cls_score = nn.Conv2d(mid_channels, num_classes, kernel_size=1)
#         self.bbox_pred = nn.Conv2d(mid_channels, num_classes * 4, kernel_size=1)

#     def forward(self, x):
#         x = F.relu(self.conv1x1_1(x))
#         x = F.relu(self.conv1x1_2(x))
#         score = self.cls_score(x).view(x.size(0), -1)  # Flatten for final output
#         bbox_delta = self.bbox_pred(x).view(x.size(0), -1)

#         return score, bbox_delta
    
class MaskRCNNPredictor(nn.Sequential): # Mask Head
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        """
        Arguments:
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
        """
        
        d = OrderedDict()
        next_feature = in_channels
        
        # Replace standard Conv2d with Depthwise Separable Convolutions
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = DepthwiseSeparableConv(next_feature, layer_features, 3, 1, 1)
            next_feature = layer_features
        
        # Replace Transposed Convolution with Bilinear Interpolation followed by a 1x1 Conv
        d['bilinear_interp'] = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        d['conv_1x1'] = nn.Conv2d(next_feature, dim_reduced, kernel_size=1)
        d['bn_interp'] = nn.BatchNorm2d(dim_reduced)
        d['relu_interp'] = nn.ReLU(inplace=False)
        
        # Final 1x1 Convolution for mask prediction
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, kernel_size=1)
        
        super().__init__(d)

        # Initialize weights
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:  # Ensure the parameter has at least 2 dimensions
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)

class MobileNetBackboneWithFPN(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = mobilenet_v2(pretrained=False).features

        # Extract layers from MobileNet and replace conv layers with depthwise separable conv
        self.layer1 = nn.Sequential(
            DepthwiseSeparableConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False)
        )
        self.layer2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False)
        )
        self.layer3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False)
        )
        self.layer4 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=False)
        )

        # FPN construction
        out_channels = 256
        in_channels_list = [32, 64, 128, 256]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelMaxPool()
        )
        self.out_channels = out_channels

    def forward(self, x):
        # Forward through MobileNet layers
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        
        # FPN expects the feature maps to be ordered from the highest resolution to the lowest.
        # Ensuring that c1 (from layer1) has the highest resolution, and c4 (from layer4) has the lowest resolution.
        # print(f"c1: {c1.shape}, c2: {c2.shape}, c3: {c3.shape}, c4: {c4.shape}") 
        
        # Pass through FPN
        features = [c1, c2, c3, c4]
        x = self.fpn(OrderedDict([(f'c{i+1}', feat) for i, feat in enumerate(features)]))
        return x

    
# Replace your existing backbone function with this
def maskrcnn_mobilenet_fpn(num_classes):
    """
    Constructs a Mask R-CNN model with a MobileNet backbone and FPN.
    
    Arguments:
        num_classes (int): number of classes (including the background).
    """
    
    backbone = MobileNetBackboneWithFPN()
    model = MaskRCNN(backbone, num_classes)

    return model