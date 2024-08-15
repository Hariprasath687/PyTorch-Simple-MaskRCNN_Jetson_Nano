import math
import torch
from .utils import roi_align

class RoIAlign:
    """
    Performs Region of Interest (RoI) Align operator described in Mask R-CNN
    """
    
    def __init__(self, output_size, sampling_ratio):
        """
        Arguments:
            output_size (Tuple[int, int]): the size of the output after the cropping
                is performed, as (height, width)
            sampling_ratio (int): number of sampling points in the interpolation grid
                used to compute the output value of each pooled output bin. If > 0,
                then exactly sampling_ratio x sampling_ratio grid points are used. If
                <= 0, then an adaptive number of grid points are used (computed as
                ceil(roi_width / pooled_w), and likewise for height). Default: -1
        """
        
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.spatial_scales = None  # Adjusted to handle multiple scales
        
    def setup_scales(self, features, image_shape):
        if self.spatial_scales is not None:
            return
        
        self.spatial_scales = []
        for feature in features.values():
            feature_shape = feature.shape[-2:]
            possible_scales = []
            for s1, s2 in zip(feature_shape, image_shape):
                scale = 2 ** int(math.log2(s1 / s2))
                possible_scales.append(scale)
            assert possible_scales[0] == possible_scales[1]
            self.spatial_scales.append(possible_scales[0])
        
    def __call__(self, features, proposal, image_shape):
        """
        Arguments:
            features (OrderedDict[Tensor]): feature maps from different levels
            proposal (Tensor[K, 4]): proposals for the image
            image_shape (Torch.Size([H, W]))

        Returns:
            output (Tensor[K, C, self.output_size[0], self.output_size[1]])
        """
        idx = proposal.new_full((proposal.shape[0], 1), 0)
        roi = torch.cat((idx, proposal), dim=1)
        
        # Setup scales for each feature map
        self.setup_scales(features, image_shape)
        
        pooled_output = []
        for feature, spatial_scale in zip(features.values(), self.spatial_scales):
            pooled_output.append(roi_align(feature.to(roi), roi, spatial_scale, 
                                           self.output_size[0], self.output_size[1], 
                                           self.sampling_ratio))
        
        # Aggregate the pooled outputs, typically by summing or averaging
        output = torch.sum(torch.stack(pooled_output), dim=0)
        return output