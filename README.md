# PyTorch-Simple-MaskRCNN_Jetson_Nano
Modified open source implementation of Mask RCNN for Edge Devices

•	Backbone: Replace ResNet with MobileNet for lighter computation

•	Convolutional Layers: Use depthwise separable convolutions to reduce computation and memory usage

•	Batch Normalization: Retain for faster inference, assuming larger batch sizes

•	ReLU Activation: Keep as is due to its efficiency

•	RoIAlign: Maintain current implementation for accuracy

•	Softmax & Sigmoid Activations: Kept as is for their respective tasks

•	Box Head: 
    o	Option 1: Replace fully connected layers with 1x1 convolutions
    o	Option 2: Use Global Average Pooling (GAP) with a single fully connected layer
    
•	Anchor Box Generation: Utilized existing caching mechanism in RPN

•	Mask Head: 
    o	Replace 3x3 convolutions with depthwise separable convolutions
    o	Substitute transposed convolution with bilinear interpolation


