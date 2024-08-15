#!/bin/bash


dataset="voc"
iters=1

if [ $dataset = "voc" ]
then
    data_dir="/Users/hariprasath/Documents/PyTorch-Simple-MaskRCNN/data/voc2012/VOCdevkit/VOC2012/"
elif [ $dataset = "coco" ]
then
    data_dir="/data/coco2017/"
fi


python train.py --iters ${iters} --dataset ${dataset} --data-dir ${data_dir}

