# Modified YOLOv9

``` shell
# preprocess and prepare dataset according to YOLO input format having both bbox and segmentation values
!python3 preprocess.py

# command to train instance segmentation + object detection (modified DualDSegment) on our custom dataset
!python yolov9/train_dual.py --batch 8 --data final_data/dataset.yaml  
--cfg yolov9/models/segment/gelan-c-dseg.yaml --epochs 1 --img 640
--weights 'https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c-seg.pt' 
--name gelan-c-dseg.yaml-custom --hyp yolov9/data/hyps/hyp.scratch-high.yaml ```
