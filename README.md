# Modified YOLOv9

``` shell
# preprocess and prepare dataset according to YOLO input format having both bbox and segmentation values
!python3 preprocess.py

# command to train instance segmentation + object detection (modified DualDSegment) on our custom dataset
!python yolov9/train_dual.py --batch 8 --data final_data/dataset.yaml --img 640 --cfg yolov9/models/segment/yolov9-c-dseg.yaml --weights '' --name yolov9-c-dseg-custom --hyp yolov9/data/hyps/hyp.scratch-high.yaml --min-items 0 --epochs 1 --close-mosaic 15```
