# DeepFashion Instance Segmentation with YOLOv9

## Overview
This project involves training a YOLOv9 model for clothes detection and instance segmentation on the DeepFashion dataset. The primary goal was to modify the YOLOv9 model to simultaneously predict detection bounding boxes and instance segmentation masks, and to analyze their performance.

## Dataset
A subset of the DeepFashion dataset containing 500 images was created, with a distribution of 70% for training, 20% for validation, and 10% for testing. Only classes with more than 50 images across the three splits were included. The dataset was converted into YOLO format to facilitate training. The code can be found in preprocess.py.

## Model Modifications
- **Architecture**: The YOLOv9-c model was fine-tuned to support both bounding box predictions and instance segmentation.
- **Training Script**: The training script was adjusted to accept a single YAML configuration file containing all necessary arguments and hyperparameters, streamlining the training process.

## Training
The model was trained using the following setup:
- **Initial Weights**: Pretrained MS COCO weights were used to initialize the model.
- **Hardware**: Training was conducted on Colab's T4 GPU.

## Evaluation and Visualization
Performance metrics for detection and instance segmentation were computed on both validation and test sets using MSCOCO evaluation metrics. These metrics helped in understanding the effectiveness of the model and provided insights into potential improvements. The following 3 strategies can futher improve the MSCOCO evaluation metrics like Average Precision and Average Recall:-

1. **Hyperparameter Tuning**
Adjusting confidence thresholds, IoU thresholds for non-max suppression, and other model parameters can refine predictions.
2. **Training on More Data**
More data, or more representative data, can improve the model's accuracy and robustness.
3. **Ensemble Methods**
Combining predictions from multiple models can often yield better results than any single model.

Detection bounding boxes and instance segmentation masks were visualized using Jupyter Notebook, offering a clear view of the model's predictions on test images. The results can be found in visualize.ipynb notebook.

## Performance Analysis
The project included a detailed analysis of the model performance, discussing how the metrics can be interpreted and suggesting potential areas for model enhancement.

## Code Quality
The preprocess.py script was documented and checked using PyLint to ensure code quality adheres to best practices.

## Conclusion
This project demonstrated the capability of YOLOv9 for instance segmentation tasks in fashion, highlighting the flexibility and efficiency of the model in handling complex datasets like DeepFashion.