# FSM-INT-2022
**INTP2022-ML-3**

The entire project has been carried out in two different parts.
- Standalone real-time PCB Fault Detection using OpenCV
- Online Web-application to facilitate sequential analysis

Phases of the Project:

1. Understanding data and Exploratory Data Analysis
2. Model Building and Training
3. Model Hyperparameter tuning
4. Model comparison and testing
5. Model Deployment

Let us go through them one-by-one

## 1. Understanding the dataset and Exploratory Data Analysis (EDA)

### About the Dataset
The collection comprises 1,500 picture pairs, each consisting of a template image devoid of defects and a tested image that has been aligned and annotated with the positions of the six most prevalent PCB defects: open, short, mouse bite, spur, pinhole, and spurious copper.

### About the image
The linear scan CCD used to capture each image in this collection has a resolution of about 48 pixels per millimeter. The defect-free template pictures are manually examined and cleaned from sampling photos. The original size of the template and the tested image is around 16k x 16k pixels. They are then divided into several 640 × 640 pixels sub-images using a cropping process, then aligned using template matching methods. However, the 1500 defective PCB images and their annotation files are going to be primary resources for training our Deep Learning Model.

### About the image annotation
For each flaw in the tested photos, we utilize an axis-aligned bounding box with a class ID. Each annotated image owns an annotation file with the same filename, e.g.00041000_test.jpg, 00041000_temp.jpg and 00041000.txt are the tested image, template image and the corresponding annotation file. Each defect on the tested image is annotated as the format: **x1, y1, x2, y2, type**, where *(x1,y1)* and *(x2,y2)* are the *top-left* and the *bottom-right* corner of the bounding box of the defect and *type* is an *integer ID* that follows the matches: 0-background (not used), 1-open, 2-short, 3-mouse-bite, 4-spur, 5-copper, 6-pin-hole.


## 2. Model Building and Training

## Approach
As the project has been divided into two independent entities, that is, real-time-offline and online-deployed methods, both of them were required to have different approaches as well.

## Documentation is still in progress

Deployment
```
http://0.0.0.0:8000/docs#/
```

### Local Deployment and real-time fault using OpenCV
Run the following command in Linux/Ubuntu (after installing requirements)
```
python3 main.py
```


```
pip install pip-tools
pip-compile requirements.in
```

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Hlkl7Vz8Quk/0.jpg)](https://www.youtube.com/watch?v=Hlkl7Vz8Quk)

### Reference Articles
- https://www.analyticsvidhya.com/blog/2021/12/how-to-use-yolo-v5-object-detection-algorithm-for-custom-object-detection-an-example-use-case/
- https://blog.roboflow.com/yolov5-improvements-and-evaluation/
- https://medium.com/analytics-vidhya/object-detection-algorithm-yolo-v5-architecture-89e0a35472ef
- https://arxiv.org/abs/1505.00853 | https://arxiv.org/abs/1908.08681
- 