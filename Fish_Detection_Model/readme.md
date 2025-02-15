## Overview
This project implements an underwater fish detection and counting system using YOLOv10. It processes and combines three datasets (Fish4Knowledge, PANGEA, and DeepFish) for robust fish detection and counting 

## Dataset Setup
Due to size constraints, raw datasets are not included. 

1. Download datasets:
   - Fish4Knowledge: https://universe.roboflow.com/g18l5754/fish4knowledge-dataset/dataset/3/download/yolov9
   - PANGEA: https://doi.pangaea.de/10.1594/PANGAEA.926930

2. Place downloaded files in Raw Dataset to perform preprocessing

## Project Structure
```
D3-GROUP1/
├── data/
│   ├── train/          
│   ├── valid/          
│   ├── test/           
│   └── data.yaml       
├── RawDatasets/
│   ├── Fish4Knowledge Dataset/
│   │   ├── Images_Fish_4_Knowledge/
│   │   └── Labels_Fish_4_Knowledge/
│   └── Pangea/
│       └── luderick_seagrass_all.csv ()
├── Results/
├── best.pt
├── main.py
├── DeepFishPreprocess.ipynb
├── transformations.ipynb
└── yolov10m.pt

```
## Dataset Processing

# transformations.ipynb:

Removes timestamps from Fish4Knowledge images
Removes timestamps from Pangea dataset
Converts Pangea CSV to YOLO format
Converts Fish4Knowledge labels to single class

# DeepFishPreprocess.ipynb:
Data Exploration
Mask → Bounding box conversion
Thresholding & filtering
Initial Model Testing

# Dataset Split (Roboflow):

Combined with DeepFish dataset
Split into train/test/val
Generated data.yaml

## Dataset Configuration
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['0']

roboflow:
  workspace: joan-lleo-jncmc
  project: data-challenge-3
  version: 11
  license: CC BY 4.0
  url: https://universe.roboflow.com/joan-lleo-jncmc/data-challenge-3/dataset/11
```

## Validation and Inference
```python
# Validation
model = YOLO('best.pt')
results = model.val(data='data.yaml', split='test')

# Inference
results = model.predict(
    source='/images',
    save=True,
    save_txt=True,
    save_conf=True,
    conf=0.25,
    iou=0.7
)
```
## Dataset Access
- Dataset: [Roboflow Universe](https://universe.roboflow.com/joan-lleo-jncmc/data-challenge-3/dataset/11)
- Version: 11
- License: CC BY 4.0


## Acknowledgments
- Fish4Knowledge Dataset
- PANGEA Dataset
- DeepFish Dataset
- Roboflow for dataset management
- Ultralytics YOLOv10