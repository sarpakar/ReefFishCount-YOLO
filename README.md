# CoralFishNet-YOLO

Real-time fish detection and counting system utilizing YOLOv10 for underwater environments, combining multiple marine datasets for robust detection.

## Project Overview
This repository implements an automated fish counting system designed for underwater environments. The system leverages YOLOv10 architecture and combines three major datasets (Fish4Knowledge, PANGEA, and DeepFish) for comprehensive fish detection and counting capabilities.

## Requirements
```python
python >= 3.8
torch >= 1.7.0
opencv-python >= 4.5.4
ultralytics >= 10.0.0
numpy >= 1.19.5
roboflow
```

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
│       └── luderick_seagrass_all.csv
├── Results/
├── best.pt
├── main.py
├── DeepFishPreprocess.ipynb
├── transformations.ipynb
└── yolov10m.pt
```

## Dataset Setup

### Data Sources
1. **Fish4Knowledge Dataset**
   - Download: https://universe.roboflow.com/g18l5754/fish4knowledge-dataset/dataset/3/download/yolov9
   - Primary source for fish detection

2. **PANGEA Dataset**
   - Access: https://doi.pangaea.de/10.1594/PANGAEA.926930
   - Provides additional environmental context

3. **DeepFish Dataset**
   - Supplementary data for model robustness

### Dataset Configuration
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

## Methodology

### Data Preprocessing
Two main preprocessing notebooks:

1. **transformations.ipynb**
   - Timestamp removal from Fish4Knowledge images
   - Timestamp removal from Pangea dataset
   - Conversion of Pangea CSV to YOLO format
   - Label consolidation to single class

2. **DeepFishPreprocess.ipynb**
   - Data exploration
   - Mask to bounding box conversion
   - Thresholding and filtering
   - Initial model testing

### Dataset Processing Pipeline
1. Raw data collection from three sources
2. Preprocessing using provided notebooks
3. Combination with DeepFish dataset
4. Split into train/test/val using Roboflow
5. Generation of data.yaml configuration

### Model Architecture
- Base: YOLOv10
- Custom modifications for underwater conditions
- Single class detection optimization

## Usage

### Training
```bash
python train.py --data data.yaml --epochs 100 --batch 16
```

### Validation
```python
model = YOLO('best.pt')
results = model.val(data='data.yaml', split='test')
```

### Inference
```python
results = model.predict(
    source='/images',
    save=True,
    save_txt=True,
    save_conf=True,
    conf=0.25,
    iou=0.7
)
```

## Results
- Validation metrics available in Results/ directory
- Performance analysis across different datasets
- Inference speed benchmarks

## Future Work
- Integration with real-time monitoring systems
- Multi-class fish species detection
- Environmental condition adaptation


## Citations
```bibtex
@dataset{fish4knowledge2023,
    title={Fish4Knowledge Dataset},
    year={2023},
    url={https://universe.roboflow.com/g18l5754/fish4knowledge-dataset}
}

@dataset{pangea2021,
    title={Luderick seagrass dataset},
    year={2021},
    doi={10.1594/PANGAEA.926930}
}
```

## License
This project is licensed under CC BY 4.0

## Acknowledgments
- Fish4Knowledge Dataset contributors
- PANGEA Dataset maintainers
- DeepFish Dataset team
- Roboflow for dataset management
