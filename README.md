# CoralFishNet-YOLO

Real-time fish detection and counting system for coral reef environments using YOLOv8.

## Project Overview
This repository contains the implementation of an automated fish counting system designed for coral reef monitoring. The system utilizes YOLOv8 architecture optimized for underwater environments, developed as part of the JBG060 Data Challenge 3 Course at TU/e Eindhoven collaborating with Fruitpunch.ai

## Requirements
```python
python >= 3.8
torch >= 1.7.0
opencv-python >= 4.5.4
ultralytics >= 8.0.0
numpy >= 1.19.5
```

## Installation

```bash
git clone https://github.com/sarpakar/CoralFishNet-YOLO.git
cd CoralFishNet-YOLO
pip install -r requirements.txt
```

## Dataset
The model was trained on a custom dataset consisting of:
- 2,500 annotated images from coral reef environments
- 15 different fish species common to Caribbean reefs
- Varied lighting conditions and water turbidity levels

Dataset structure:
```
data/
    ├── train/
    ├── val/
    ├── test/
    └── classes.txt
```

## Methodology

### Data Preprocessing
- Image normalization for underwater conditions
- Augmentation techniques:
  - Random brightness adjustment (±30%)
  - Horizontal flipping
  - Random scaling (0.8-1.2)
  - Color space adjustments for water refraction

### Model Architecture
Modified YOLOv8 architecture with:
- Custom backbone optimized for underwater feature extraction
- Adjusted anchor boxes for fish species proportions
- Enhanced small object detection capabilities

### Training Configuration
```python
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IMG_SIZE = 640
OPTIMIZER = 'Adam'
```

## Results
Model performance metrics:
- mAP@0.5: 0.85
- Average precision per class: 0.82
- Real-time inference: 25 FPS on RTX 3080

Detailed results available in `results/` directory.

## Usage

### Training
```bash
python train.py --data data.yaml --epochs 100 --batch 16
```

### Inference
```bash
python detect.py --source video.mp4 --weights best.pt
```

### Evaluation
```bash
python eval.py --weights best.pt --data data.yaml
```

## Project Structure
```
├── data/
├── models/
├── scripts/
│   ├── train.py
│   ├── detect.py
│   └── eval.py
├── utils/
├── results/
└── configs/
```

## Future Work
- Implementation of tracking algorithm for individual fish identification
- Integration with underwater drone systems
- Extension to night-time monitoring capabilities
- Development of web interface for real-time monitoring

## Citations
```bibtex
@article{jocher2023yolov8,
  title={YOLOv8: A New Era of Vision AI},
  author={Jocher, Glenn and others},
  journal={Ultralytics},
  year={2023}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Prof. [Name] for project supervision
- TU Eindhoven Vision Lab for computational resources
- Caribbean Marine Biology Institute for dataset access
