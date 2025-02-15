
# Fish Tracking and Counting Project

This project focuses on fish tracking and counting using object detection and tracking algorithms. It leverages YOLO for object detection and includes two implementations for tracking systems:

1. **Self Implementation**: Uses a custom Kalman filter-based tracking system to manage fish movements across video frames.
2. **ByteTrack Implementation**: A ready-to-use tracking implementation that utilizes ByteTrack for enhanced tracking.

## Table of Contents

1. Project Overview
2. Project Structure
3. Requirements
4. Usage
5. Customization

## Project Overview

The goal of this project is to identify and count fish in underwater video frames while accurately tracking individual fish movements. The system avoids double-counting fish even if they temporarily leave the frame or overlap with others. This project includes a custom tracking algorithm and an alternative ByteTrack implementation using the YOLO object detection model for fish localization.

## Project Structure

### Self Implementation (Kalman Filter Based)
- **main.py**: The main script that initiates the video processing and tracking pipeline.
- **tracker_manager.py**: Manages multiple trackers, initializes them, and coordinates tracking across video frames.
- **yolo_model.py**: Contains the code for loading and running the YOLO object detection model on video frames.
- **kalman_filter.py**: Implements the Kalman filter for predicting fish movements.
- **kalman_tracker.py**: Builds on the Kalman filter for tracking individual fish across frames.
- **video_processor.py**: Handles video processing, frame extraction, and integration with the tracking and detection modules.

### ByteTrack Implementation
- **ByteTrack_implementation.ipynb**: Jupyter notebook implementing the ByteTrack model for fish tracking in videos. It demonstrates how to set up and run ByteTrack for fish tracking using YOLO detections.

### Additional Tools
- **Frames_to_video.ipynb**: Jupyter notebook to convert extracted frames into a video, assisting in post-processing or video creation from processed frames.

## Requirements

- Python 3.x
- Required libraries can be installed by running:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

### Self Implementation

#### Example Commands

If you want to run the tracking on a video file:

```bash
python main.py --model_path yolo_model.pt \
               --video_path "/home/example/fish.mp4" \
               --output_folder "/home/example/processed_output"
```

If you already have the frames extracted and want to process them directly:

```bash
python main.py --model_path yolo_model.pt \
               --frames_folder "/home/example/DeepFish/Classification/7482/valid" \
               --output_folder "/home/example/processed_output"
```

### ByteTrack Implementation

To use the ByteTrack implementation, open and run the `ByteTrack_implementation.ipynb` notebook. This notebook guides you through setting up and running ByteTrack on a video file, using YOLO for object detection and ByteTrack for tracking.

### Frames to Video

The `Frames_to_video.ipynb` notebook can be used to convert processed frames back into a video format. This is useful for creating video outputs of tracked frames.

## Customization

- Modify detection thresholds or parameters in `yolo_model.py` for tuning object detection in the self-implementation.
- Adjust tracking parameters, such as prediction and update rates, in `kalman_filter.py` and `kalman_tracker.py` for optimal tracking performance in the self-implementation.
- For ByteTrack customization, refer to the parameters specified in `ByteTrack_implementation.ipynb`.
