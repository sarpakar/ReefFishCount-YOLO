# Group 1 Data Challenge JBI100 

## Project Structure

### Fish_Detection_Model Folder
The **Fish_Detection_Model** folder contains the YOLOv10-based fish detection model. This part of the project is focused on robust fish detection across diverse datasets. It combines data from Fish4Knowledge, PANGEA, and DeepFish to ensure accurate fish localization and counting under various underwater conditions. The model includes dataset setup instructions, preprocessing notebooks, and YOLO configurations.

### Fish_Tracking_and_Counting Folder
The **Fish_Tracking_and_Counting** folder is dedicated to fish tracking and counting. It contains implementations for tracking individual fish across video frames, helping avoid double-counting even if fish temporarily leave the frame or overlap. Two implementations are provided:
- **Self Implementation**: Custom Kalman filter-based tracking system for fish movement tracking.
- **ByteTrack Implementation**: An enhanced tracking option using the ByteTrack model, integrated with YOLO detections for fish tracking.

Each folder includes its specific README for detailed setup and usage instructions.

---

This README provides a high-level overview; refer to each folder for in-depth instructions and guidelines.
