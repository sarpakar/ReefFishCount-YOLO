import cv2
import os
import numpy as np
from yolo_model import YOLOModel
from tracker_manager import TrackerManager
from video_processor import VideoProcessor
import argparse

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process videos or directories of frames using YOLO and Kalman tracking.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLO model file.")
    parser.add_argument("--video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--frames_folder", type=str, help="Path to the folder containing frames extracted from a video.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder where processed frames will be saved.")
    args = parser.parse_args()

    # Load the YOLO model using the provided model path
    model_path = args.model_path
    if not model_path:
        raise ValueError("Model path is not specified. Please provide a valid path to the YOLO model.")

    model = YOLOModel(model_path, 0.65)

    # Set up Kalman filter parameters for tracking
    dt = 1 / 60  # Time step (assumes 60 FPS)
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])  # State transition matrix
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Observation matrix
    Q = np.eye(4) * 0.1  # Process noise covariance
    R = np.eye(2) * 10  # Measurement noise covariance
    P = np.eye(4) * 1000  # Initial state covariance
    INACTIVE_REMOVAL_THRESHOLD = 60  # Number of frames after which inactive trackers are removed

    # Initialize the tracker manager with the Kalman filter parameters
    tracker_manager = TrackerManager(dt, F, H, Q, R, P, INACTIVE_REMOVAL_THRESHOLD)
    video_processor = VideoProcessor(model, tracker_manager)

    # Extract frames from video if video_path is provided
    if args.video_path:
        frames_output_folder = os.path.join(args.output_folder, "frames")
        video_processor.extract_frames_from_video(args.video_path, frames_output_folder)
        input_folder = frames_output_folder  # Set the input folder to the extracted frames
    elif args.frames_folder:
        input_folder = args.frames_folder  # Use the provided frames folder
    else:
        raise ValueError("Either video_path or frames_folder must be provided.")

    # Process the frames in the input folder and save the output
    output_folder = os.path.join(args.output_folder, "processed")
    video_processor.process_video(input_folder, output_folder)

    print("Processing complete.")
