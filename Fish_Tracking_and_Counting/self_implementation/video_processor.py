import numpy as np
import cv2
import os

class VideoProcessor:
    def __init__(self, model, tracker_manager):
        # Initialize the video processor with a detection model and tracker manager
        self.model = model
        self.tracker_manager = tracker_manager

    def process_image(self, image_path):
        # Load and resize the image
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (640, 640))

        # Run the YOLO model to detect objects
        results = self.model.predict(image_resized)
        detections = []

        # Extract detection coordinates and convert them to (center_x, center_y, width, height)
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = box[:4]
            width = x2 - x1
            height = y2 - y1
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            detections.append([x_center, y_center, width, height])

        return image_resized, detections

    def annotate_image(self, image, trackers, inactive_trackers, detections):
        # Annotate active trackers in red
        for tracker, track_id in trackers:
            pred_pos = tracker.predict()[:2]
            pred_x, pred_y = int(pred_pos[0][0]), int(pred_pos[1][0])
            cv2.rectangle(image, (pred_x - 20, pred_y - 20), (pred_x + 20, pred_y + 20), (0, 0, 255), 2)
            cv2.putText(image, f"ID: {track_id}", (pred_x, pred_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Annotate inactive trackers in yellow
        for tracker, track_id, frames_inactive in inactive_trackers:
            pred_pos = tracker.predict()[:2]
            pred_x, pred_y = int(pred_pos[0][0]), int(pred_pos[1][0])
            cv2.rectangle(image, (pred_x - 20, pred_y - 20), (pred_x + 20, pred_y + 20), (0, 255, 255), 2)
            cv2.putText(image, f"Inactive ID: {track_id} ({frames_inactive})", (pred_x, pred_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Annotate current detections in green
        for x, y, w, h in detections:
            top_left = (int(x - w / 2), int(y - h / 2))
            bottom_right = (int(x + w / 2), int(y + h / 2))
            cv2.rectangle(image, top_left, bottom_right, color=(0, 255, 0), thickness=2)

        return image

    def process_video(self, input_folder, output_folder):
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")])

        # Process each image frame by frame
        for i, image_file in enumerate(image_files):
            print(f"Processing image {i + 1}/{len(image_files)}: {image_file}")

            image_path = os.path.join(input_folder, image_file)
            image_resized, detections = self.process_image(image_path)

            # Update trackers with the detections
            self.tracker_manager.update_trackers(detections)

            # Annotate the image with tracker information
            annotated_image = self.annotate_image(
                image_resized,
                self.tracker_manager.get_active_trackers(),
                self.tracker_manager.get_inactive_trackers(),
                detections
            )

            # Save the annotated image
            output_image_path = os.path.join(output_folder, f"annotated_{image_file}")
            cv2.imwrite(output_image_path, annotated_image)

        print(f"Total unique fish IDs counted in video: {len(self.tracker_manager.get_all_fish_ids())}")
        print("Processing complete.")

    def extract_frames_from_video(self, video_path, output_folder):
        """
        Extract frames from a video file and save them to the specified output folder as individual image files.
        :param video_path: Path to the input video file.
        :param output_folder: Path to the folder where frames will be saved.
        """
        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        frame_count = 0

        # Extract frames from the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save each frame as an image file
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()
        print(f"Extracted {frame_count} frames to '{output_folder}'")
        