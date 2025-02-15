import numpy as np
import cv2
import os
from kalman_tracker import KalmanTracker

class TrackerManager:
    def __init__(self, dt, F, H, Q, R, P, inactive_removal_threshold):
        # Initialize tracking parameters
        self.global_tracker_id = 1  # Unique ID for each tracker
        self.all_fish_ids = set()  # Set of all unique fish IDs
        self.trackers = []  # List of active trackers
        self.inactive_trackers = []  # List of inactive trackers
        self.no_detection_count = []  # List to track the number of frames without detection for each tracker
        self.dt = dt  # Time step
        self.F = F  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Initial state covariance
        self.inactive_removal_threshold = inactive_removal_threshold  # Threshold to remove inactive trackers

    def check_inactive_tracker_overlap(self, inactive_tracker, detection, max_dist=80):
        # Check if an inactive tracker is close to a detection
        tracker_pred = inactive_tracker.predict()[:2]  # Predicted position of the inactive tracker
        detection_center = np.array([[detection[0]], [detection[1]]])  # Center of the detection
        dist = np.linalg.norm(tracker_pred - detection_center)  # Calculate Euclidean distance
        return dist < max_dist  # Return True if the distance is less than the threshold

    def assign_detections_to_tracks(self, detections, max_dist=80):
        # Assign detections to existing trackers using a distance-based approach
        matched_indices = []  # List of matched tracker and detection indices
        unmatched_detections = list(range(len(detections)))  # List of indices for unmatched detections
        unmatched_trackers = list(range(len(self.trackers)))  # List of indices for unmatched trackers

        matched_detections_set = set()  # Set to keep track of matched detections

        # Iterate over each tracker to find the best matching detection
        for t, (tracker, _) in enumerate(self.trackers):
            pred_pos = tracker.predict()[:2]  # Predicted position of the tracker
            min_dist = float('inf')
            match_idx = -1

            # Iterate over each detection to find the closest one to the tracker
            for d, det in enumerate(detections):
                if d in matched_detections_set:
                    continue  # Skip if this detection is already matched

                det_pos = np.array([[det[0]], [det[1]]])  # Center of the detection
                dist = np.linalg.norm(pred_pos - det_pos)  # Calculate Euclidean distance

                # Update the minimum distance and matched index if conditions are met
                if dist < min_dist and dist < max_dist:
                    min_dist = dist
                    match_idx = d

            # If a match is found, add to the matched list and update unmatched lists
            if match_idx >= 0:
                matched_indices.append((t, match_idx))
                matched_detections_set.add(match_idx)
                unmatched_trackers.remove(t)

        # Update unmatched detections list
        unmatched_detections = [d for d in range(len(detections)) if d not in matched_detections_set]

        return matched_indices, unmatched_detections, unmatched_trackers

    def update_trackers(self, detections):
        # Update existing trackers and create new trackers for unmatched detections
        matched, unmatched_detections, unmatched_trackers = self.assign_detections_to_tracks(detections)

        # Update matched trackers with the new detection
        for t_idx, d_idx in matched:
            x, y, w, h = detections[d_idx]
            self.trackers[t_idx][0].update(np.array([[x], [y]]))  # Update tracker with the detection
            self.no_detection_count[t_idx] = 0  # Reset no detection count for matched trackers

        # Create new trackers for unmatched detections
        for d_idx in unmatched_detections:
            x, y, w, h = detections[d_idx]
            assigned_tracker = False

            # Check if any inactive tracker can be reassigned
            for t_idx, (inactive_tracker, inactive_id, frames_inactive) in enumerate(self.inactive_trackers):
                if self.check_inactive_tracker_overlap(inactive_tracker, [x, y, w, h]):
                    self.trackers.append((inactive_tracker, inactive_id))  # Reactivate the tracker
                    self.no_detection_count.append(0)  # Reset no detection count
                    del self.inactive_trackers[t_idx]  # Remove from inactive list
                    assigned_tracker = True
                    break

            # If no inactive tracker matches, create a new tracker
            if not assigned_tracker:
                initial_state = np.array([[x], [y], [0], [0]])  # Initial state with zero velocity
                new_tracker = KalmanTracker(self.dt, self.F, self.H, self.Q, self.R, self.P, initial_state)
                self.trackers.append((new_tracker, self.global_tracker_id))  # Assign a new unique ID
                self.no_detection_count.append(0)  # Initialize no detection count for the new tracker
                self.all_fish_ids.add(self.global_tracker_id)  # Add the new ID to the set of all fish IDs
                self.global_tracker_id += 1  # Increment the global tracker ID counter

        # Increment no detection count for unmatched trackers
        for t_idx in unmatched_trackers:
            self.no_detection_count[t_idx] += 1

        # Move trackers that have gone unmatched for a while to inactive trackers
        trackers_to_move = [t_idx for t_idx, count in enumerate(self.no_detection_count) if count > 0]
        for t_idx in reversed(trackers_to_move):  # Iterate in reverse to avoid indexing issues
            self.inactive_trackers.append((*self.trackers[t_idx], 0))  # Add to inactive list with inactive frame count
            del self.trackers[t_idx]
            del self.no_detection_count[t_idx]

        # Remove inactive trackers that have been inactive for too long
        self.inactive_trackers = [
            (tracker, track_id, frames_inactive + 1)
            for tracker, track_id, frames_inactive in self.inactive_trackers
            if frames_inactive < self.inactive_removal_threshold
        ]

    def get_active_trackers(self):
        # Return the list of active trackers
        return self.trackers

    def get_inactive_trackers(self):
        # Return the list of inactive trackers
        return self.inactive_trackers

    def get_all_fish_ids(self):
        # Return the set of all unique fish IDs tracked so far
        return self.all_fish_ids
