# Kalman Filter implementation inspired by https://github.com/zziz/kalman-filter and https://en.wikipedia.org/wiki/Kalman_filter

import numpy as np
from typing import Optional, Tuple

class KalmanFilter:
    def __init__(self, 
                 F: np.ndarray, 
                 B: Optional[np.ndarray] = None, 
                 H: np.ndarray = None, 
                 Q: Optional[np.ndarray] = None, 
                 R: Optional[np.ndarray] = None, 
                 P: Optional[np.ndarray] = None, 
                 x0: Optional[np.ndarray] = None) -> None:

        """
        Initializes the Kalman Filter with given matrices and initial state.
        
        Parameters:
            F (np.ndarray): State transition matrix.
            B (Optional[np.ndarray]): Control input matrix.
            H (np.ndarray): Observation matrix.
            Q (Optional[np.ndarray]): Process noise covariance.
            R (Optional[np.ndarray]): Measurement noise covariance.
            P (Optional[np.ndarray]): Initial estimation error covariance.
            x0 (Optional[np.ndarray]): Initial state.
        """

        if F is None or H is None:
            raise ValueError("Set proper system dynamics.")

        # State vector dimension
        self.n = F.shape[1]
        # Measurement vector dimension
        self.m = H.shape[1]

        # System dynamics matrices
        self.F = F  # State transition matrix
        self.H = H  # Observation matrix
        self.B = np.zeros((self.n, 1)) if B is None else B  # Default B is zero if not provided

        # Covariance matrices
        self.Q = np.eye(self.n) if Q is None else Q  # Process noise covariance
        self.R = np.eye(self.m) if R is None else R  # Measurement noise covariance
        self.P = np.eye(self.n) if P is None else P  # Initial estimation error covariance

        # Initial state estimate
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u: Optional[np.ndarray] = 0, image_shape: Optional[Tuple[int, int]] = (640, 640)) -> np.ndarray:
        """Predicts the next state."""
        
        # State prediction and error covariance update
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        # Constrain prediction within image boundaries, if provided
        if image_shape:
            self.x[0][0] = np.clip(self.x[0][0], 0, image_shape[1] - 1)
            self.x[1][0] = np.clip(self.x[1][0], 0, image_shape[0] - 1)

        return self.x

    def update(self, z: np.ndarray) -> None:
        """Updates the state with a new measurement."""
        
        y = z - np.dot(self.H, self.x)  # Measurement residual
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain
        self.x = self.x + np.dot(K, y)  # State update
        I = np.eye(self.n)
        self.P = (I - np.dot(K, self.H)) @ self.P