from kalman_filter import KalmanFilter

class KalmanTracker:
    def __init__(self, dt, F, H, Q, R, P, initial_state):
        # Initialize the Kalman filter with the given parameters
        self.filter = KalmanFilter(F=F, H=H, Q=Q, R=R, P=P, x0=initial_state)

    def predict(self):
        # Predict the next state using the Kalman filter
        return self.filter.predict()
            
    def update(self, measurement):
        # Update the Kalman filter with the new measurement
        self.filter.update(measurement)
