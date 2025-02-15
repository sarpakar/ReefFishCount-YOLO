from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path, confidence):
        # Initialize the YOLO model using the provided model path
        self.model = YOLO(model_path)
        self.conf = confidence

    def predict(self, image):
        # Run inference on the image and return results
        return self.model.predict(source=image, conf=self.conf)