from ultralytics import YOLO

# Load model and train
model = YOLO('yolov10m.pt')
results = model.train(
    data='data/data.yaml',
    epochs=100,
    batch=10,
    augment=True,
    degrees=10.0,
    mosaic=1,
    hsv_v=0.8,
    project='runs/train',
    name='yolov10_experiment',
    device='0'
)

# ===================== Alternative Training Experiments =====================
# Experiment 3: Lower learning rate finish
#results = model.train(
#    data='data.yaml',
#    epochs=20,
#    lrf=0.001,
#    batch=10
#)

# Experiment 5: Higher learning rate finish
#results = model.train(
#    data='data.yaml',
#    epochs=20,
#    lrf=0.1,
#    batch=10
#)

# Experiment with combined parameters
#results = model.train(
#    data='data.yaml',
#    epochs=100,
#    batch=10,
#    hsv_v=0.5,
#    lr0=0.001,
#    lrf=0.1
#)
# ========================================================================

# Load best weights and validate
model = YOLO('best.pt')
results = model.val(data='data.yaml', split='test')

# Run predictions
results = model.predict(
    source='/images',
    save=True,
    save_txt=True,
    save_conf=True,
    conf=0.25,
    iou=0.7
)

# Display results
results.show()