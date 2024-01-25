from ultralytics import YOLO

# Model Load
model = YOLO('yolov8m.pt')

# Model Train
model.train(data='data.yaml', epochs=100, imgsz=640)