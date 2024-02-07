from ultralytics import YOLO

dataset_source = '/Users/seunghunjang/Desktop/YOLO_Detect/data.yaml'

# Model Load
model = YOLO('yolov8m.pt')

# Model Train
model.train(data=dataset_source, epochs=10, imgsz=640)