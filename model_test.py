from ultralytics import YOLO
import ultralytics

# Model Load
model = YOLO('best.pt')

source = '/Users/seunghunjang/Desktop/YOLO_Detect/Test_Set'

result = model.predict(source, save=True)