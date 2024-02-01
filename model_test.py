from ultralytics import YOLO
import ultralytics

# Model Load
model = YOLO('best.pt')

source = '/Users/seunghunjang/Desktop/Project/IMG'

result = model.predict(source, save=True)