from ultralytics import YOLO
import ultralytics

# Model Load
model = YOLO('best.pt')

source = '/Users/seunghunjang/Desktop/Project/EX_IMG/*.jpg'

result = model.predict(source, save=True)