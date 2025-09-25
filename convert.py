from ultralytics import YOLO
model=YOLO('yolov8l.pt')
model.export(format='torchscript',imgsz=128,optimize=True, half=True) 
