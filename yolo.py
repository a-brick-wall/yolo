import cv2
from ultralytics import YOLO
import PIL

model = YOLO('yolov8l.pt')
cap = cv2.VideoCapture(0)

while True:
  cap = cv2.VideoCapture(0)
  ret, frame = cap.read()

  results = model(frame, stream=True)
  
  for result in results:
    annotated_frame = result.plot()
  
  try:
    cv2.imshow('Yolo live predictions', annotated_frame)#annotated_frame)
    cv2.waitKey(.01)

  except:
    pass
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
  
  cap.release()


