from ultralytics import YOLO
import cv2

# Cargar modelo de detección de pose humana
model = YOLO("yolov8n-pose.pt")

cap = cv2.VideoCapture(0)  # Usa tu webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()  # Dibuja los keypoints detectados
    cv2.imshow("Pose detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc para salir
        break

cap.release()
cv2.destroyAllWindows()