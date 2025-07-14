from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# === Paths ===
model_path = "/mnt/SamsungSSD/Prtljaga/yolov11_new_model_100epochs_indian_data/yolov11_default_indian_train_100_epochs/weights/best.pt"
model_path = "/mnt/SamsungSSD/Prtljaga/yolov11_new_model_indian_data/yolov11_indian_train/weights/best.pt"
video_path = "/mnt/SamsungSSD/Prtljaga/Hailuo_Video_Create a CCTV footage of peopl_398561169233838086.mp4"

# === Load model ===
model = YOLO(model_path)

# === Read first frame ===
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
cap.release()
if not success:
    raise RuntimeError("Failed to read video or no frames found.")

# === Convert BGR to RGB ===
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# === Run inference ===
results = model(frame_rgb)[0]  # results[0] = first image

# === Manual class mapping ===
class_names = {
    0: "luggage",
    1: "person"
}

colors = {
    0: (0, 0, 255),    # luggage → red (BGR)
    1: (255, 0, 0)     # person → blue (BGR)
}

# === Draw detections manually with class-specific colors ===
for box in results.boxes:
    cls_id = int(box.cls)
    label = class_names.get(cls_id, f"class_{cls_id}")
    conf = box.conf.item()
    
    # Get coordinates
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # Get color
    color = colors.get(cls_id, (0, 255, 0))  # fallback green

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Draw label
    label_text = f"{label} {conf:.2f}"
    cv2.putText(frame, label_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# === Convert for display ===
frame_rgb_annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# === Show result ===
plt.figure(figsize=(12, 8))
plt.imshow(frame_rgb_annotated)
plt.axis("off")
plt.title("Manual YOLOv8 Detection with Correct Labels")
plt.show()

# Optional: Save
cv2.imwrite("manual_labeled_detections.jpg", frame)
