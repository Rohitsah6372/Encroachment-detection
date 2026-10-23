from ultralytics import YOLO

# Pretrained YOLOv8 detection model
model = YOLO("yolov8n.pt")

model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="runs/encroachment",
    name="yolov8n",
)

# Evaluate the best checkpoint on the validation split
trained_model = YOLO("runs/encroachment/yolov8n/weights/best.pt")
metrics = trained_model.val(data="dataset.yaml")

print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision:{metrics.box.mp:.4f}")
print(f"Recall:   {metrics.box.mr:.4f}")