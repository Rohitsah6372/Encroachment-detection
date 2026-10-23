
from ultralytics import YOLO
from pathlib import Path

# Load your model weights
weights = Path("models/weights.pt")
model = YOLO(str(weights))

# Run detection on a test image
results = model.predict(
    source="test.jpg",
    conf=0.25,
    save=True
)

print("Detection complete!")
