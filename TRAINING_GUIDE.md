# YOLOv8 Enemy Detection Training

This guide shows how to fine-tune your YOLOv8 model using the augmented data.

## Prerequisites

Install YOLOv8:
```bash
pip install ultralytics
```

## Training Steps

### 1. Quick Training (Recommended)
```bash
python train_yolo.py
```

### 2. Manual Training
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.

# Train
results = model.train(
    data='augmented_data/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### 3. Command Line Training
```bash
yolo train data=augmented_data/dataset.yaml model=yolov8n.pt epochs=100 imgsz=640
```

## Model Selection

Choose based on your needs:
- **yolov8n.pt**: Fastest, smallest (6.2M parameters)
- **yolov8s.pt**: Balanced (11.2M parameters) 
- **yolov8m.pt**: More accurate (25.9M parameters)
- **yolov8l.pt**: High accuracy (43.7M parameters)
- **yolov8x.pt**: Best accuracy (68.2M parameters)

## Training Parameters

Key parameters to adjust:
- `epochs`: Number of training epochs (100-300)
- `batch`: Batch size (adjust based on GPU memory)
- `imgsz`: Image size (640, 1280)
- `patience`: Early stopping patience
- `device`: GPU device (0, 1, 2...) or 'cpu'

## Output Structure

After training, you'll get:
```
enemy_detection/
└── run1/
    ├── weights/
    │   ├── best.pt          # Best model weights
    │   └── last.pt          # Latest model weights
    ├── train_batch*.jpg     # Training visualizations
    ├── val_batch*.jpg       # Validation visualizations
    ├── confusion_matrix.png # Confusion matrix
    ├── results.png          # Training metrics plot
    └── ...
```

## Using the Trained Model

### Python Inference
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('enemy_detection/run1/weights/best.pt')

# Run inference
results = model('path/to/image.jpg')

# Process results
for r in results:
    boxes = r.boxes
    if boxes is not None:
        for box in boxes:
            conf = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"Enemy detected: confidence={conf:.3f}, bbox=({x1},{y1},{x2},{y2})")
```

### Real-time Detection
```python
import cv2
from ultralytics import YOLO

model = YOLO('enemy_detection/run1/weights/best.pt')

# For webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('Enemy Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

## Performance Tips

1. **GPU Training**: Use GPU for faster training
2. **Batch Size**: Increase if you have more GPU memory
3. **Image Size**: Larger images (1280) for better accuracy
4. **Data Augmentation**: YOLOv8 handles this automatically
5. **Early Stopping**: Use patience parameter to avoid overfitting

## Monitoring Training

Watch training progress:
- Live metrics in terminal
- TensorBoard logs (if enabled)
- Validation images in output folder
- Loss curves in results.png
