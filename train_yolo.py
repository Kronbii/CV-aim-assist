#!/usr/bin/env python3
"""
YOLOv8 Fine-tuning Script for Enemy Detection
"""

from ultralytics import YOLO
import os

def train_enemy_detection_model():
    """
    Fine-tune YOLOv8 model for enemy detection using augmented data
    """
    
    # Configuration
    DATA_YAML = "augmented_data/dataset.yaml"
    PRETRAINED_MODEL = "yolov8n.pt"  # or yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    PROJECT_NAME = "enemy_detection"
    EXPERIMENT_NAME = "run1"
    
    # Training parameters
    EPOCHS = 100
    BATCH_SIZE = 16  # Adjust based on your GPU memory
    IMAGE_SIZE = 640
    
    print("=" * 60)
    print("YOLOv8 Enemy Detection Fine-tuning")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(DATA_YAML):
        print(f"❌ Dataset file not found: {DATA_YAML}")
        print("Please run the data augmentation notebook first!")
        return
    
    # Load pretrained model
    print(f"📦 Loading pretrained model: {PRETRAINED_MODEL}")
    model = YOLO(PRETRAINED_MODEL)
    
    # Display model info
    print(f"📊 Model architecture: {PRETRAINED_MODEL}")
    print(f"📁 Dataset: {DATA_YAML}")
    print(f"🎯 Target: Enemy detection (1 class)")
    print(f"⚙️  Epochs: {EPOCHS}")
    print(f"📏 Image size: {IMAGE_SIZE}")
    print(f"🔢 Batch size: {BATCH_SIZE}")
    
    print("\n🚀 Starting training...")
    
    # Train the model
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        patience=50,     # Early stopping patience
        device=0,        # Use GPU 0, change to 'cpu' if no GPU
        workers=8,       # Number of worker threads
        cache=True,      # Cache images for faster training
        # Data augmentation (YOLOv8 handles this automatically)
        hsv_h=0.015,     # Image HSV-Hue augmentation
        hsv_s=0.7,       # Image HSV-Saturation augmentation
        hsv_v=0.4,       # Image HSV-Value augmentation
        degrees=0.0,     # Image rotation (+/- deg)
        translate=0.1,   # Image translation (+/- fraction)
        scale=0.5,       # Image scale (+/- gain)
        shear=0.0,       # Image shear (+/- deg)
        perspective=0.0, # Image perspective (+/- fraction)
        flipud=0.0,      # Image flip up-down (probability)
        fliplr=0.5,      # Image flip left-right (probability)
        mosaic=1.0,      # Image mosaic (probability)
        mixup=0.0,       # Image mixup (probability)
    )
    
    print("\n✅ Training completed!")
    
    # Print results
    print(f"📁 Results saved to: {PROJECT_NAME}/{EXPERIMENT_NAME}")
    print(f"🏆 Best model: {PROJECT_NAME}/{EXPERIMENT_NAME}/weights/best.pt")
    print(f"📈 Training plots: {PROJECT_NAME}/{EXPERIMENT_NAME}/")
    
    # Validate the model
    print("\n🔍 Running validation...")
    metrics = model.val()
    
    print("\n📊 Validation Metrics:")
    print(f"   mAP50: {metrics.box.map50:.3f}")
    print(f"   mAP50-95: {metrics.box.map:.3f}")
    print(f"   Precision: {metrics.box.mp:.3f}")
    print(f"   Recall: {metrics.box.mr:.3f}")
    
    return model, results

def test_model_inference():
    """
    Test the trained model on a sample image
    """
    
    # Load the best trained model
    model_path = "enemy_detection/run1/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        print("Please run training first!")
        return
    
    print("\n🧪 Testing model inference...")
    model = YOLO(model_path)
    
    # Test on sample images
    test_images = [
        "augmented_data/1.jpg",
        "augmented_data/2.jpg",
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"🔍 Testing on: {img_path}")
            results = model(img_path)
            
            # Print detections
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    print(f"   Found {len(boxes)} enemies")
                    for i, box in enumerate(boxes):
                        conf = box.conf[0].item()
                        print(f"   Enemy {i+1}: Confidence {conf:.3f}")
                else:
                    print("   No enemies detected")
        else:
            print(f"⚠️  Test image not found: {img_path}")

if __name__ == "__main__":
    # Train the model
    model, results = train_enemy_detection_model()
    
    # Test inference
    test_model_inference()
    
    print("\n🎉 All done! Your enemy detection model is ready!")
    print("\nNext steps:")
    print("1. Check training plots in enemy_detection/run1/")
    print("2. Use the best model: enemy_detection/run1/weights/best.pt")
    print("3. Deploy for real-time enemy detection!")
