# Client Dataset Fine-Tuning Guide

This guide explains how to fine-tune your existing YOLO model with the client dataset containing enemy/player detections.

## Overview

The fine-tuning process takes your existing trained YOLO model (trained on synthetic data) and continues training with real-world client data to improve accuracy and adapt to actual game scenarios.

## Files Created

1. **`fine_tune_client_dataset.py`** - Main fine-tuning script with comprehensive configuration
2. **`quick_finetune.sh`** - Convenient launcher script with preset configurations
3. **`CLIENT_FINETUNE_GUIDE.md`** - This documentation

## Dataset Structure

Your client dataset should have this structure (already verified):
```
client-dataset/
├── data.yaml           # Dataset configuration
├── train/
│   ├── images/         # Training images
│   └── labels/         # Training labels (YOLO format)
├── val/
│   ├── images/         # Validation images
│   └── labels/         # Validation labels
└── test/
    ├── images/         # Test images (optional)
    └── labels/         # Test labels (optional)
```

## Quick Start

### Option 1: Use the Launcher Script (Recommended)

```bash
# Quick training (20 epochs, smaller batch)
./quick_finetune.sh quick

# Standard training (50 epochs, balanced settings)
./quick_finetune.sh standard

# Intensive training (100 epochs, larger batch)
./quick_finetune.sh intensive
```

### Option 2: Use Python Script Directly

```bash
# Basic usage with default settings
python3 fine_tune_client_dataset.py

# Custom configuration
python3 fine_tune_client_dataset.py --epochs 30 --batch-size 8 --patience 8

# Use a different base model
python3 fine_tune_client_dataset.py --model path/to/your/model.pt --output-name my_custom_model.pt
```

## Training Parameters

### Default Settings (Optimized for Fine-tuning)
- **Learning Rate**: 0.001 (reduced for fine-tuning)
- **Epochs**: 50 (with early stopping)
- **Batch Size**: 16
- **Patience**: 10 epochs
- **Optimizer**: AdamW
- **Augmentation**: Moderate (reduced from training-from-scratch)

### Hardware Requirements
- **Minimum**: 8GB RAM, 4GB VRAM
- **Recommended**: 16GB RAM, 8GB VRAM
- **Optimal**: 32GB RAM, 12GB+ VRAM

## Expected Training Process

1. **Validation** - Script validates dataset structure and paths
2. **Configuration** - Creates training-specific configuration files
3. **Model Loading** - Loads your existing model weights
4. **Training** - Fine-tunes with client data
5. **Saving** - Saves best weights and training artifacts

## Output Files

After training, you'll find:

```
runs/detect/client_finetune_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt         # Best model weights
│   ├── last.pt         # Latest epoch weights
│   └── epoch*.pt       # Periodic checkpoints
├── training_data.yaml  # Training configuration used
├── results.png         # Training curves
├── confusion_matrix.png
├── val_batch*.jpg      # Validation visualizations
└── args.yaml          # All training arguments
```

The script also copies the best weights to:
- `client_finetuned_best.pt` (in project root)

## Using the Fine-tuned Model

After training, use the fine-tuned model in your applications:

```python
from ultralytics import YOLO

# Load the fine-tuned model
model = YOLO('client_finetuned_best.pt')

# Run inference
results = model('path/to/your/image.jpg')

# Or use in your existing test notebooks
model = YOLO('client_finetuned_best.pt')
# ... rest of your inference code
```

## Monitoring Training

### Real-time Monitoring
Training progress is displayed in the terminal with:
- Loss curves (train/validation)
- mAP metrics
- Learning rate schedule
- ETA and time per epoch

### Visualizations
Generated automatically:
- Training/validation loss curves
- Precision/Recall curves
- Confusion matrix
- Sample predictions on validation set

### TensorBoard (Optional)
If you have TensorBoard installed:
```bash
tensorboard --logdir runs/detect/
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python3 fine_tune_client_dataset.py --batch-size 8
   ```

2. **Dataset Path Issues**
   - Ensure `client-dataset/` exists in project root
   - Check that `data.yaml` has correct paths

3. **No Existing Model Found**
   - Script will automatically fall back to YOLO11n base model
   - You can specify a custom model with `--model path/to/model.pt`

### Performance Tips

1. **For Better Results**:
   - Use `intensive` mode for higher quality
   - Ensure good train/val split (80/20)
   - Monitor validation metrics to avoid overfitting

2. **For Faster Training**:
   - Use `quick` mode for testing
   - Reduce batch size if memory constrained
   - Use smaller input image size (modify in data.yaml)

## Class Mapping

Your client dataset uses:
- Class 0: `enemy` (based on data.yaml)

The fine-tuning will adapt your existing model to better detect these classes in real game scenarios.

## Next Steps

After fine-tuning:

1. **Test the Model**: Use `test-fine-tuned.ipynb` with your new model
2. **Compare Performance**: Run inference on both old and new models
3. **Deploy**: Use the fine-tuned model in your aim assist application
4. **Iterate**: Collect more data and repeat fine-tuning as needed

## Advanced Configuration

For advanced users, you can modify the script to:
- Adjust hyperparameters
- Change augmentation settings
- Modify training schedule
- Add custom callbacks

See the script comments for detailed parameter explanations.