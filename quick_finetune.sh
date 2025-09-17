#!/bin/bash
"""
Quick launcher script for client dataset fine-tuning

This script provides common training configurations for different scenarios.
Usage:
  ./quick_finetune.sh [quick|standard|intensive]
"""

# Check if we're in the right directory
if [ ! -f "fine_tune_client_dataset.py" ]; then
    echo "Error: fine_tune_client_dataset.py not found. Please run this script from the project root."
    exit 1
fi

# Default configuration
EPOCHS=50
BATCH_SIZE=16
PATIENCE=10

# Parse arguments
case "${1:-standard}" in
    "quick")
        EPOCHS=20
        BATCH_SIZE=8
        PATIENCE=5
        echo "🚀 Quick training mode: $EPOCHS epochs, batch size $BATCH_SIZE"
        ;;
    "standard")
        EPOCHS=50
        BATCH_SIZE=16
        PATIENCE=10
        echo "⚖️  Standard training mode: $EPOCHS epochs, batch size $BATCH_SIZE"
        ;;
    "intensive")
        EPOCHS=100
        BATCH_SIZE=32
        PATIENCE=15
        echo "🔥 Intensive training mode: $EPOCHS epochs, batch size $BATCH_SIZE"
        ;;
    *)
        echo "Usage: $0 [quick|standard|intensive]"
        echo "  quick     - 20 epochs, batch 8, patience 5"
        echo "  standard  - 50 epochs, batch 16, patience 10"
        echo "  intensive - 100 epochs, batch 32, patience 15"
        exit 1
        ;;
esac

echo "Starting training with client dataset..."
echo "⏰ Started at: $(date)"

# Run the training script
python3 fine_tune_client_dataset.py \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --patience $PATIENCE

echo "⏰ Finished at: $(date)"
echo "✅ Training completed!"