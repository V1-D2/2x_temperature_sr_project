#!/bin/bash
#SBATCH --job-name=temperature_sr_finetune
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH --time=72:00:00

echo "============================================"
echo "Temperature SR Fine-tuning Job Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "Memory allocated: 64GB per GPU"
echo "============================================"

# Set environment variables to suppress nvidia-smi warnings
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1

# Path to pretrained model
PRETRAINED_MODEL="$HOME/2x_temperature_sr_project/experiments/TemperatureSR_SwinIR_ESRGAN_x2_90k/models/net_g_45738.pth"

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "ERROR: Pretrained model not found at: $PRETRAINED_MODEL"
    echo "Please ensure the model exists before running fine-tuning"
    exit 1
fi

echo "Found pretrained model: $PRETRAINED_MODEL"

# Install required packages if not already installed
echo "Installing required packages..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user basicsr opencv-python timm facexlib gfpgan tqdm matplotlib

# Проверим среду в TensorFlow контейнере
echo "Testing environment in TensorFlow container:"
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import sys
print(f'Python: {sys.version}')
try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except Exception as e:
    print(f'❌ NumPy: {e}')
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU count: {torch.cuda.device_count()}')
        print(f'✅ GPU name: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch: {e}')
try:
    import cv2
    print(f'✅ OpenCV: {cv2.__version__}')
except Exception as e:
    print(f'❌ OpenCV: {e}')
try:
    import basicsr
    print(f'✅ BasicSR: {basicsr.__version__}')
except Exception as e:
    print(f'❌ BasicSR: {e}')
"

echo "============================================"
echo "Creating backup of original model..."
# Create backup directory with timestamp
BACKUP_DIR="$HOME/2x_temperature_sr_project/model_backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Copy original model to backup
cp "$PRETRAINED_MODEL" "$BACKUP_DIR/"
echo "Original model backed up to: $BACKUP_DIR"

echo "============================================"
echo "Checking data directory..."
echo "Data should be available at: /scratch/tmp/data"
ls -la /scratch/tmp/data | head -5
echo "Total NPZ files: $(ls /scratch/tmp/data/*.npz 2>/dev/null | wc -l)"

echo "============================================"
echo "Starting Temperature SR Fine-tuning with TensorFlow container:"

# Change to project directory
cd $HOME/2x_temperature_sr_project

# Fix CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set fine-tuning parameters
NUM_EPOCHS=50  # Number of additional epochs for fine-tuning
LEARNING_RATE=5e-5  # Lower learning rate for fine-tuning
EXPERIMENT_NAME="TemperatureSR_FineTune_$(date +%Y%m%d_%H%M%S)"

echo "Fine-tuning parameters:"
echo "  - Pretrained model: $PRETRAINED_MODEL"
echo "  - Additional epochs: $NUM_EPOCHS"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Experiment name: $EXPERIMENT_NAME"

# Запускаем Temperature SR fine-tuning в TensorFlow контейнере
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $HOME/2x_temperature_sr_project:$HOME/2x_temperature_sr_project \
    --bind /scratch/tmp/data:/scratch/tmp/data \
    --env PYTHONPATH=$HOME/2x_temperature_sr_project:$PYTHONPATH \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python fine_tune_temperature_sr.py \
    --pretrained_model "$PRETRAINED_MODEL" \
    --data_dir /scratch/tmp/data \
    --output_dir ./experiments_finetune \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --experiment_name "$EXPERIMENT_NAME" \
    --save_original

# Check if fine-tuning was successful
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "Fine-tuning completed successfully!"
    echo "Results saved in: ./experiments_finetune/$EXPERIMENT_NAME"
    echo ""
    echo "Model files:"
    ls -la ./experiments_finetune/$EXPERIMENT_NAME/models/
    echo ""
    echo "Original model backup is preserved in: $BACKUP_DIR"
else
    echo "============================================"
    echo "ERROR: Fine-tuning failed!"
    echo "Check the error messages above"
    echo "Original model is preserved in: $BACKUP_DIR"
fi

echo "============================================"
echo "Temperature SR Fine-tuning Job Finished: $(date)"
echo "============================================"