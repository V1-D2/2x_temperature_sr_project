#!/bin/bash
#SBATCH --job-name=cascaded_sr_4x
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=64G
#SBATCH --time=24:00:00

echo "============================================"
echo "Cascaded Temperature SR 4x Job Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "Memory allocated: 64GB per GPU"
echo "============================================"

# Set environment variables to suppress nvidia-smi warnings
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1

# Project paths
PROJECT_DIR="$HOME/2x_temperature_sr_project"
DATA_DIR="/scratch/tmp/data"
OUTPUT_DIR="$PROJECT_DIR/cascaded_results"

# Model path - update this to your trained model checkpoint
# Assuming the model is in experiments folder from training
MODEL_PATH="$PROJECT_DIR/experiments/TemperatureSR_SwinIR_ESRGAN_x2_90k/models/net_g_latest.pth"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please update MODEL_PATH in this script to point to your trained model"
    echo "Looking for available models..."
    find $PROJECT_DIR/experiments -name "*.pth" -type f
    exit 1
fi

echo "Using model: $MODEL_PATH"

# Install required packages if not already installed
echo "Checking/Installing required packages..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user basicsr opencv-python timm matplotlib tqdm

# Create output directory
mkdir -p $OUTPUT_DIR

echo "============================================"
echo "Checking environment..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
        print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
except Exception as e:
    print(f'❌ PyTorch: {e}')
"

echo "============================================"
echo "Checking data directory..."
ls -la $DATA_DIR | head -10
echo "Total NPZ files: $(ls $DATA_DIR/*.npz 2>/dev/null | wc -l)"

echo "============================================"
echo "Starting Cascaded Temperature SR 4x processing:"

# Change to project directory
cd $PROJECT_DIR

# Add project to PYTHONPATH
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# Fix CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the cascaded super-resolution
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    --bind $DATA_DIR:$DATA_DIR \
    --env PYTHONPATH=$PROJECT_DIR:$PYTHONPATH \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python cascaded_temperature_sr_4x.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --num_samples 5 \
    --patch_size 1000 110 \
    --overlap_ratio 0.75

# Check if processing was successful
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "Processing completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Output directory contents:"
    ls -la $OUTPUT_DIR
    echo ""
    echo "To view results:"
    echo "  - 4-panel comparisons: $OUTPUT_DIR/comparison_sample_*.png"
    echo "  - Individual samples: $OUTPUT_DIR/sample_*/"
    echo "  - NPZ data files: $OUTPUT_DIR/result_sample_*.npz"
    echo "  - Statistics: $OUTPUT_DIR/cascaded_sr_statistics.txt"
else
    echo "============================================"
    echo "ERROR: Processing failed!"
    echo "Check the error messages above for details"
fi

echo "============================================"
echo "Cascaded Temperature SR 4x Job Finished: $(date)"
echo "============================================"