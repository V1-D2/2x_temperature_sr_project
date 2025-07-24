#!/bin/bash
#SBATCH --job-name=test_temperature_sr
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=01:00:00

echo "============================================"
echo "Testing Temperature SR Model: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Project directory
PROJECT_DIR="$HOME/2x_temperature_sr_project"
cd $PROJECT_DIR

# Create test results directory
mkdir -p test_temperature_results

echo "============================================"
echo "Checking environment..."
echo "============================================"

# Install required packages if needed
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user basicsr opencv-python timm matplotlib tqdm

# Test environment
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
except Exception as e:
    print(f'❌ PyTorch: {e}')
try:
    import basicsr
    print(f'✅ BasicSR: {basicsr.__version__}')
except Exception as e:
    print(f'❌ BasicSR: {e}')
"

echo "============================================"
echo "Testing Temperature SR Model..."
echo "============================================"

# Run the test script
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    --bind /home/vdidur/temperature_sr_project/data:/home/vdidur/temperature_sr_project/data:ro \
    --env PYTHONPATH=$PROJECT_DIR:$PYTHONPATH \
    --workdir $PROJECT_DIR \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python test_temperature_sr_model.py

# Check results
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "✅ Testing completed successfully!"
    echo "============================================"

    # Show test results summary from output
    echo ""
    echo "Test results have been displayed in the console output above."
    echo "Look for the TEST SUMMARY section for average metrics."

    # If SAVE_FILES was True, show saved files
    if [ -d "./test_temperature_results" ] && [ "$(ls -A ./test_temperature_results)" ]; then
        echo ""
        echo "Saved files in test_temperature_results/:"
        ls -la test_temperature_results/
    fi
else
    echo "============================================"
    echo "❌ Testing failed!"
    echo "Check the error messages above for details"
    echo "============================================"
fi

echo ""
echo "============================================"
echo "Temperature SR Testing Finished: $(date)"
echo "============================================"