# Thermal Satellite Image Super-Resolution: SwinIR + Real-ESRGAN Hybrid Architecture

**A transformer-based super-resolution system for AMSR-2 thermal satellite imagery, achieving 2× upsampling with cascaded 8× capability**

---

## Overview

This repository implements a specialized deep learning pipeline for super-resolution of thermal satellite imagery from the AMSR-2 (Advanced Microwave Scanning Radiometer 2) sensor. The system combines SwinIR's transformer-based architecture with Real-ESRGAN's adversarial training framework to reconstruct high-resolution temperature fields from low-resolution microwave radiometer observations.

**Key capabilities:**
- Native 2× super-resolution with preserved thermodynamic properties
- Cascaded inference for 4× and 8× upsampling
- Physics-aware loss functions for temperature field consistency
- Patch-based processing for arbitrary image sizes
- Gaussian-weighted blending for seamless reconstruction

**Performance metrics:**
- PSNR: 39 dB (2× SR)
- SSIM: 0.97 (structural similarity preserved)
- Temperature drift: 0.2K across 8× cascaded stages

---

## Architecture

The system consists of a generator (SwinIR-based) and discriminator (U-Net with spectral normalization) operating in an adversarial training framework.

### Generator: SwinIR Temperature Transformer

The generator processes single-channel thermal images (1×H×W) through a hierarchical transformer architecture:

**1. Shallow Feature Extraction**
```
Conv2d(1 → 60, kernel=3×3) → 60×H×W
```
Embeds raw temperature values into a 60-dimensional feature space.

**2. Deep Feature Extraction: 6 RSTB Blocks**

Each Residual Swin Transformer Block (RSTB) contains:
- 6 Swin Transformer layers with alternating attention patterns
- Window size: 8×8 pixels (optimal for thermal gradients)
- Embedding dimension: 60 channels
- Attention heads: 6 per layer
- MLP expansion ratio: 4× (60 → 240 → 60)

**Attention mechanism:**
- **W-MSA (Window Multi-head Self-Attention):** Layers 1, 3, 5 compute attention within 8×8 windows
- **SW-MSA (Shifted Window):** Layers 2, 4, 6 shift features by 4 pixels before windowing, enabling cross-window information flow

**Relative position bias:** Each attention layer uses a learnable (169 × 6) bias table encoding spatial relationships between query-key pairs.

**3-Conv Residual Connection:**
```
Conv(60 → 15, 3×3) → LeakyReLU → Conv(15 → 15, 1×1) → LeakyReLU → Conv(15 → 60, 3×3)
```
Refines features with efficient channel reduction before adding back to the main path.

**3. Feature Fusion**
```
Conv2d(60 → 60, kernel=3×3)
```
Aggregates multi-scale features from all RSTB blocks.

**4. Progressive Upsampling (2× scale)**
```
Conv(60 → 64, 3×3) → LeakyReLU → Conv(64 → 256, 3×3) → PixelShuffle(r=2) → 64×2H×2W
```
PixelShuffle rearranges 256 channels into spatial dimensions (256 = 64×2²), avoiding checkerboard artifacts common in transposed convolutions.

**5. Reconstruction**
```
Conv2d(64 → 1, kernel=3×3) → 1×2H×2W
```
Projects features back to temperature space.

**Parameter count:** ~3.7M (generator)

### Discriminator: U-Net with Spectral Normalization

The discriminator distinguishes real high-resolution images from generator outputs using a U-Net architecture:

**Encoder (downsampling):**
```
Conv0: 1 → 64 (stride=1)
Conv1: 64 → 128 (stride=2) + SpectralNorm → H/2×W/2
Conv2: 128 → 256 (stride=2) + SpectralNorm → H/4×W/4
Conv3: 256 → 512 (stride=2) + SpectralNorm → H/8×W/8
```

**Decoder (upsampling + skip connections):**
```
Upsample + Conv4: 512 → 256 + skip(256) → H/4×W/4
Upsample + Conv5: 256 → 128 + skip(128) → H/2×W/2
Upsample + Conv6: 128 → 64 + skip(64) → H×W
```

**Classification head:**
```
Conv7: 64 → 64 (feature refinement)
Conv8: 64 → 64 (feature refinement)
Conv9: 64 → 1 (authenticity map) → 1×H×W
```
Outputs per-pixel authenticity scores.

**Spectral normalization:** Constrains weight matrices by their largest singular value (W_SN = W / σ(W)), preventing gradient explosion and stabilizing GAN training by enforcing Lipschitz continuity ≤ 1.

**Parameter count:** ~2.8M (discriminator)

---

## Loss Functions

### Physics-Consistency Loss (Pixel-Level)

```python
L_pixel = L1(pred, target) + λ_grad·L_gradient + λ_smooth·L_smoothness
```

**Components:**
- **L1 loss:** Preserves absolute temperature values
- **Gradient loss:** Maintains sharpness at thermal boundaries by matching spatial derivatives
- **Smoothness loss:** Penalizes second-order derivatives to prevent artifacts

**Weights:** 100× pixel + 0.08× gradient + 0.03× smoothness

### Temperature Perceptual Loss

A custom feature extractor designed for thermal data:

```
[Conv(1→32, 3×3) + ReLU + Conv(32→32, 3×3, stride=2)] × 4 stages
```

Features extracted at 4 scales (1×, 1/2×, 1/4×, 1/8×) capture hierarchical thermal patterns. L1 distance between multi-scale features preserves perceptual quality beyond pixel-wise metrics.

**Weight:** 10× perceptual

### GAN Loss

LSGAN (Least Squares GAN) formulation:
```
L_G = E[(D(G(x)) - 1)²]  (generator)
L_D = E[(D(y) - 1)²] + E[D(G(x))²]  (discriminator)
```

Encourages realistic texture synthesis while avoiding vanishing gradients.

**Weight:** 1× GAN

**Total loss:** 100×L_pixel + 10×L_perceptual + 1×L_GAN

---

## Training Configuration

**Optimizer:** AdamW (lr=1×10⁻⁴, weight_decay=1×10⁻³)

**Schedule:** Cosine annealing (T_max=100k iterations, η_min=1×10⁻⁶)

**Mixed precision:** FP16 forward/backward, FP32 optimizer states

**Discriminator updates:** 5:1 ratio (5 D updates per 1 G update)

**Warmup:** 5,000 iterations discriminator pre-training

**Gradient clipping:** Max norm = 7.0 (prevents instability in transformer layers)

**Batch size:** 2-8 (depending on GPU memory)

**Data augmentation:** None (thermal data integrity preserved)

---

## Cascaded Inference for 4× and 8× Upsampling

The 2× model is applied recursively:

```
Original (H×W) → [2× SR] → 2H×2W → [2× SR] → 4H×4W → [2× SR] → 8H×8W
```

### Patch-Based Processing

For large images, overlapping patches with Gaussian-weighted blending:

**1. Patch extraction:**
- Patch size: 1000×110 pixels (divisible by window_size × scale)
- Overlap ratio: 75%
- Stride: 250×27 pixels

**2. Gaussian weight map:**
```python
W(y,x) = exp(-((y-H/2)²/(2σ_y²) + (x-W/2)²/(2σ_x²)))
```
σ = 0.3 × patch_dimension

**3. Weighted accumulation:**
```python
Output[region] = Σ(SR_patch × W) / Σ(W)
```

**Result:** Seamless reconstruction without boundary artifacts

**Cumulative error:** 0.2K drift over 8× cascading (0.07% relative error)

---

## Repository Structure

```
2x_temperature_sr_project/
├── hybrid_model.py                    # SwinIR + Real-ESRGAN integration
├── models/
│   └── network_swinir.py              # SwinIR architecture implementation
├── realesrgan/
│   ├── archs/
│   │   └── discriminator_arch.py      # UNetDiscriminatorSN
│   └── models/
│       └── realesrgan_model.py        # GAN training logic
├── data_preprocessing.py              # Temperature normalization & LR-HR pair generation
├── config_temperature.py              # Hyperparameters & network configs
├── train_temperature_sr.py            # Training script
├── fine_tune_temperature_sr.py        # Fine-tuning on additional data
├── cascaded_temperature_sr_unified.py # 4×/8× cascaded inference
├── utils/
│   └── util_calculate_psnr_ssim.py    # Evaluation metrics
└── run/
    ├── SR_run.sbatch                  # SLURM training job
    └── inference/
        └── run_cascded_temperature_sr.sbatch  # SLURM inference job
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/thermal-satellite-sr.git
cd thermal-satellite-sr

# Install dependencies
pip install torch torchvision basicsr opencv-python timm matplotlib tqdm
```

**Requirements:**
- Python ≥ 3.7
- PyTorch ≥ 1.7
- CUDA 11.0+ (for GPU acceleration)

---

## Usage

### Training

```bash
python train_temperature_sr.py \
    --data_dir /path/to/amsr2/data \
    --output_dir ./experiments \
    --num_epochs 100 \
    --batch_size 4
```

**Input format:** NPZ files with `temperature` (H×W array) and `metadata` fields.

**Data preprocessing:**
- Crop/pad to 2000×220 pixels
- Normalize to [0, 1] range
- Generate LR images via 2×2 averaging (physically accurate downsampling)

### Cascaded Inference (4× and 8×)

```bash
python cascaded_temperature_sr_unified.py \
    --npz-dir /path/to/test/data \
    --model-path ./experiments/models/net_g_45738.pth \
    --num-samples 5 \
    --save-dir ./cascaded_results
```

**Output structure:**
```
cascaded_results/
├── results_4x/
│   ├── arrays/          # NPZ files (original, sr, bicubic)
│   └── images/          # PNG visualizations
├── results_8x/
│   ├── arrays/
│   └── images/
├── 4x_comparison.png
├── 8x_comparison.png
└── statistics_report.txt
```

### Fine-Tuning

```bash
python fine_tune_temperature_sr.py \
    --pretrained_model ./experiments/models/net_g_50000.pth \
    --data_dir /path/to/new/data \
    --num_epochs 50 \
    --learning_rate 5e-5
```

Automatically backs up the original model before fine-tuning.

---

## Evaluation

```bash
python test_temperature_sr_model.py \
    --model_path ./experiments/models/net_g_45738.pth \
    --data_dir /path/to/test/data \
    --num_samples 500
```

**Metrics calculated:**
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Mean absolute temperature error (K)
- Max absolute temperature error (K)

---

## Technical Highlights

**1. Thermal-specific design choices:**
- Single-channel input/output (temperature field)
- Physics-aware normalization preserving relative temperature differences
- Gradient loss component maintaining thermal boundary sharpness

**2. Transformer advantages over CNNs:**
- Global receptive field via shifted window attention
- Better capture of long-range temperature correlations
- Reduced parameter count (60-dim embedding vs. 240-dim CNN features)

**3. Adversarial training benefits:**
- Sharper reconstruction of thermal gradients
- Improved perceptual quality (human preference: 98% over bicubic)
- Realistic texture synthesis for cloud/surface features

**4. Computational optimizations:**
- Patch-based inference for arbitrary image sizes
- Gaussian blending eliminates boundary artifacts
- Mixed-precision training (2× speedup, same accuracy)

---

## Limitations

**1. Cascading error accumulation:** Each 2× stage introduces ~0.1K drift. For 8× upsampling, consider direct multi-scale training.

**2. Fixed window size:** 8×8 windows optimized for AMSR-2's spatial resolution. Other sensors may require adjustment.

**3. Self-supervised training:** LR images synthesized via averaging. Real sensor degradation (noise, blur) may differ.

**4. Memory requirements:** Full image processing requires ~8GB GPU memory for 2048×208 inputs.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Volodymyr_Didur_@025
  title={Thermal Satellite Image Super-Resolution via SwinIR-ESRGAN Hybrid Architecture},
  author={Volodymyr Didur},
  journal={AGU2025},
  year={2025}
}
```

**Related work:**
- SwinIR: [Liang et al., 2021](https://arxiv.org/abs/2108.10257)
- Real-ESRGAN: [Wang et al., 2021](https://arxiv.org/abs/2107.10833)

---

## License

This project is released under the MIT License.

---

## Acknowledgments

Built upon [BasicSR](https://github.com/XPixelGroup/BasicSR) framework for image restoration. SwinIR implementation adapted from the [official repository](https://github.com/JingyunLiang/SwinIR). Discriminator architecture from [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN).
