#!/usr/bin/env python3
"""
Unified Analysis of Cascaded Temperature Super-Resolution Results
Analyzes both 4x and 8x results with comprehensive metrics
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_cascaded_results(results_dir: str) -> Dict[str, List[Dict]]:
    """Load all results from cascaded SR directory"""
    results = {
        '4x': [],
        '8x': []
    }

    # Load 4x results
    results_4x_dir = os.path.join(results_dir, 'results_4x', 'arrays')
    if os.path.exists(results_4x_dir):
        npz_files = sorted([f for f in os.listdir(results_4x_dir) if f.endswith('.npz')])
        for npz_file in npz_files:
            data = np.load(os.path.join(results_4x_dir, npz_file), allow_pickle=True)
            results['4x'].append({
                'filename': npz_file,
                'data': dict(data)
            })
            data.close()

    # Load 8x results
    results_8x_dir = os.path.join(results_dir, 'results_8x', 'arrays')
    if os.path.exists(results_8x_dir):
        npz_files = sorted([f for f in os.listdir(results_8x_dir) if f.endswith('.npz')])
        for npz_file in npz_files:
            data = np.load(os.path.join(results_8x_dir, npz_file), allow_pickle=True)
            results['8x'].append({
                'filename': npz_file,
                'data': dict(data)
            })
            data.close()

    return results


def analyze_temperature_preservation(results: Dict[str, List[Dict]]) -> Dict[str, pd.DataFrame]:
    """Analyze temperature preservation for both 4x and 8x"""

    analysis_results = {}

    for variant in ['4x', '8x']:
        if not results[variant]:
            continue

        analysis_data = []

        for i, result in enumerate(results[variant]):
            data = result['data']

            # Extract arrays
            original = data['original']
            sr = data['sr']
            bicubic = data['bicubic']

            # Calculate statistics
            analysis_data.append({
                'sample': i + 1,
                'orig_min': np.min(original),
                'orig_max': np.max(original),
                'orig_avg': np.mean(original),
                'orig_std': np.std(original),
                'sr_min': np.min(sr),
                'sr_max': np.max(sr),
                'sr_avg': np.mean(sr),
                'sr_std': np.std(sr),
                'bicubic_min': np.min(bicubic),
                'bicubic_max': np.max(bicubic),
                'bicubic_avg': np.mean(bicubic),
                'bicubic_std': np.std(bicubic),
                'sr_min_diff': np.min(sr) - np.min(original),
                'sr_max_diff': np.max(sr) - np.max(original),
                'sr_avg_diff': np.mean(sr) - np.mean(original),
                'bicubic_min_diff': np.min(bicubic) - np.min(original),
                'bicubic_max_diff': np.max(bicubic) - np.max(original),
                'bicubic_avg_diff': np.mean(bicubic) - np.mean(original),
            })

        analysis_results[variant] = pd.DataFrame(analysis_data)

    return analysis_results


def create_quality_comparison_plot(results: Dict[str, List[Dict]], save_path: str):
    """Create quality comparison plots for both 4x and 8x"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for row, variant in enumerate(['4x', '8x']):
        if not results[variant]:
            continue

        samples = []
        sr_min_diffs = []
        sr_max_diffs = []
        sr_avg_diffs = []
        bicubic_min_diffs = []
        bicubic_max_diffs = []
        bicubic_avg_diffs = []

        for i, result in enumerate(results[variant]):
            data = result['data']
            original = data['original']
            sr = data['sr']
            bicubic = data['bicubic']

            samples.append(i + 1)
            sr_min_diffs.append(abs(np.min(sr) - np.min(original)))
            sr_max_diffs.append(abs(np.max(sr) - np.max(original)))
            sr_avg_diffs.append(abs(np.mean(sr) - np.mean(original)))
            bicubic_min_diffs.append(abs(np.min(bicubic) - np.min(original)))
            bicubic_max_diffs.append(abs(np.max(bicubic) - np.max(original)))
            bicubic_avg_diffs.append(abs(np.mean(bicubic) - np.mean(original)))

        x = np.arange(len(samples))
        width = 0.35

        # Minimum temperature preservation
        ax = axes[row, 0]
        ax.bar(x - width / 2, sr_min_diffs, width, label=f'SR {variant}', color='blue', alpha=0.7)
        ax.bar(x + width / 2, bicubic_min_diffs, width, label=f'Bicubic {variant}', color='red', alpha=0.7)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Absolute Difference (K)')
        ax.set_title(f'{variant} - Minimum Temperature Preservation')
        ax.set_xticks(x)
        ax.set_xticklabels(samples)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Maximum temperature preservation
        ax = axes[row, 1]
        ax.bar(x - width / 2, sr_max_diffs, width, label=f'SR {variant}', color='blue', alpha=0.7)
        ax.bar(x + width / 2, bicubic_max_diffs, width, label=f'Bicubic {variant}', color='red', alpha=0.7)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Absolute Difference (K)')
        ax.set_title(f'{variant} - Maximum Temperature Preservation')
        ax.set_xticks(x)
        ax.set_xticklabels(samples)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Average temperature preservation
        ax = axes[row, 2]
        ax.bar(x - width / 2, sr_avg_diffs, width, label=f'SR {variant}', color='blue', alpha=0.7)
        ax.bar(x + width / 2, bicubic_avg_diffs, width, label=f'Bicubic {variant}', color='red', alpha=0.7)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Absolute Difference (K)')
        ax.set_title(f'{variant} - Average Temperature Preservation')
        ax.set_xticks(x)
        ax.set_xticklabels(samples)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add improvement percentages
        for ax_idx, (sr_diffs, bic_diffs) in enumerate([
            (sr_min_diffs, bicubic_min_diffs),
            (sr_max_diffs, bicubic_max_diffs),
            (sr_avg_diffs, bicubic_avg_diffs)
        ]):
            ax = axes[row, ax_idx]
            for i, (sr, bic) in enumerate(zip(sr_diffs, bic_diffs)):
                if bic > 0:
                    improvement = (bic - sr) / bic * 100
                    ax.text(i, max(sr, bic) + 0.5, f'{improvement:.1f}%',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Temperature Preservation Analysis: 4x and 8x Super-Resolution',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'quality_comparison_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def create_sharpness_analysis(results: Dict[str, List[Dict]], save_path: str):
    """Analyze edge sharpness and detail enhancement"""

    def compute_gradient_magnitude(img):
        gy, gx = np.gradient(img)
        return np.sqrt(gx ** 2 + gy ** 2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sharpness_data = {'4x': {'sr': [], 'bicubic': []},
                      '8x': {'sr': [], 'bicubic': []}}

    for variant in ['4x', '8x']:
        if not results[variant]:
            continue

        for result in results[variant]:
            data = result['data']
            original = data['original']
            sr = data['sr']
            bicubic = data['bicubic']

            # Calculate sharpness metrics
            grad_original = compute_gradient_magnitude(original)
            grad_sr = compute_gradient_magnitude(sr)
            grad_bicubic = compute_gradient_magnitude(bicubic)

            # Normalize by original sharpness
            sharpness_sr = np.mean(grad_sr) / np.mean(grad_original)
            sharpness_bicubic = np.mean(grad_bicubic) / np.mean(grad_original)

            sharpness_data[variant]['sr'].append(sharpness_sr)
            sharpness_data[variant]['bicubic'].append(sharpness_bicubic)

    # Plot sharpness comparison
    for idx, variant in enumerate(['4x', '8x']):
        if not sharpness_data[variant]['sr']:
            continue

        ax = axes[0, idx]
        x = np.arange(len(sharpness_data[variant]['sr']))
        width = 0.35

        ax.bar(x - width / 2, sharpness_data[variant]['sr'], width,
               label=f'SR {variant}', color='green', alpha=0.7)
        ax.bar(x + width / 2, sharpness_data[variant]['bicubic'], width,
               label=f'Bicubic {variant}', color='orange', alpha=0.7)

        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Original')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Relative Sharpness')
        ax.set_title(f'{variant} - Edge Sharpness (Higher is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot average sharpness
    ax = axes[1, 0]
    variants = []
    sr_means = []
    bicubic_means = []

    for variant in ['4x', '8x']:
        if sharpness_data[variant]['sr']:
            variants.append(variant)
            sr_means.append(np.mean(sharpness_data[variant]['sr']))
            bicubic_means.append(np.mean(sharpness_data[variant]['bicubic']))

    x = np.arange(len(variants))
    width = 0.35

    ax.bar(x - width / 2, sr_means, width, label='SR', color='green', alpha=0.7)
    ax.bar(x + width / 2, bicubic_means, width, label='Bicubic', color='orange', alpha=0.7)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Original')

    ax.set_xlabel('Variant')
    ax.set_ylabel('Average Relative Sharpness')
    ax.set_title('Average Edge Sharpness Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add percentage improvements
    for i, (sr, bic) in enumerate(zip(sr_means, bicubic_means)):
        improvement = (sr - bic) / bic * 100
        ax.text(i, max(sr, bic) + 0.05, f'{improvement:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[1, 1].axis('off')

    plt.suptitle('Sharpness and Detail Enhancement Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sharpness_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def generate_comprehensive_report(results_dir: str):
    """Generate comprehensive analysis report"""

    logger.info("=" * 80)
    logger.info("CASCADED TEMPERATURE SR - COMPREHENSIVE ANALYSIS")
    logger.info("=" * 80)

    # Load results
    results = load_cascaded_results(results_dir)
    logger.info(f"Found {len(results['4x'])} 4x samples and {len(results['8x'])} 8x samples")

    # Analyze temperature preservation
    df_analysis = analyze_temperature_preservation(results)

    # Print temperature statistics
    logger.info("\nTEMPERATURE STATISTICS:")
    logger.info("-" * 60)

    for variant in ['4x', '8x']:
        if variant in df_analysis:
            df = df_analysis[variant]
            logger.info(f"\n{variant} Super-Resolution:")
            logger.info(f"  SR average temperature difference: {df['sr_avg_diff'].abs().mean():.2f} K")
            logger.info(f"  SR min temperature difference: {df['sr_min_diff'].abs().mean():.2f} K")
            logger.info(f"  SR max temperature difference: {df['sr_max_diff'].abs().mean():.2f} K")
            logger.info(f"  Bicubic average temperature difference: {df['bicubic_avg_diff'].abs().mean():.2f} K")
            logger.info(f"  Bicubic min temperature difference: {df['bicubic_min_diff'].abs().mean():.2f} K")
            logger.info(f"  Bicubic max temperature difference: {df['bicubic_max_diff'].abs().mean():.2f} K")

            # Calculate improvements
            sr_total_diff = (df['sr_min_diff'].abs().mean() +
                             df['sr_max_diff'].abs().mean() +
                             df['sr_avg_diff'].abs().mean()) / 3
            bicubic_total_diff = (df['bicubic_min_diff'].abs().mean() +
                                  df['bicubic_max_diff'].abs().mean() +
                                  df['bicubic_avg_diff'].abs().mean()) / 3

            if bicubic_total_diff > 0:
                improvement = (bicubic_total_diff - sr_total_diff) / bicubic_total_diff * 100
                logger.info(f"  Overall improvement over bicubic: {improvement:.1f}%")

    # Create visualizations
    logger.info("\nCreating analysis visualizations...")
    create_quality_comparison_plot(results, results_dir)
    create_sharpness_analysis(results, results_dir)

    # Save detailed report
    report_path = os.path.join(results_dir, 'comprehensive_analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CASCADED TEMPERATURE SR - COMPREHENSIVE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        for variant in ['4x', '8x']:
            if variant not in df_analysis:
                continue

            df = df_analysis[variant]
            f.write(f"\n{variant} SUPER-RESOLUTION ANALYSIS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total samples analyzed: {len(df)}\n\n")

            f.write("Temperature Statistics:\n")
            f.write(df.to_string(index=False))
            f.write("\n\nAverage Deviations from Original:\n")
            f.write(f"  SR {variant}:\n")
            f.write(f"    Min temp diff: {df['sr_min_diff'].abs().mean():.2f} K\n")
            f.write(f"    Max temp diff: {df['sr_max_diff'].abs().mean():.2f} K\n")
            f.write(f"    Avg temp diff: {df['sr_avg_diff'].abs().mean():.2f} K\n")
            f.write(f"  Bicubic {variant}:\n")
            f.write(f"    Min temp diff: {df['bicubic_min_diff'].abs().mean():.2f} K\n")
            f.write(f"    Max temp diff: {df['bicubic_max_diff'].abs().mean():.2f} K\n")
            f.write(f"    Avg temp diff: {df['bicubic_avg_diff'].abs().mean():.2f} K\n")

            # Calculate improvement
            sr_total = (df['sr_min_diff'].abs().mean() +
                        df['sr_max_diff'].abs().mean() +
                        df['sr_avg_diff'].abs().mean()) / 3
            bicubic_total = (df['bicubic_min_diff'].abs().mean() +
                             df['bicubic_max_diff'].abs().mean() +
                             df['bicubic_avg_diff'].abs().mean()) / 3

            if bicubic_total > 0:
                improvement = (bicubic_total - sr_total) / bicubic_total * 100
                f.write(f"\nOverall improvement of SR {variant} over Bicubic: {improvement:.1f}%\n")

    logger.info(f"\nReport saved to: {report_path}")
    logger.info("Visualizations saved:")
    logger.info(f"  - {os.path.join(results_dir, 'quality_comparison_analysis.png')}")
    logger.info(f"  - {os.path.join(results_dir, 'sharpness_analysis.png')}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Cascaded SR Results')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing cascaded SR results')
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        logger.error(f"Results directory not found: {args.results_dir}")
        return

    generate_comprehensive_report(args.results_dir)
    logger.info("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()