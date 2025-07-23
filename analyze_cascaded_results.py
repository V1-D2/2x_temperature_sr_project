#!/usr/bin/env python3
"""
Анализ результатов Cascaded Temperature Super-Resolution
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd


def load_cascaded_results(results_dir: str) -> List[Dict]:
    """Загрузка всех результатов из директории"""
    results = []

    # Находим все NPZ файлы с результатами
    npz_files = sorted([f for f in os.listdir(results_dir)
                        if f.startswith('result_sample_') and f.endswith('.npz')])

    for npz_file in npz_files:
        data = np.load(os.path.join(results_dir, npz_file), allow_pickle=True)
        results.append({
            'filename': npz_file,
            'data': dict(data)
        })
        data.close()

    return results


def analyze_temperature_preservation(results: List[Dict]) -> pd.DataFrame:
    """Анализ сохранения температурных характеристик"""

    analysis_data = []

    for i, result in enumerate(results):
        data = result['data']

        # Извлекаем статистику
        stats_orig = data['temperature_stats_original'].item()
        stats_2x = data['temperature_stats_2x'].item()
        stats_4x = data['temperature_stats_4x'].item()
        stats_bicubic = data['temperature_stats_bicubic'].item()

        # Вычисляем отклонения
        analysis_data.append({
            'sample': i + 1,
            'orig_min': stats_orig['min_temp'],
            'orig_max': stats_orig['max_temp'],
            'orig_avg': stats_orig['avg_temp'],
            'sr4x_min': stats_4x['min_temp'],
            'sr4x_max': stats_4x['max_temp'],
            'sr4x_avg': stats_4x['avg_temp'],
            'bicubic_min': stats_bicubic['min_temp'],
            'bicubic_max': stats_bicubic['max_temp'],
            'bicubic_avg': stats_bicubic['avg_temp'],
            'sr4x_min_diff': stats_4x['min_temp'] - stats_orig['min_temp'],
            'sr4x_max_diff': stats_4x['max_temp'] - stats_orig['max_temp'],
            'sr4x_avg_diff': stats_4x['avg_temp'] - stats_orig['avg_temp'],
            'bicubic_min_diff': stats_bicubic['min_temp'] - stats_orig['min_temp'],
            'bicubic_max_diff': stats_bicubic['max_temp'] - stats_orig['max_temp'],
            'bicubic_avg_diff': stats_bicubic['avg_temp'] - stats_orig['avg_temp'],
        })

    return pd.DataFrame(analysis_data)


def create_quality_comparison_plot(results: List[Dict], save_path: str):
    """Создание графика сравнения качества SR vs Bicubic"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    samples = []
    sr_min_diffs = []
    sr_max_diffs = []
    sr_avg_diffs = []
    bicubic_min_diffs = []
    bicubic_max_diffs = []
    bicubic_avg_diffs = []

    for i, result in enumerate(results):
        data = result['data']
        stats_orig = data['temperature_stats_original'].item()
        stats_4x = data['temperature_stats_4x'].item()
        stats_bicubic = data['temperature_stats_bicubic'].item()

        samples.append(i + 1)
        sr_min_diffs.append(abs(stats_4x['min_temp'] - stats_orig['min_temp']))
        sr_max_diffs.append(abs(stats_4x['max_temp'] - stats_orig['max_temp']))
        sr_avg_diffs.append(abs(stats_4x['avg_temp'] - stats_orig['avg_temp']))
        bicubic_min_diffs.append(abs(stats_bicubic['min_temp'] - stats_orig['min_temp']))
        bicubic_max_diffs.append(abs(stats_bicubic['max_temp'] - stats_orig['max_temp']))
        bicubic_avg_diffs.append(abs(stats_bicubic['avg_temp'] - stats_orig['avg_temp']))

    # График 1: Отклонение минимальных температур
    ax = axes[0, 0]
    x = np.arange(len(samples))
    width = 0.35
    ax.bar(x - width / 2, sr_min_diffs, width, label='SR 4x', color='blue', alpha=0.7)
    ax.bar(x + width / 2, bicubic_min_diffs, width, label='Bicubic 4x', color='red', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Absolute Difference (K)')
    ax.set_title('Minimum Temperature Preservation')
    ax.set_xticks(x)
    ax.set_xticklabels(samples)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # График 2: Отклонение максимальных температур
    ax = axes[0, 1]
    ax.bar(x - width / 2, sr_max_diffs, width, label='SR 4x', color='blue', alpha=0.7)
    ax.bar(x + width / 2, bicubic_max_diffs, width, label='Bicubic 4x', color='red', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Absolute Difference (K)')
    ax.set_title('Maximum Temperature Preservation')
    ax.set_xticks(x)
    ax.set_xticklabels(samples)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # График 3: Отклонение средних температур
    ax = axes[1, 0]
    ax.bar(x - width / 2, sr_avg_diffs, width, label='SR 4x', color='blue', alpha=0.7)
    ax.bar(x + width / 2, bicubic_avg_diffs, width, label='Bicubic 4x', color='red', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Absolute Difference (K)')
    ax.set_title('Average Temperature Preservation')
    ax.set_xticks(x)
    ax.set_xticklabels(samples)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # График 4: Общая статистика
    ax = axes[1, 1]
    metrics = ['Min Temp', 'Max Temp', 'Avg Temp']
    sr_means = [np.mean(sr_min_diffs), np.mean(sr_max_diffs), np.mean(sr_avg_diffs)]
    bicubic_means = [np.mean(bicubic_min_diffs), np.mean(bicubic_max_diffs), np.mean(bicubic_avg_diffs)]

    x = np.arange(len(metrics))
    ax.bar(x - width / 2, sr_means, width, label='SR 4x', color='blue', alpha=0.7)
    ax.bar(x + width / 2, bicubic_means, width, label='Bicubic 4x', color='red', alpha=0.7)
    ax.set_xlabel('Metric')
    ax.set_ylabel('Mean Absolute Difference (K)')
    ax.set_title('Overall Temperature Preservation')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Добавляем процентное улучшение
    for i, (sr, bic) in enumerate(zip(sr_means, bicubic_means)):
        improvement = (bic - sr) / bic * 100
        ax.text(i, max(sr, bic) + 0.5, f'{improvement:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Cascaded SR 4x vs Bicubic 4x: Temperature Preservation Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'quality_comparison_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def create_detail_enhancement_visualization(results: List[Dict], save_path: str):
    """Визуализация улучшения деталей"""

    # Выбираем первый образец для детального анализа
    if not results:
        return

    data = results[0]['data']
    original = data['original']
    sr_4x = data['sr_4x']
    bicubic_4x = data['bicubic_4x']

    # Вычисляем градиенты для оценки резкости
    def compute_gradient_magnitude(img):
        gy, gx = np.gradient(img)
        return np.sqrt(gx ** 2 + gy ** 2)

    grad_original = compute_gradient_magnitude(original)
    grad_sr = compute_gradient_magnitude(sr_4x)
    grad_bicubic = compute_gradient_magnitude(bicubic_4x)

    # Создаем визуализацию
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Температурные карты
    im1 = axes[0, 0].imshow(original, cmap='turbo', aspect='auto')
    axes[0, 0].set_title(f'Original {original.shape}')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    im2 = axes[0, 1].imshow(sr_4x, cmap='turbo', aspect='auto')
    axes[0, 1].set_title(f'SR 4x {sr_4x.shape}')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    im3 = axes[0, 2].imshow(bicubic_4x, cmap='turbo', aspect='auto')
    axes[0, 2].set_title(f'Bicubic 4x {bicubic_4x.shape}')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # Градиентные карты
    vmax = max(grad_original.max(), grad_sr.max(), grad_bicubic.max())

    im4 = axes[1, 0].imshow(grad_original, cmap='hot', vmax=vmax, aspect='auto')
    axes[1, 0].set_title('Original Gradients')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

    im5 = axes[1, 1].imshow(grad_sr, cmap='hot', vmax=vmax, aspect='auto')
    axes[1, 1].set_title('SR 4x Gradients')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

    im6 = axes[1, 2].imshow(grad_bicubic, cmap='hot', vmax=vmax, aspect='auto')
    axes[1, 2].set_title('Bicubic 4x Gradients')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

    # Добавляем метрики резкости
    sharpness_original = np.mean(grad_original)
    sharpness_sr = np.mean(grad_sr)
    sharpness_bicubic = np.mean(grad_bicubic)

    metrics_text = f"Average Gradient Magnitude (Sharpness):\n"
    metrics_text += f"Original: {sharpness_original:.3f}\n"
    metrics_text += f"SR 4x: {sharpness_sr:.3f} ({sharpness_sr / sharpness_original:.1%} of original)\n"
    metrics_text += f"Bicubic 4x: {sharpness_bicubic:.3f} ({sharpness_bicubic / sharpness_original:.1%} of original)"

    fig.text(0.02, 0.02, metrics_text, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))

    plt.suptitle('Detail Enhancement Analysis: Temperature Gradients',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'detail_enhancement_analysis.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(results_dir: str):
    """Генерация полного отчета о результатах"""

    print("=" * 60)
    print("Cascaded Temperature SR 4x - Analysis Report")
    print("=" * 60)

    # Загружаем результаты
    results = load_cascaded_results(results_dir)
    print(f"\nНайдено образцов: {len(results)}")

    # Анализ температур
    df = analyze_temperature_preservation(results)

    print("\nТемпературная статистика:")
    print("-" * 40)
    print(df.to_string(index=False))

    print("\nСредние отклонения от оригинала:")
    print("-" * 40)
    print(f"SR 4x:")
    print(f"  Min temp diff: {df['sr4x_min_diff'].abs().mean():.2f} K")
    print(f"  Max temp diff: {df['sr4x_max_diff'].abs().mean():.2f} K")
    print(f"  Avg temp diff: {df['sr4x_avg_diff'].abs().mean():.2f} K")
    print(f"\nBicubic 4x:")
    print(f"  Min temp diff: {df['bicubic_min_diff'].abs().mean():.2f} K")
    print(f"  Max temp diff: {df['bicubic_max_diff'].abs().mean():.2f} K")
    print(f"  Avg temp diff: {df['bicubic_avg_diff'].abs().mean():.2f} K")

    # Процентное улучшение
    sr_total_diff = (df['sr4x_min_diff'].abs().mean() +
                     df['sr4x_max_diff'].abs().mean() +
                     df['sr4x_avg_diff'].abs().mean()) / 3

    bicubic_total_diff = (df['bicubic_min_diff'].abs().mean() +
                          df['bicubic_max_diff'].abs().mean() +
                          df['bicubic_avg_diff'].abs().mean()) / 3

    improvement = (bicubic_total_diff - sr_total_diff) / bicubic_total_diff * 100

    print(f"\nОбщее улучшение SR 4x над Bicubic 4x: {improvement:.1f}%")

    # Создаем визуализации
    print("\nСоздание аналитических графиков...")
    create_quality_comparison_plot(results, results_dir)
    create_detail_enhancement_visualization(results, results_dir)

    # Сохраняем отчет
    report_path = os.path.join(results_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("Cascaded Temperature SR 4x - Analysis Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total samples analyzed: {len(results)}\n\n")
        f.write("Temperature Statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\nAverage Deviations from Original:\n")
        f.write("-" * 40 + "\n")
        f.write(f"SR 4x:\n")
        f.write(f"  Min temp diff: {df['sr4x_min_diff'].abs().mean():.2f} K\n")
        f.write(f"  Max temp diff: {df['sr4x_max_diff'].abs().mean():.2f} K\n")
        f.write(f"  Avg temp diff: {df['sr4x_avg_diff'].abs().mean():.2f} K\n")
        f.write(f"\nBicubic 4x:\n")
        f.write(f"  Min temp diff: {df['bicubic_min_diff'].abs().mean():.2f} K\n")
        f.write(f"  Max temp diff: {df['bicubic_max_diff'].abs().mean():.2f} K\n")
        f.write(f"  Avg temp diff: {df['bicubic_avg_diff'].abs().mean():.2f} K\n")
        f.write(f"\nOverall improvement of SR 4x over Bicubic 4x: {improvement:.1f}%\n")

    print(f"\nОтчет сохранен в: {report_path}")
    print("Графики сохранены:")
    print(f"  - {os.path.join(results_dir, 'quality_comparison_analysis.png')}")
    print(f"  - {os.path.join(results_dir, 'detail_enhancement_analysis.png')}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Cascaded SR Results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing cascaded SR results')
    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return

    generate_report(args.results_dir)


if __name__ == "__main__":
    main()