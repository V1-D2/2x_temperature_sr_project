#!/usr/bin/env python3
"""
Cascaded 4x Super-Resolution для температурных данных AMSR2
используя обученную SwinIR+Real-ESRGAN модель (2x)
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Существующие модули
from hybrid_model import TemperatureSRModel
from data_preprocessing import TemperatureDataPreprocessor
from config_temperature import *
from utils import calculate_psnr, calculate_ssim
from basicsr.utils import tensor2img, imwrite


class CascadedTemperatureSR:
    """Cascaded Super-Resolution для температурных данных"""

    def __init__(self, model_path: str, device: torch.device = torch.device('cuda')):
        self.device = device
        self.model = self.load_temperature_sr_model(model_path, device)
        self.preprocessor = TemperatureDataPreprocessor()

    def load_temperature_sr_model(self, model_path: str, device: torch.device) -> TemperatureSRModel:
        """Загрузка обученной TemperatureSRModel"""

        # Создаем конфигурацию из config_temperature.py
        opt = {
            'name': name,
            'model_type': model_type,
            'scale': scale,
            'num_gpu': 1,
            'network_g': network_g,
            'network_d': network_d,
            'path': path,
            'train': train,
            'is_train': False,
            'dist': False
        }

        # Создаем модель
        model = TemperatureSRModel(opt)

        # Загружаем checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Определяем формат checkpoint и загружаем веса
        if isinstance(checkpoint, dict) and 'params' in checkpoint:
            model.net_g.load_state_dict(checkpoint['params'], strict=True)
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.net_g.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            model.net_g.load_state_dict(checkpoint, strict=True)

        model.net_g.eval()
        model.net_g.to(device)

        print(f"✓ Модель загружена из {model_path}")
        return model

    def calculate_swinir_patch_size(self, input_shape: Tuple[int, int],
                                    target_patch_size: Tuple[int, int] = (1000, 110)) -> Tuple[int, int]:
        """Рассчитать оптимальный размер патчей для SwinIR"""
        h, w = input_shape
        window_size = 8  # SwinIR window size

        # Ensure patch dimensions are divisible by window_size * scale
        factor = window_size * 2  # scale = 2

        patch_h = (target_patch_size[0] // factor) * factor
        patch_w = (target_patch_size[1] // factor) * factor

        # Ensure patches are not larger than input
        patch_h = min(patch_h, h)
        patch_w = min(patch_w, w)

        return (patch_h, patch_w)

    def extract_patches(self, image: np.ndarray, patch_size: Tuple[int, int],
                        overlap_ratio: float = 0.75) -> List[Dict]:
        """Извлечение патчей с перекрытием"""
        h, w = image.shape
        patch_h, patch_w = patch_size

        # Рассчитываем шаг
        stride_h = int(patch_h * (1 - overlap_ratio))
        stride_w = int(patch_w * (1 - overlap_ratio))

        patches = []

        # Извлекаем патчи
        for y in range(0, h - patch_h + 1, stride_h):
            for x in range(0, w - patch_w + 1, stride_w):
                # Убеждаемся, что не выходим за границы
                if y + patch_h > h:
                    y = h - patch_h
                if x + patch_w > w:
                    x = w - patch_w

                patch = image[y:y + patch_h, x:x + patch_w]

                patches.append({
                    'data': patch,
                    'position': (y, x),
                    'size': (patch_h, patch_w)
                })

        # Добавляем патчи для покрытия краев, если необходимо
        if (h - 1) % stride_h != 0:
            for x in range(0, w - patch_w + 1, stride_w):
                if x + patch_w > w:
                    x = w - patch_w
                patch = image[h - patch_h:h, x:x + patch_w]
                patches.append({
                    'data': patch,
                    'position': (h - patch_h, x),
                    'size': (patch_h, patch_w)
                })

        if (w - 1) % stride_w != 0:
            for y in range(0, h - patch_h + 1, stride_h):
                if y + patch_h > h:
                    y = h - patch_h
                patch = image[y:y + patch_h, w - patch_w:w]
                patches.append({
                    'data': patch,
                    'position': (y, w - patch_w),
                    'size': (patch_h, patch_w)
                })

        # Угловой патч
        if (h - 1) % stride_h != 0 and (w - 1) % stride_w != 0:
            patch = image[h - patch_h:h, w - patch_w:w]
            patches.append({
                'data': patch,
                'position': (h - patch_h, w - patch_w),
                'size': (patch_h, patch_w)
            })

        return patches

    def gaussian_kernel_2d(self, size: int, sigma: float) -> torch.Tensor:
        """Создание 2D Gaussian kernel для blending"""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2

        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        kernel = g.unsqueeze(1) * g.unsqueeze(0)
        return kernel

    def create_blending_mask(self, patch_size: Tuple[int, int],
                             overlap_ratio: float = 0.75) -> torch.Tensor:
        """Создание маски для Gaussian blending"""
        h, w = patch_size

        # Создаем Gaussian маски для каждого измерения
        sigma_h = h * overlap_ratio / 4
        sigma_w = w * overlap_ratio / 4

        mask_h = self.gaussian_kernel_2d(h, sigma_h)
        mask_w = self.gaussian_kernel_2d(w, sigma_w)

        # Для 2D маски используем внешнее произведение
        mask = torch.ones((h, w), dtype=torch.float32)

        # Применяем затухание по краям
        fade_size_h = int(h * overlap_ratio / 2)
        fade_size_w = int(w * overlap_ratio / 2)

        # Верх и низ
        for i in range(fade_size_h):
            weight = i / fade_size_h
            mask[i, :] *= weight
            mask[h - i - 1, :] *= weight

        # Лево и право
        for i in range(fade_size_w):
            weight = i / fade_size_w
            mask[:, i] *= weight
            mask[:, w - i - 1] *= weight

        return mask

    def swinir_patch_based_sr(self, image: np.ndarray,
                              patch_size: Tuple[int, int] = (1000, 110),
                              overlap_ratio: float = 0.75) -> np.ndarray:
        """Patch-based super-resolution используя SwinIR+Real-ESRGAN модель"""

        h, w = image.shape

        # Адаптируем размер патчей под ограничения SwinIR
        patch_size = self.calculate_swinir_patch_size((h, w), patch_size)

        # Извлекаем патчи
        patches = self.extract_patches(image, patch_size, overlap_ratio)

        if len(patches) == 0:
            # Если изображение меньше размера патча, обрабатываем целиком
            patches = [{'data': image, 'position': (0, 0), 'size': image.shape}]

        print(f"  Извлечено {len(patches)} патчей размером {patch_size}")

        # Подготовка выходного изображения и маски весов
        scale_factor = 2
        output_h, output_w = h * scale_factor, w * scale_factor
        output_image = torch.zeros((output_h, output_w), device=self.device)
        weight_map = torch.zeros((output_h, output_w), device=self.device)

        # Создаем маску для blending
        blending_mask = self.create_blending_mask(patch_size, overlap_ratio).to(self.device)

        # Обрабатываем каждый патч
        with torch.no_grad():
            for patch_info in tqdm(patches, desc="  Обработка патчей", leave=False):
                patch_data = patch_info['data']
                y, x = patch_info['position']

                # Нормализация патча
                patch_min = np.min(patch_data)
                patch_max = np.max(patch_data)

                if patch_max > patch_min:
                    patch_norm = (patch_data - patch_min) / (patch_max - patch_min)
                else:
                    patch_norm = np.zeros_like(patch_data)

                # Конвертируем в тензор
                patch_tensor = torch.from_numpy(patch_norm).float()
                patch_tensor = patch_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

                # Super-resolution
                sr_patch = self.model.net_g(patch_tensor)
                sr_patch = torch.clamp(sr_patch, 0, 1)
                sr_patch = sr_patch.squeeze()

                # Денормализация
                sr_patch = sr_patch * (patch_max - patch_min) + patch_min

                # Размер выходного патча
                out_h, out_w = sr_patch.shape
                out_y, out_x = y * scale_factor, x * scale_factor

                # Применяем маску и добавляем к выходному изображению
                if blending_mask.shape != sr_patch.shape:
                    # Resize blending mask if needed
                    blending_mask_resized = F.interpolate(
                        blending_mask.unsqueeze(0).unsqueeze(0),
                        size=sr_patch.shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                else:
                    blending_mask_resized = blending_mask

                output_image[out_y:out_y + out_h, out_x:out_x + out_w] += sr_patch * blending_mask_resized
                weight_map[out_y:out_y + out_h, out_x:out_x + out_w] += blending_mask_resized

        # Нормализуем по весам
        output_image = output_image / (weight_map + 1e-8)

        # Конвертируем обратно в numpy
        return output_image.cpu().numpy()

    def bicubic_upscale_4x(self, image: np.ndarray) -> np.ndarray:
        """Бикубическая интерполяция 4x для baseline"""
        h, w = image.shape

        # Конвертируем в тензор для интерполяции
        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

        # Бикубическая интерполяция
        upscaled = F.interpolate(tensor, size=(h * 4, w * 4),
                                 mode='bicubic', align_corners=False)

        return upscaled.squeeze().numpy()

    def process_single_sample(self, temperature_data: np.ndarray,
                              metadata: Dict,
                              sample_idx: int,
                              patch_size: Tuple[int, int] = (1000, 110),
                              overlap_ratio: float = 0.75) -> Dict:
        """Обработка одного образца температурных данных"""

        print(f"\nОбработка образца {sample_idx + 1}:")
        print(f"  Исходный размер: {temperature_data.shape}")

        # Сохраняем исходные данные
        original = temperature_data.copy()

        # Этап 1: Первое 2x увеличение
        print("  Этап 1: 2x super-resolution...")
        sr_2x = self.swinir_patch_based_sr(original, patch_size, overlap_ratio)
        print(f"  Размер после 1-го этапа: {sr_2x.shape}")

        # Clear GPU memory
        torch.cuda.empty_cache()

        # Этап 2: Второе 2x увеличение (итого 4x)
        print("  Этап 2: еще 2x super-resolution...")
        # Увеличиваем размер патчей для второго прохода
        patch_size_2x = (patch_size[0] * 2, patch_size[1] * 2)
        sr_4x = self.swinir_patch_based_sr(sr_2x, patch_size_2x, overlap_ratio)
        print(f"  Размер после 2-го этапа: {sr_4x.shape}")

        # Создаем 4x bicubic baseline
        print("  Создание 4x bicubic baseline...")
        bicubic_4x = self.bicubic_upscale_4x(original)

        # Вычисляем статистику температур
        def compute_temp_stats(data):
            return {
                'min_temp': float(np.min(data)),
                'max_temp': float(np.max(data)),
                'avg_temp': float(np.mean(data)),
                'std_temp': float(np.std(data))
            }

        result = {
            'original': original,
            'sr_2x': sr_2x,
            'sr_4x': sr_4x,
            'bicubic_4x': bicubic_4x,
            'temperature_stats_original': compute_temp_stats(original),
            'temperature_stats_2x': compute_temp_stats(sr_2x),
            'temperature_stats_4x': compute_temp_stats(sr_4x),
            'temperature_stats_bicubic': compute_temp_stats(bicubic_4x),
            'metadata': {
                'original_shape': original.shape,
                'sr_2x_shape': sr_2x.shape,
                'sr_4x_shape': sr_4x.shape,
                'swath_index': sample_idx,
                'orbit_type': metadata.get('orbit_type', 'unknown'),
                'scale_factor': metadata.get('scale_factor', 1.0),
                'preprocessing_stats': {
                    'original_min': float(np.min(original)),
                    'original_max': float(np.max(original))
                }
            }
        }

        return result

    def create_4panel_comparison(self, result: Dict, save_path: str, sample_idx: int):
        """Создать сравнение: Original | 2x SR | 4x SR | 4x Bicubic"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Общие параметры для всех подграфиков
        vmin = min(result['original'].min(), result['sr_2x'].min(),
                   result['sr_4x'].min(), result['bicubic_4x'].min())
        vmax = max(result['original'].max(), result['sr_2x'].max(),
                   result['sr_4x'].max(), result['bicubic_4x'].max())

        # Original
        im1 = axes[0, 0].imshow(result['original'], cmap='turbo',
                                vmin=vmin, vmax=vmax, aspect='auto')
        axes[0, 0].set_title(f'Original {result["original"].shape}')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

        # 2x SR
        im2 = axes[0, 1].imshow(result['sr_2x'], cmap='turbo',
                                vmin=vmin, vmax=vmax, aspect='auto')
        axes[0, 1].set_title(f'2x SR {result["sr_2x"].shape}')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

        # 4x SR
        im3 = axes[1, 0].imshow(result['sr_4x'], cmap='turbo',
                                vmin=vmin, vmax=vmax, aspect='auto')
        axes[1, 0].set_title(f'4x SR (Cascaded) {result["sr_4x"].shape}')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

        # 4x Bicubic
        im4 = axes[1, 1].imshow(result['bicubic_4x'], cmap='turbo',
                                vmin=vmin, vmax=vmax, aspect='auto')
        axes[1, 1].set_title(f'4x Bicubic {result["bicubic_4x"].shape}')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

        # Добавляем температурную статистику
        stats_text = f"Sample {sample_idx + 1} Temperature Statistics:\n\n"
        stats_text += f"Original: [{result['temperature_stats_original']['min_temp']:.1f}, "
        stats_text += f"{result['temperature_stats_original']['max_temp']:.1f}] K\n"
        stats_text += f"2x SR: [{result['temperature_stats_2x']['min_temp']:.1f}, "
        stats_text += f"{result['temperature_stats_2x']['max_temp']:.1f}] K\n"
        stats_text += f"4x SR: [{result['temperature_stats_4x']['min_temp']:.1f}, "
        stats_text += f"{result['temperature_stats_4x']['max_temp']:.1f}] K\n"
        stats_text += f"4x Bicubic: [{result['temperature_stats_bicubic']['min_temp']:.1f}, "
        stats_text += f"{result['temperature_stats_bicubic']['max_temp']:.1f}] K"

        fig.text(0.02, 0.02, stats_text, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

        plt.suptitle(f'Cascaded Temperature Super-Resolution - Sample {sample_idx + 1}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'comparison_sample_{sample_idx + 1:02d}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Cleanup
        del fig, axes
        gc.collect()

    def save_individual_results(self, result: Dict, save_path: str, sample_idx: int):
        """Сохранить индивидуальные изображения в полном разрешении"""
        sample_dir = os.path.join(save_path, f'sample_{sample_idx + 1:02d}')
        os.makedirs(sample_dir, exist_ok=True)

        # Функция для сохранения одного изображения
        def save_temperature_image(data, filename, title):
            plt.figure(figsize=(12, 8))
            plt.imshow(data, cmap='turbo', aspect='auto')
            plt.colorbar(label='Temperature (K)')
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()

        # Сохраняем все результаты
        save_temperature_image(result['original'], 'original.png',
                               f'Original {result["original"].shape}')
        save_temperature_image(result['sr_2x'], 'sr_2x.png',
                               f'2x SR {result["sr_2x"].shape}')
        save_temperature_image(result['sr_4x'], 'sr_4x.png',
                               f'4x SR Cascaded {result["sr_4x"].shape}')
        save_temperature_image(result['bicubic_4x'], 'bicubic_4x.png',
                               f'4x Bicubic {result["bicubic_4x"].shape}')

        # Сохраняем также как grayscale для BasicSR compatibility
        for name, data in [('original', result['original']),
                           ('sr_2x', result['sr_2x']),
                           ('sr_4x', result['sr_4x']),
                           ('bicubic_4x', result['bicubic_4x'])]:
            # Нормализуем в [0, 1] для tensor2img
            data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
            tensor = torch.from_numpy(data_norm).float().unsqueeze(0)
            img = tensor2img([tensor])
            imwrite(img, os.path.join(sample_dir, f'{name}_grayscale.png'))

        # Cleanup
        gc.collect()


def cascaded_temperature_sr_4x(npz_dir: str,
                               model_path: str,
                               num_samples: int = 5,
                               output_dir: str = "cascaded_swinir_results") -> List[Dict]:
    """
    Cascaded 4x super-resolution для температурных данных
    """

    # Создаем выходную директорию
    os.makedirs(output_dir, exist_ok=True)

    # Инициализируем систему
    print("=" * 60)
    print("Cascaded Temperature Super-Resolution 4x")
    print("=" * 60)
    print(f"Модель: {model_path}")
    print(f"Директория данных: {npz_dir}")
    print(f"Выходная директория: {output_dir}")
    print(f"Количество образцов: {num_samples}")

    # Создаем cascaded SR систему
    sr_system = CascadedTemperatureSR(model_path)

    # Находим последний NPZ файл
    npz_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
    if not npz_files:
        raise ValueError(f"Не найдены NPZ файлы в {npz_dir}")

    last_npz = os.path.join(npz_dir, npz_files[-1])
    print(f"\nИспользуется файл: {last_npz}")

    # Загружаем данные
    print("\nЗагрузка данных...")
    data = np.load(last_npz, allow_pickle=True)

    # Проверяем формат данных
    if 'swaths' in data:
        swaths = data['swaths']
    elif 'swath_array' in data:
        swaths = data['swath_array']
    else:
        # Если это одиночный файл
        temperature = data['temperature'].astype(np.float32)
        metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
        swaths = [{'temperature': temperature, 'metadata': metadata}]

    # Ограничиваем количество образцов
    actual_samples = min(num_samples, len(swaths))
    print(f"Доступно образцов: {len(swaths)}, будет обработано: {actual_samples}")

    # Обрабатываем образцы
    results = []

    for i in range(actual_samples):
        swath = swaths[i]
        temp_data = swath['temperature'].astype(np.float32)
        metadata = swath.get('metadata', {})

        # Предобработка
        temp_data = sr_system.preprocessor.crop_or_pad(temp_data)

        # Обработка
        result = sr_system.process_single_sample(
            temp_data,
            metadata,
            i,
            patch_size=(1000, 110),
            overlap_ratio=0.75
        )

        results.append(result)

        # Сохранение визуализаций
        print(f"\nСохранение результатов для образца {i + 1}...")

        # 4-panel comparison
        sr_system.create_4panel_comparison(result, output_dir, i)

        # Индивидуальные изображения
        sr_system.save_individual_results(result, output_dir, i)

        # NPZ файл с результатами
        npz_path = os.path.join(output_dir, f'result_sample_{i + 1:02d}.npz')
        np.savez(npz_path, **result)

        # Очистка памяти
        torch.cuda.empty_cache()
        gc.collect()

    # Сохраняем общую статистику
    stats_path = os.path.join(output_dir, 'cascaded_sr_statistics.txt')
    with open(stats_path, 'w') as f:
        f.write("Cascaded Temperature Super-Resolution 4x Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Data: {last_npz}\n")
        f.write(f"Processed samples: {actual_samples}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for i, result in enumerate(results):
            f.write(f"\nSample {i + 1}:\n")
            f.write(f"  Original shape: {result['metadata']['original_shape']}\n")
            f.write(f"  2x SR shape: {result['metadata']['sr_2x_shape']}\n")
            f.write(f"  4x SR shape: {result['metadata']['sr_4x_shape']}\n")
            f.write(f"  Temperature range (4x SR): [{result['temperature_stats_4x']['min_temp']:.1f}, "
                    f"{result['temperature_stats_4x']['max_temp']:.1f}] K\n")
            f.write(f"  Average temperature (4x SR): {result['temperature_stats_4x']['avg_temp']:.1f} K\n")

    print("\n" + "=" * 60)
    print("Обработка завершена!")
    print(f"Результаты сохранены в: {output_dir}")
    print("=" * 60)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Cascaded 4x Temperature Super-Resolution')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained 2x SR model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing NPZ files')
    parser.add_argument('--output_dir', type=str, default='./cascaded_swinir_results',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to process')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[1000, 110],
                        help='Patch size for processing (height width)')
    parser.add_argument('--overlap_ratio', type=float, default=0.75,
                        help='Overlap ratio for patches')
    return parser.parse_args()


def main():
    args = parse_args()

    # Запускаем cascaded super-resolution
    results = cascaded_temperature_sr_4x(
        npz_dir=args.data_dir,
        model_path=args.model_path,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )

    # Выводим итоговую информацию
    print("\nИтоговая статистика:")
    for i, result in enumerate(results):
        print(f"\nSample {i + 1}:")
        print(f"  Original: {result['original'].shape}")
        print(f"  2x SR: {result['sr_2x'].shape}")
        print(f"  4x SR: {result['sr_4x'].shape}")
        print(f"  Temperature range 4x: [{result['temperature_stats_4x']['min_temp']:.1f}, "
              f"{result['temperature_stats_4x']['max_temp']:.1f}] K")


if __name__ == "__main__":
    main()