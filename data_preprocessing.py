import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Dict, Tuple, List, Optional
import gc


class TemperatureDataPreprocessor:
    """Препроцессор для температурных данных AMSR-2"""

    def __init__(self, target_height: int = 2000, target_width: int = 220):
        self.target_height = target_height
        self.target_width = target_width
        self.stats = {'min': float('inf'), 'max': float('-inf'), 'mean': 0, 'std': 0}
        self.n_samples = 0

    def normalize_temperature(self, temp_array: np.ndarray, preserve_relative: bool = True) -> np.ndarray:
        """Нормализация температур с сохранением физического смысла"""
        # Удаляем NaN и заменяем на среднее
        mask = np.isnan(temp_array)
        if mask.any():
            mean_val = np.nanmean(temp_array)
            temp_array[mask] = mean_val

        if preserve_relative:
            # Нормализация в диапазон [0, 1] с сохранением относительных значений
            min_temp = np.min(temp_array)
            max_temp = np.max(temp_array)

            # Обновляем глобальную статистику
            self.stats['min'] = min(self.stats['min'], min_temp)
            self.stats['max'] = max(self.stats['max'], max_temp)

            # Нормализация
            if max_temp > min_temp:
                normalized = (temp_array - min_temp) / (max_temp - min_temp)
            else:
                normalized = np.zeros_like(temp_array)

            return normalized.astype(np.float32)
        else:
            # Стандартная нормализация (z-score)
            mean = np.mean(temp_array)
            std = np.std(temp_array)
            if std > 0:
                return ((temp_array - mean) / std).astype(np.float32)
            else:
                return np.zeros_like(temp_array, dtype=np.float32)

    def denormalize_temperature(self, normalized: np.ndarray, original_min: float, original_max: float) -> np.ndarray:
        """Обратная нормализация для восстановления физических значений"""
        return normalized * (original_max - original_min) + original_min

    def crop_or_pad(self, temp_array: np.ndarray) -> np.ndarray:
        """Обрезка или паддинг до целевого размера"""
        h, w = temp_array.shape

        # Обрезка по высоте
        if h > self.target_height:
            start_h = (h - self.target_height) // 2
            temp_array = temp_array[start_h:start_h + self.target_height, :]
        elif h < self.target_height:
            pad_h = self.target_height - h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            temp_array = np.pad(temp_array, ((pad_top, pad_bottom), (0, 0)), mode='edge')

        # Обрезка по ширине
        h, w = temp_array.shape
        if w > self.target_width:
            start_w = (w - self.target_width) // 2
            temp_array = temp_array[:, start_w:start_w + self.target_width]
        elif w < self.target_width:
            pad_w = self.target_width - w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            temp_array = np.pad(temp_array, ((0, 0), (pad_left, pad_right)), mode='edge')

        return temp_array

    def create_lr_hr_pair(self, hr_temp: np.ndarray, scale_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Создание пары низкое-высокое разрешение для обучения"""
        h, w = hr_temp.shape

        # Ensure dimensions are divisible by scale_factor * window_size
        window_size = 8  # SwinIR window size
        factor = scale_factor * window_size

        # Crop HR to nearest smaller size divisible by factor
        new_h = (h // factor) * factor
        new_w = (w // factor) * factor
        hr_temp = hr_temp[:new_h, :new_w]

        # Создаем LR версию через среднее по областям (физически корректно)
        lr_h, lr_w = new_h // scale_factor, new_w // scale_factor
        lr_temp = np.zeros((lr_h, lr_w), dtype=np.float32)

        for i in range(lr_h):
            for j in range(lr_w):
                # Берем среднее по области scale_factor x scale_factor
                region = hr_temp[i * scale_factor:(i + 1) * scale_factor,
                         j * scale_factor:(j + 1) * scale_factor]
                lr_temp[i, j] = np.mean(region)

        return lr_temp, hr_temp

'''
class TemperatureDataset(Dataset):
    """Dataset для температурных данных с инкрементальной загрузкой"""

    def __init__(self, npz_file: str, preprocessor: TemperatureDataPreprocessor,
                 scale_factor: int = 2, max_samples: Optional[int] = None):
        self.npz_file = npz_file
        self.preprocessor = preprocessor
        self.scale_factor = scale_factor
        self.max_samples = max_samples

        self.opt = {'name': 'TemperatureDataset'}

        # Загружаем данные
        print(f"Loading {npz_file}...")
        data = np.load(npz_file, allow_pickle=True)
        if 'swaths' in data:
            self.swaths = data['swaths']
        elif 'swath_array' in data:
            self.swaths = data['swath_array']
        else:
            print(f"Available keys in NPZ: {list(data.keys())}")
            raise KeyError(f"Neither 'swaths' nor 'swath_array' found in {npz_file}")

        # Подготавливаем пары LR-HR
        self.lr_hr_pairs = []
        self.metadata = []

        n_samples = len(self.swaths) if max_samples is None else min(len(self.swaths), max_samples)

        print(f"Preprocessing {n_samples} samples...")
        for i in range(n_samples):
            swath = self.swaths[i]
            temp = swath['temperature']
            meta = swath['metadata']

            # Предобработка
            temp = self.preprocessor.crop_or_pad(temp)

            # Сохраняем мин/макс для денормализации
            temp_min, temp_max = np.min(temp), np.max(temp)

            # Нормализация
            temp_norm = self.preprocessor.normalize_temperature(temp)

            # Создаем пару LR-HR
            lr, hr = self.preprocessor.create_lr_hr_pair(temp_norm, self.scale_factor)

            self.lr_hr_pairs.append((lr, hr))
            self.metadata.append({
                'original_min': temp_min,
                'original_max': temp_max,
                'orbit_type': meta.get('orbit_type', 'unknown'),
                'scale_factor': meta.get('scale_factor', 1.0)
            })

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{n_samples} samples")
                gc.collect()

        data.close()
        gc.collect()

    def __len__(self):
        return len(self.lr_hr_pairs)

    def __getitem__(self, idx):
        lr, hr = self.lr_hr_pairs[idx]
        meta = self.metadata[idx]

        # Конвертируем в тензоры и добавляем канальное измерение
        lr_tensor = torch.from_numpy(lr).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(hr).unsqueeze(0).float()

        return {
            'lq': lr_tensor,  # low quality (low resolution)
            'gt': hr_tensor,  # ground truth (high resolution)
            'metadata': meta
        }

'''


class TemperatureDataset(Dataset):
    """Dataset для температурных данных с ленивой загрузкой"""

    def __init__(self, npz_file: str, preprocessor: TemperatureDataPreprocessor,
                 scale_factor: int = 2, max_samples: Optional[int] = None):
        self.npz_file = npz_file
        self.preprocessor = preprocessor
        self.scale_factor = scale_factor
        self.max_samples = max_samples
        self.opt = {'name': 'TemperatureDataset'}

        # Только проверяем структуру файла, не загружаем данные
        print(f"Initializing dataset from {npz_file}...")
        with np.load(npz_file, allow_pickle=True) as data:
            if 'swaths' in data:
                self.data_key = 'swaths'
            elif 'swath_array' in data:
                self.data_key = 'swath_array'
            else:
                print(f"Available keys in NPZ: {list(data.keys())}")
                raise KeyError(f"Neither 'swaths' nor 'swath_array' found in {npz_file}")

            self.total_samples = len(data[self.data_key])

        self.n_samples = self.total_samples if max_samples is None else min(self.total_samples, max_samples)
        print(f"Dataset initialized with {self.n_samples} samples (lazy loading enabled)")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Загружаем только нужный элемент
        with np.load(self.npz_file, allow_pickle=True) as data:
            swath = data[self.data_key][idx]

        temp = swath['temperature']
        meta = swath['metadata']

        # Предобработка
        temp = self.preprocessor.crop_or_pad(temp)

        # Сохраняем мин/макс для денормализации
        temp_min, temp_max = np.min(temp), np.max(temp)

        # Нормализация
        temp_norm = self.preprocessor.normalize_temperature(temp)

        # Создаем пару LR-HR
        lr, hr = self.preprocessor.create_lr_hr_pair(temp_norm, self.scale_factor)

        # Конвертируем в тензоры и добавляем канальное измерение
        lr_tensor = torch.from_numpy(lr).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(hr).unsqueeze(0).float()

        return {
            'lq': lr_tensor,  # low quality (low resolution)
            'gt': hr_tensor,  # ground truth (high resolution)
            'metadata': {
                'original_min': temp_min,
                'original_max': temp_max,
                'orbit_type': meta.get('orbit_type', 'unknown'),
                'scale_factor': meta.get('scale_factor', 1.0)
            }
        }


class IncrementalDataLoader:
    """Загрузчик данных с инкрементальной загрузкой по файлам"""

    def __init__(self, npz_files: List[str], preprocessor: TemperatureDataPreprocessor,
                 batch_size: int = 4, scale_factor: int = 2,
                 samples_per_file: Optional[int] = None):
        self.npz_files = npz_files
        self.preprocessor = preprocessor
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.samples_per_file = samples_per_file
        self.current_file_idx = 0

    def get_next_dataloader(self) -> Optional[DataLoader]:
        """Получить DataLoader для следующего файла"""
        if self.current_file_idx >= len(self.npz_files):
            return None

        npz_file = self.npz_files[self.current_file_idx]
        dataset = TemperatureDataset(
            npz_file,
            self.preprocessor,
            self.scale_factor,
            self.samples_per_file
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )

        self.current_file_idx += 1
        return dataloader

    def reset(self):
        """Сброс для нового прохода по всем файлам"""
        self.current_file_idx = 0


def create_validation_set(npz_file: str, preprocessor: TemperatureDataPreprocessor,
                          n_samples: int = 100, scale_factor: int = 2) -> DataLoader:
    """Создание валидационного набора из отдельного файла"""
    dataset = TemperatureDataset(npz_file, preprocessor, scale_factor, max_samples=n_samples)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)


if __name__ == "__main__":
    # Тестирование
    preprocessor = TemperatureDataPreprocessor()

    # Тест нормализации
    test_temp = np.random.randn(2041, 421) * 50 + 273  # Кельвины
    norm_temp = preprocessor.normalize_temperature(test_temp)
    print(f"Original range: [{test_temp.min():.2f}, {test_temp.max():.2f}]")
    print(f"Normalized range: [{norm_temp.min():.2f}, {norm_temp.max():.2f}]")

    # Тест обрезки
    cropped = preprocessor.crop_or_pad(test_temp)
    print(f"Original shape: {test_temp.shape}, Cropped shape: {cropped.shape}")

    # Тест создания пар
    lr, hr = preprocessor.create_lr_hr_pair(cropped)
    print(f"LR shape: {lr.shape}, HR shape: {hr.shape}")