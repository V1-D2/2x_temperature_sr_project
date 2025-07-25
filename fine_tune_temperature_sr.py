#!/usr/bin/env python3
"""
Скрипт для дообучения (fine-tuning) предобученной температурной Super-Resolution модели
с сохранением оригинальной модели
"""

import argparse
import logging
import os
import random
import torch
import numpy as np
from datetime import datetime
from collections import OrderedDict
import gc
import shutil

from basicsr.utils import (get_time_str, get_root_logger, get_env_info,
                           make_exp_dirs, set_random_seed, tensor2img)
from basicsr.utils.options import dict2str
from basicsr.data.prefetch_dataloader import CPUPrefetcher
from basicsr.utils.registry import MODEL_REGISTRY

# Импортируем наши модули
from data_preprocessing import (TemperatureDataPreprocessor,
                                IncrementalDataLoader,
                                create_validation_set)
from hybrid_model import TemperatureSRModel
from config_temperature import (
    name, model_type, scale, num_gpu, datasets, network_g, network_d,
    path, train, val, logger as logger_config, dist_params,
    temperature_specific, incremental_training
)


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Temperature Super-Resolution Model')
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing NPZ files')
    parser.add_argument('--output_dir', type=str, default='./experiments_finetune',
                        help='Output directory for fine-tuned models')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of additional epochs for fine-tuning')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate for fine-tuning (default: 5e-5)')
    parser.add_argument('--save_original', action='store_true', default=True,
                        help='Save a backup of the original model')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Custom experiment name for fine-tuning')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode with reduced data')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def backup_original_model(model_path, backup_dir):
    """Создает резервную копию оригинальной модели"""
    os.makedirs(backup_dir, exist_ok=True)

    # Создаем имя для бэкапа с временной меткой
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = os.path.basename(model_path)
    backup_name = f"backup_{timestamp}_{model_name}"
    backup_path = os.path.join(backup_dir, backup_name)

    # Копируем файл
    shutil.copy2(model_path, backup_path)

    # Также копируем информацию об итерации если есть
    iter_file = model_path.replace('.pth', '_iter.txt')
    if os.path.exists(iter_file):
        backup_iter_file = backup_path.replace('.pth', '_iter.txt')
        shutil.copy2(iter_file, backup_iter_file)

    return backup_path


def load_pretrained_checkpoint(model_path):
    """Загружает checkpoint и извлекает информацию о предыдущем обучении"""
    checkpoint = torch.load(model_path, map_location='cpu')

    # Определяем формат checkpoint
    if isinstance(checkpoint, dict):
        if 'params' in checkpoint:
            # Полный checkpoint с информацией о тренировке
            state_dict = checkpoint['params']
            iter_info = checkpoint.get('iter', 0)
            optimizer_state = checkpoint.get('optimizers', None)
            schedulers_state = checkpoint.get('schedulers', None)
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            iter_info = checkpoint.get('iter', 0)
            optimizer_state = checkpoint.get('optimizers', None)
            schedulers_state = checkpoint.get('schedulers', None)
        else:
            # Неизвестный формат словаря
            state_dict = checkpoint
            iter_info = 0
            optimizer_state = None
            schedulers_state = None
    else:
        # Просто state dict
        state_dict = checkpoint
        iter_info = 0
        optimizer_state = None
        schedulers_state = None

    return state_dict, iter_info, optimizer_state, schedulers_state


def setup_logger(opt):
    """Настройка логирования для fine-tuning"""
    log_file = os.path.join(opt['path']['log'], f"finetune_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr',
                             log_level=logging.INFO,
                             log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    return logger


def create_dataloaders(args, opt, preprocessor, logger):
    """Создание загрузчиков данных для fine-tuning"""
    # Получаем список NPZ файлов
    npz_files = sorted([os.path.join(args.data_dir, f)
                        for f in os.listdir(args.data_dir)
                        if f.endswith('.npz')])

    if args.debug:
        npz_files = npz_files[:2]  # Используем только 2 файла в debug режиме

    logger.info(f"Found {len(npz_files)} NPZ files for fine-tuning")

    # Разделяем на train и validation
    val_file = npz_files[-1]
    train_files = npz_files[:-1]

    # Создаем инкрементальный загрузчик для обучения
    train_loader = IncrementalDataLoader(
        train_files,
        preprocessor,
        batch_size=opt['datasets']['train']['batch_size'],
        scale_factor=opt['datasets']['train']['scale_factor'],
        samples_per_file=opt['datasets']['train']['samples_per_file'] if not args.debug else 100
    )

    # Создаем валидационный датасет
    val_loader = create_validation_set(
        val_file,
        preprocessor,
        n_samples=opt['datasets']['val']['n_samples'] if not args.debug else 10,
        scale_factor=opt['datasets']['val']['scale_factor']
    )

    return train_loader, val_loader


def train_one_epoch(model, dataloader, current_iter, opt, logger, val_loader, epoch, total_epochs):
    """Обучение на одной эпохе для fine-tuning"""
    model.net_g.train()
    if hasattr(model, 'net_d'):
        model.net_d.train()

    prefetcher = CPUPrefetcher(dataloader)
    train_data = prefetcher.next()

    while train_data is not None:
        current_iter += 1

        # Обучение модели
        model.feed_data(train_data)
        model.optimize_parameters(current_iter)

        # Обновление learning rate
        model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

        # Логирование
        if current_iter % opt['logger']['print_freq'] == 0:
            log_vars = model.get_current_log()
            message = f'[Fine-tuning] [Эпоха: {epoch + 1}/{total_epochs}] [Итер: {current_iter:08d}]'

            # Разделяем потери и метрики
            losses = {}
            metrics = {}

            for k, v in log_vars.items():
                if k in ['psnr', 'ssim']:
                    metrics[k] = v
                else:
                    losses[k] = v

            # Логируем потери
            if losses:
                loss_msg = ' | Потери: '
                for k, v in losses.items():
                    if k == 'l_g_pix':
                        loss_msg += f'Пиксель: {v:.4e} '
                    elif k == 'l_g_percep':
                        loss_msg += f'Перцепт: {v:.4e} '
                    elif k == 'l_g_gan':
                        loss_msg += f'GAN_G: {v:.4e} '
                    elif k == 'l_d_real':
                        loss_msg += f'D_Real: {v:.4e} '
                    elif k == 'l_d_fake':
                        loss_msg += f'D_Fake: {v:.4e} '
                    else:
                        loss_msg += f'{k}: {v:.4e} '
                message += loss_msg

            # Логируем метрики
            if metrics:
                metric_msg = ' | Метрики: '
                for k, v in metrics.items():
                    if k == 'psnr':
                        metric_msg += f'PSNR: {v:.2f}dB '
                    elif k == 'ssim':
                        metric_msg += f'SSIM: {v:.4f} '
                    else:
                        metric_msg += f'{k}: {v:.4f} '
                message += metric_msg

            logger.info(message)

        # Очистка памяти GPU
        if current_iter % 10 == 0:
            torch.cuda.empty_cache()

        # Сохранение модели
        if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
            logger.info('Saving fine-tuned models and training states.')
            model.save(epoch, current_iter)

        # Валидация
        if opt['val'] and current_iter % opt['val']['val_freq'] == 0:
            model.validation(val_loader, current_iter, None,
                             save_img=opt['val']['save_img'])

        train_data = prefetcher.next()

    return current_iter


def main():
    args = parse_args()

    # Проверяем существование pretrained модели
    if not os.path.exists(args.pretrained_model):
        raise FileNotFoundError(f"Pretrained model not found: {args.pretrained_model}")

    # Создаем директории для fine-tuning
    os.makedirs(args.output_dir, exist_ok=True)

    # Создаем имя эксперимента для fine-tuning
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"TemperatureSR_FineTune_{timestamp}"

    # Обновляем конфигурацию путями
    path['root'] = args.output_dir
    path['experiments_root'] = os.path.join(args.output_dir, experiment_name)
    path['models'] = os.path.join(path['experiments_root'], 'models')
    path['training_states'] = os.path.join(path['experiments_root'], 'training_states')
    path['log'] = os.path.join(path['experiments_root'], 'log')
    path['visualization'] = os.path.join(path['experiments_root'], 'visualization')

    # Создаем директорию для бэкапов
    backup_dir = os.path.join(path['experiments_root'], 'original_model_backup')

    # Обновляем learning rate для fine-tuning
    train['optim_g']['lr'] = args.learning_rate
    train['optim_d']['lr'] = args.learning_rate * 0.5  # Дискриминатор с меньшим lr

    # Отключаем warmup для fine-tuning
    train['warmup_iter'] = 0

    # Создаем полную конфигурацию
    opt = {
        'name': experiment_name,
        'model_type': model_type,
        'scale': scale,
        'num_gpu': num_gpu,
        'manual_seed': train['manual_seed'],
        'datasets': datasets,
        'network_g': network_g,
        'network_d': network_d,
        'path': path,
        'train': train,
        'val': val,
        'logger': logger_config,
        'dist_params': dist_params,
        'temperature_specific': temperature_specific,
        'incremental_training': incremental_training,
        'is_train': True
    }

    # Инициализация distributed training
    if args.launcher == 'none':
        opt['dist'] = False
    else:
        opt['dist'] = True
        if args.launcher == 'pytorch':
            torch.distributed.init_process_group(backend='nccl')

    # Создаем директории
    make_exp_dirs(opt)

    # Настройка логгера
    logger = setup_logger(opt)
    logger.info(f"Fine-tuning experiment: {experiment_name}")
    logger.info(f"Loading pretrained model from: {args.pretrained_model}")

    # Создаем резервную копию оригинальной модели
    if args.save_original:
        backup_path = backup_original_model(args.pretrained_model, backup_dir)
        logger.info(f"Original model backed up to: {backup_path}")

    # Установка random seed
    seed = opt.get('manual_seed', None)
    if seed is None:
        seed = random.randint(1, 10000)
    set_random_seed(seed)
    logger.info(f'Random seed: {seed}')

    # Создаем препроцессор
    preprocessor = TemperatureDataPreprocessor(
        target_height=datasets['train']['preprocessor_args']['target_height'],
        target_width=datasets['train']['preprocessor_args']['target_width']
    )

    # Создаем загрузчики данных
    logger.info('Creating dataloaders for fine-tuning...')
    train_loader_manager, val_loader = create_dataloaders(args, opt, preprocessor, logger)

    # Создаем модель
    logger.info('Creating model for fine-tuning...')
    if 'TemperatureSRModel' not in MODEL_REGISTRY._obj_map:
        MODEL_REGISTRY.register(TemperatureSRModel)

    model = TemperatureSRModel(opt)

    # Загружаем pretrained веса
    logger.info('Loading pretrained weights...')
    state_dict, pretrained_iter, optimizer_state, schedulers_state = load_pretrained_checkpoint(args.pretrained_model)

    # Загружаем веса генератора
    model.net_g.load_state_dict(state_dict, strict=True)
    logger.info(f"Loaded generator weights from iteration: {pretrained_iter}")

    # Пытаемся загрузить веса дискриминатора если есть
    discriminator_path = args.pretrained_model.replace('net_g_', 'net_d_')
    if os.path.exists(discriminator_path):
        try:
            d_checkpoint = torch.load(discriminator_path, map_location='cpu')
            if isinstance(d_checkpoint, dict) and 'params' in d_checkpoint:
                model.net_d.load_state_dict(d_checkpoint['params'], strict=True)
            else:
                model.net_d.load_state_dict(d_checkpoint, strict=True)
            logger.info("Loaded discriminator weights")
        except Exception as e:
            logger.warning(f"Could not load discriminator weights: {e}")
            logger.info("Starting with fresh discriminator")
    else:
        logger.info("No discriminator checkpoint found, starting with fresh discriminator")

    # Восстанавливаем состояние оптимизаторов если есть
    if optimizer_state is not None:
        try:
            model.optimizers[0].load_state_dict(optimizer_state[0])  # generator optimizer
            if len(optimizer_state) > 1 and len(model.optimizers) > 1:
                model.optimizers[1].load_state_dict(optimizer_state[1])  # discriminator optimizer
            logger.info("Restored optimizer states")
        except Exception as e:
            logger.warning(f"Could not restore optimizer states: {e}")
            logger.info("Starting with fresh optimizers")

    # Основной цикл fine-tuning
    logger.info('Starting fine-tuning...')
    current_iter = pretrained_iter  # Продолжаем с предыдущей итерации
    start_epoch = 0
    total_epochs = args.num_epochs

    for epoch in range(start_epoch, total_epochs):
        logger.info(f'\n=== Fine-tuning Epoch {epoch + 1}/{total_epochs} ===')

        # Инкрементальное обучение по файлам
        train_loader_manager.reset()
        file_idx = 0

        while True:
            # Получаем dataloader для следующего файла
            train_loader = train_loader_manager.get_next_dataloader()
            if train_loader is None:
                break

            logger.info(f'Fine-tuning on file {file_idx + 1}/{len(train_loader_manager.npz_files)}')

            # Обучаем на текущем файле
            for file_epoch in range(incremental_training['epochs_per_file']):
                logger.info(f'  File epoch {file_epoch + 1}/{incremental_training["epochs_per_file"]}')
                current_iter = train_one_epoch(
                    model, train_loader, current_iter, opt, logger,
                    val_loader, epoch, total_epochs
                )

            # Сохраняем checkpoint после каждого файла при fine-tuning
            logger.info(f'Saving checkpoint after file {file_idx + 1}')
            model.save(epoch, current_iter)

            file_idx += 1

            # Очистка памяти
            del train_loader
            gc.collect()
            torch.cuda.empty_cache()

        # Валидация в конце эпохи
        logger.info('Epoch validation...')
        model.validation(val_loader, current_iter, None, save_img=True)

        # Сохранение модели в конце эпохи
        logger.info(f'Saving fine-tuned models at epoch {epoch + 1}')
        model.save(epoch, current_iter)

    logger.info('Fine-tuning completed!')

    # Финальное сохранение
    model.save(total_epochs - 1, current_iter)

    # Сохраняем информацию о fine-tuning
    finetune_info = {
        'pretrained_model': args.pretrained_model,
        'pretrained_iter': pretrained_iter,
        'finetune_start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'finetune_epochs': total_epochs,
        'finetune_final_iter': current_iter,
        'learning_rate': args.learning_rate,
        'data_dir': args.data_dir,
        'experiment_name': experiment_name
    }

    info_path = os.path.join(path['models'], 'finetune_info.txt')
    with open(info_path, 'w') as f:
        for key, value in finetune_info.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"Fine-tuning information saved to: {info_path}")
    logger.info(f"Fine-tuned models saved in: {path['models']}")


if __name__ == '__main__':
    main()