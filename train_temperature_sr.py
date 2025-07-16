#!/usr/bin/env python3
"""
Основной скрипт для обучения температурной Super-Resolution модели
с инкрементальной загрузкой данных
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
    parser = argparse.ArgumentParser(description='Train Temperature Super-Resolution Model')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing NPZ files')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                        help='Output directory for models and logs')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Total number of epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode with reduced data')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def setup_logger(opt):
    """Настройка логирования"""
    log_file = os.path.join(opt['path']['log'], f"train_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr',
                             log_level=logging.INFO,
                             log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    return logger


def create_dataloaders(args, opt, preprocessor, logger):
    """Создание загрузчиков данных"""
    # Получаем список NPZ файлов
    npz_files = sorted([os.path.join(args.data_dir, f)
                        for f in os.listdir(args.data_dir)
                        if f.endswith('.npz')])

    if args.debug:
        npz_files = npz_files[:2]  # Используем только 2 файла в debug режиме

    logger.info(f"Found {len(npz_files)} NPZ files")

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


def train_one_epoch(model, dataloader, current_iter, opt, logger, val_loader, epoch):
    """Обучение на одной эпохе"""
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
        # Улучшенное логирование с метриками
        if current_iter % opt['logger']['print_freq'] == 0:
            log_vars = model.get_current_log()
            message = f'[Эпоха: {epoch + 1:03d}] [Итер: {current_iter:07d}]'

            # Разделяем потери и метрики для лучшей читаемости
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

        # Clean GPU memory every 50 iterations
        if current_iter % 5 == 0:
            torch.cuda.empty_cache()

        # Сохранение модели
        if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
            logger.info('Saving models and training states.')
            model.save(epoch, current_iter)

        # Валидация
        if opt['val'] and current_iter % opt['val']['val_freq'] == 0:
            model.validation(val_loader, current_iter, None,
                             save_img=opt['val']['save_img'])

        train_data = prefetcher.next()

    return current_iter


def main():
    args = parse_args()

    # Создаем директории
    os.makedirs(args.output_dir, exist_ok=True)

    # Обновляем конфигурацию путями
    path['root'] = args.output_dir
    path['experiments_root'] = os.path.join(args.output_dir, name)
    path['models'] = os.path.join(path['experiments_root'], 'models')
    path['training_states'] = os.path.join(path['experiments_root'], 'training_states')
    path['log'] = os.path.join(path['experiments_root'], 'log')
    path['visualization'] = os.path.join(path['experiments_root'], 'visualization')

    # Создаем полную конфигурацию
    opt = {
        'name': name,
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
    try:
        make_exp_dirs(opt)
    except FileNotFoundError:
        # Create the directory if it doesn't exist
        os.makedirs(opt['path']['experiments_root'], exist_ok=True)

    # Настройка логгера
    logger = setup_logger(opt)

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
    logger.info('Creating dataloaders...')
    train_loader_manager, val_loader = create_dataloaders(args, opt, preprocessor, logger)

    # Создаем модель напрямую
    logger.info('Creating model...')
    # Регистрируем модель, если еще не зарегистрирована
    if 'TemperatureSRModel' not in MODEL_REGISTRY._obj_map:
        MODEL_REGISTRY.register(TemperatureSRModel)

    # Создаем экземпляр модели напрямую
    model = TemperatureSRModel(opt)

    # Возобновление обучения
    start_iter = 0
    if args.resume:
        logger.info(f'Resuming from {args.resume}')
        model.resume_training(args.resume)
        start_iter = model.begin_iter

    # Основной цикл обучения
    logger.info('Start training...')
    current_iter = start_iter
    total_epochs = args.num_epochs

    for epoch in range(total_epochs):
        logger.info(f'\n=== Epoch {epoch + 1}/{total_epochs} ===')

        # Инкрементальное обучение по файлам
        train_loader_manager.reset()
        file_idx = 0

        while True:
            # Получаем dataloader для следующего файла
            train_loader = train_loader_manager.get_next_dataloader()
            if train_loader is None:
                break

            logger.info(f'Training on file {file_idx + 1}/{len(train_loader_manager.npz_files)}')

            # Обучаем на текущем файле
            for file_epoch in range(incremental_training['epochs_per_file']):
                logger.info(f'  File epoch {file_epoch + 1}/{incremental_training["epochs_per_file"]}')
                current_iter = train_one_epoch(model, train_loader, current_iter, opt, logger, val_loader, epoch)

            # Сохраняем checkpoint после каждого файла
            if incremental_training['checkpoint_per_file']:
                logger.info(f'Saving checkpoint after file {file_idx + 1}')
                model.save(epoch, current_iter)

            # Уменьшаем learning rate между файлами
            if incremental_training['learning_rate_decay_per_file'] < 1.0:
                for optimizer in [model.optimizer_g, model.optimizer_d]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= incremental_training['learning_rate_decay_per_file']
                logger.info(f'Learning rate decayed to: {model.optimizer_g.param_groups[0]["lr"]:.2e}')

            file_idx += 1

            # Очистка памяти
            del train_loader
            gc.collect()
            torch.cuda.empty_cache()

        # Валидация в конце эпохи
        logger.info('Epoch validation...')
        model.validation(val_loader, current_iter, None, save_img=True)

        # Сохранение модели в конце эпохи
        logger.info(f'Saving models at epoch {epoch + 1}')
        model.save(epoch, current_iter)

    logger.info('Training completed!')

    # Финальное сохранение
    model.save(total_epochs - 1, current_iter)

    # Сохраняем статистику препроцессора
    stats_path = os.path.join(path['models'], 'preprocessor_stats.npz')
    np.savez(stats_path, **preprocessor.stats)
    logger.info(f'Preprocessor statistics saved to {stats_path}')


if __name__ == '__main__':
    main()